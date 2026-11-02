"""Metrics only: call the provided docker/v1 API; never select a model or batch."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from experiments.resnet.client import Client as SharedClient,replay,summary,read_trace,percentile
from .infaas_entry import verify,CATALOGUE
from .metrics import add_boundaries

MODELS={"resnet-"+str(n):"resnet"+str(n)+"_v1" for n in (18,34,50,101,152)}


def metadata():
    from infaas.metadata.redis_metadata import RedisMetadata
    md=RedisMetadata()
    workers={}
    for name in md.get_all_executors():
        workers[name]={"info":md.get_executor_info(name),"variants":md.get_variants_on_executor(name)}
    registered={name:md.get_model_info(name+"__v100") for name in MODELS}
    return {"source":verify(),"workers":workers,"registered_profiles":registered,
            "request_model":os.environ["SOURCE_REQUEST_MODEL"],
            "policy":"docker/v1 controller/worker/autoscalers, unchanged","batch":1,
            "profile_mode":os.environ["PROFILE_MODE"],"preprocessing":"provided docker/v1 DALI/PIL path retained"}


class Client(SharedClient):
    def __init__(self,timeout):
        import grpc
        from infaas.proto import queryfe_pb2,queryfe_pb2_grpc,request_reply_pb2
        self.timeout=timeout
        self.pb=queryfe_pb2
        self.success=request_reply_pb2.SUCCESS
        self.channel=grpc.insecure_channel("controller:50052",options=[
            ("grpc.max_receive_message_length",32<<20),("grpc.max_send_message_length",32<<20)])
        grpc.channel_ready_future(self.channel).result(timeout=timeout)
        self.stub=queryfe_pb2_grpc.QueryStub(self.channel)

    def query(self,raw,slo,request_id):
        request=self.pb.QueryOnlineRequest(parent_model=os.environ["SOURCE_REQUEST_MODEL"],raw_input=[raw])
        request.slo.LatencyInUSec=int(slo*1000)
        start=time.perf_counter()
        row={"sent":True,"ok":False,"model":"","batch_size":1,"server_ms":None,
             "timings":{},"reject_kind":"","error":"","rpc_started":start}
        try:
            response=self.stub.QueryOnline(request,timeout=self.timeout)
            t=dict(response.timings)
            name=response.variant.split("__",1)[0]
            # Observational decomposition already supplied by docker/v1.
            keys=("worker.lock_wait","worker.forward","worker.postprocess")
            common=sum(t[k] for k in keys) if all(k in t for k in keys) else None
            if common is not None:t["measurement.post_preprocess"]=common
            t["measurement.native_api_latency"]=response.latency_ms
            row.update(ok=response.status.status==self.success,model=MODELS.get(name,name),
                server_ms=common,timings=t,reject_kind=response.reject_kind,error=response.status.msg)
        except Exception as error:
            row.update(reject_kind=type(error).__name__,error=str(error))
        row["e2e_ms"]=(time.perf_counter()-start)*1000
        row["slo_met"]=bool(row["ok"] and row["e2e_ms"]<=slo)
        return add_boundaries(row)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("health","state","query","trace"))
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--speed",type=float,default=float(os.environ["TRACE_SPEED"]))
    ap.add_argument("--concurrency",type=int,default=int(os.environ["TRACE_CONCURRENCY"]))
    ap.add_argument("--timeout",type=float,default=float(os.environ["RPC_TIMEOUT_SECONDS"]))
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()
    if args.speed<=0 or args.timeout<=0 or not 1<=args.concurrency<=1024:ap.error("Invalid replay settings")
    name=os.environ["SOURCE_REQUEST_MODEL"]
    if name not in MODELS:ap.error("SOURCE_REQUEST_MODEL must be one configured ResNet")
    if args.action=="state":print(json.dumps(metadata(),indent=2));return
    if args.action=="health":
        from grpc_health.v1 import health_pb2,health_pb2_grpc
        import grpc
        for address in ("controller:50052","worker:9000"):
            with grpc.insecure_channel(address) as channel:
                reply=health_pb2_grpc.HealthStub(channel).Check(health_pb2.HealthCheckRequest(),timeout=5)
                if reply.status!=health_pb2.HealthCheckResponse.SERVING:raise RuntimeError("Not healthy: "+address)
        if not metadata()["workers"]:raise RuntimeError("Native VM autoscaler has not registered the worker")
        return
    image=Path("/inputs/image.jpg").read_bytes()
    slo=float(os.environ["SLO_MS"])
    arrivals=read_trace("/data/trace.txt",args.limit) if args.action=="trace" else []
    duration=arrivals[-1][0]/1000/args.speed if arrivals else 0
    import importlib.metadata
    config={"implementation":"provided-infaas-docker-v1","source":verify(),"selected_requests":len(arrivals),
        "checkpoint_revisions":json.loads(CATALOGUE.read_text()),
        "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","pillow","nvidia-dali-cuda120")},
        "request_model":name,"speed":args.speed,"planned_duration_seconds":duration,"slo_ms":slo,
        "trace_sha256":hashlib.sha256(Path("/data/trace.txt").read_bytes()).hexdigest(),
        "input_sha256":hashlib.sha256(image).hexdigest(),"concurrency":args.concurrency,"timeout":args.timeout,
        "source_image":os.environ["SOURCE_IMAGE_ID"],"adapter_commit":os.environ["SOURCE_COMMIT"],
        "profile_mode":"original","preprocessing":"retained, not bypassed",
        "common_boundary":"worker preprocessing complete -> GPU queue/forward/postprocess complete",
        "boundary_note":"Native load occurs before preprocessing; load is a separate native timing, not silently added to post-preprocess time",
        "slo_note":"Native deadlines/selection are unchanged. Client E2E SLO and measured post-preprocess time are separate.",
        "start_state":"native state recorded below; no policy reset or synthetic warmup"}
    if args.dry_run:print(json.dumps(config,indent=2));return
    client=Client(args.timeout)
    if args.action=="query":
        row=client.query(image,slo,uuid.uuid4().hex);row.pop("rpc_started",None)
        print(json.dumps(row,indent=2));return
    before=metadata()
    if not before["registered_profiles"].get(name):raise RuntimeError("Register this model before replay")
    started_epoch=time.time()
    rows,elapsed=replay(client,image,arrivals,slo,args.speed,args.concurrency)
    finished_epoch=time.time()
    result=summary(rows,duration,slo)
    result["elapsed_seconds_including_drain"]=elapsed
    result["worker_reported_interval"]["boundary_note"]=config["common_boundary"]
    result["native_policy_preserved"]=True
    event_file=Path("/logs/events.jsonl")
    events=[]
    if event_file.exists():
        for line in event_file.read_text().splitlines():
            try:event=json.loads(line)
            except json.JSONDecodeError:continue
            if started_epoch<=event.get("ts",0)<=finished_epoch:events.append(event)
    result["observed_native_load_events"]=[e for e in events if e.get("kind")=="load"]
    result["observed_native_unload_events"]=[e for e in events if e.get("kind")=="unload"]
    result["observed_total_load_ms"]=sum(e.get("load_ms",0) for e in result["observed_native_load_events"])
    result["event_log_note"]="Original asynchronous logger snapshot; raw log retained, a just-finished event may arrive later"
    token=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output=Path("/results")/("trace-"+name+"-"+token)
    output.with_suffix(".json").write_text(json.dumps({"config":config,"summary":result,
        "worker_before":before,"worker_after":metadata(),"native_events_snapshot":events,
        "requests":rows},indent=2,allow_nan=False))
    import csv
    fields=["index","arrival_ms","qos","source_slo_ms","sent","ok","slo_met","scheduled_slo_met",
            "e2e_ms","server_ms","send_lag_ms","model","batch_size","reject_kind","error","timings"]
    with output.with_suffix(".csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for row in rows:writer.writerow({**row,"timings":json.dumps(row["timings"])})
    print(json.dumps(result,indent=2))
    print("Results:",output.with_suffix(".json"),output.with_suffix(".csv"))


if __name__=="__main__":
    main()
