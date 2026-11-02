"""Deploy the unchanged user-provided INFaaS docker/v1 stack on one V100."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.resnet import host as common


def settings():
    c = common.settings("infaas")
    if c["EXPERIMENT_PHASE"] != "v1":
        raise ValueError("This source-preserving deployment is v1 only")
    c.update(PROJECT="source-infaas-v1",WORKER_IMAGE="infaas-docker-v1-source:v1")
    c["RUN_DIR"] = str(Path(c["DATA_DIR"])/"infaas-docker-v1"/"v1")
    c["MODEL_DIR"] = str(Path(c["MODEL_DIR"])/"infaas-hf")
    if c["REQUEST_MODEL"] not in json.loads(Path(__file__).with_name("infaas_models.json").read_text()):
        raise ValueError("REQUEST_MODEL must be one of the five registered ResNets")
    return c


def environment(c):
    # Defaults below are copied from docker/v1. GPU type, paths and endpoints are deployment settings.
    return dict(HW_TYPES="v100",HW_COST="v100:1",REDIS_HOST="redis",REDIS_PORT="16379",
        CONTROLLER_ADDR="controller:50054",ORCHESTRATOR="static",
        STATIC_WORKERS="source-worker=v100@worker:9000",MAX_WORKERS="v100:1",
        WORKER_NAME="source-worker",GPU_TYPE="v100",WORKER_PORT="9000",
        PREPROCESS_BACKEND="auto",PREPROCESS_MODE="cpu_decode_gpu",
        DECISION_MODE="6",AUTOSCALER_TYPE="infaas",LAMBDA="1.0",SLACK_THRESHOLD="1.05",
        VM_UTIL_THRESHOLD="80",MONITOR_INTERVAL_S="2.0",MODEL_AUTOSCALER_INTERVAL_S="1.0",
        VM_AUTOSCALER_INTERVAL_S="2.0",EXEC_BLACKLIST_UTIL_ENABLED="false",
        EXEC_BLACKLIST_QPS_ENABLED="false",PROFILE_MODE="original",PROFILE_RUNS=c["PROFILE_TRIALS"],
        PROFILE_SAT_SECONDS="5",PROFILE_SAT_CONCURRENCY="4",WORKER_MODE="static",
        PROFILE_DIR="/profiles",REQUEST_LOG="/logs/requests.jsonl",EVENT_LOG="/logs/events.jsonl",
        HF_HOME="/models",HF_HUB_OFFLINE="1",SOURCE_REQUEST_MODEL=c["REQUEST_MODEL"])


def volumes(c):
    return [{"type":"bind","source":c["MODEL_DIR"],"target":"/models"},
        {"type":"bind","source":str(Path(c["RUN_DIR"])/"profiles"),"target":"/profiles"},
        {"type":"bind","source":str(Path(c["RUN_DIR"])/"logs"),"target":"/logs"},
        {"type":"bind","source":str(Path(c["RUN_DIR"])/"results"),"target":"/results"},
        {"type":"bind","source":c["INPUT_IMAGE"],"target":"/inputs/image.jpg","read_only":True}]


def spec(c,user="0:0"):
    base=dict(image=c["WORKER_IMAGE"],user=user,environment=environment(c),volumes=volumes(c))
    entry=["python","-m","experiments.reproduce.infaas_entry"]
    redis={"image":c["WORKER_IMAGE"],"user":user,
        "command":["redis-server","--port","16379","--save","","--appendonly","no"],
        "healthcheck":{"test":["CMD","redis-cli","-p","16379","ping"],"interval":"2s","timeout":"2s","retries":30}}
    depends={"redis":{"condition":"service_healthy"}}
    worker=dict(base,command=entry+["infaas.worker.main"],depends_on=depends,shm_size="1g",
        deploy={"resources":{"reservations":{"devices":[{"driver":"nvidia","device_ids":[c["GPU"]],"capabilities":["gpu"]}]}}})
    controller=dict(base,command=entry+["infaas.controller.main"],depends_on=depends,
        ports=[f"127.0.0.1:{c['HOST_PORT']}:50052"])
    scaler=dict(base,command=entry+["infaas.controller.vm_autoscaler"],depends_on=depends)
    return {"services":{"redis":redis,"worker":worker,"controller":controller,"vm-autoscaler":scaler}}


def run_client(c,module,args=(),network=True,online=False,capture=False):
    info=common.docker_info(c)
    build=common.verify_build(c)
    cmd=["docker","run","--rm","--user",common.docker_user(info),"--network",
         c["PROJECT"]+"_default" if network else "none"]
    env=environment(c)
    env.update(ENGINE="infaas",EXPERIMENT_PHASE="v1",SLO_MS=c["SLO_MS"],
        TRACE_SPEED=c["TRACE_SPEED"],TRACE_CONCURRENCY=c["TRACE_CONCURRENCY"],
        RPC_TIMEOUT_SECONDS=c["RPC_TIMEOUT_SECONDS"],START_STATE=c["START_STATE"],
        SOURCE_COMMIT=build["source_commit"],SOURCE_IMAGE_ID=build["images"][c["WORKER_IMAGE"]])
    if online: env["HF_HUB_OFFLINE"]="0"
    for k,v in env.items(): cmd+=["-e",f"{k}={v}"]
    for v in volumes(c):
        cmd+=["--mount",f"type=bind,src={v['source']},dst={v['target']}"+(",readonly" if v.get("read_only") else "")]
    trace=Path(c["MODEL_DIR"]).parent/("trace-"+c["WORKLOAD"]+".txt")
    cmd+=["--mount",f"type=bind,src={trace},dst=/data/trace.txt,readonly",c["WORKER_IMAGE"],
          "python","-m","experiments.reproduce.infaas_entry",module,*args]
    return common.run(cmd,capture)


def main(action,extra=()):
    c=settings()
    if action=="settings":
        show={k:c[k] for k in ("GPU","GPU_TYPE","EXPECTED_DOCKER_ROOT","REQUEST_MODEL","MODEL_DIR","DATA_DIR",
            "RUN_DIR","INPUT_IMAGE","SLO_MS","PROFILE_TRIALS","WORKLOAD","TRACE_SPEED","TRACE_CONCURRENCY",
            "RPC_TIMEOUT_SECONDS","HOST_PORT","WORKER_IMAGE","PROJECT")}
        show.update(BASELINE="provided docker/v1@baa65befb0ee551d589926a6d44f7343072ec9eb",
                    REQUEST_BATCH_SIZE=1,NATIVE_ENVIRONMENT=environment(c))
        print(json.dumps(show,indent=2));return
    if action=="compose": print(json.dumps(spec(c),indent=2)); return
    info=common.docker_info(c)
    if action=="build":
        common.run(["docker","build","-t",c["WORKER_IMAGE"],"-f","experiments/reproduce/infaas.Dockerfile","."])
        p=common.manifest_path(c);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({"source_sha256":common.source_fingerprint(c),
            "source_commit":common.run(["git","rev-parse","HEAD"],True),
            "images":{c["WORKER_IMAGE"]:common.run(["docker","image","inspect","--format","{{.Id}}",c["WORKER_IMAGE"]],True)}}))
        return
    if action=="down":
        p=Path(c["RUN_DIR"])/"compose.json"
        if p.exists(): common.compose(c,"down")
        return
    if action=="logs":return common.compose(c,"logs","--tail","100",*extra)
    common.verify_build(c)
    for name in ("profiles","logs","results"): (Path(c["RUN_DIR"])/name).mkdir(parents=True,exist_ok=True)
    Path(c["MODEL_DIR"]).mkdir(parents=True,exist_ok=True)
    if action=="setup":
        from experiments.resnet import assets
        import hashlib,urllib.request
        parent=Path(c["MODEL_DIR"]).parent
        for name,v in json.loads((common.HERE/"traces.json").read_text()).items():
            p=parent/("trace-"+name+".txt")
            if not p.exists():
                raw=urllib.request.urlopen(v["url"],timeout=60).read()
                if hashlib.sha256(raw).hexdigest()!=v["sha256"]: raise ValueError("Trace checksum differs")
                p.write_bytes(raw)
            if hashlib.sha256(p.read_bytes()).hexdigest()!=v["sha256"]: raise ValueError("Trace checksum differs")
        image=Path(c["INPUT_IMAGE"])
        if not image.exists():
            image.parent.mkdir(parents=True,exist_ok=True)
            raw=urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",timeout=60).read()
            if hashlib.sha256(raw).hexdigest()!="f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a":raise ValueError("JPEG changed")
            image.write_bytes(raw)
        run_client(c,"download",online=True)
        return
    if action=="up":
        if common.run(["docker","ps","-q","--filter","label=com.docker.compose.project="+c["PROJECT"]],True):
            raise ValueError("This experiment is already running")
        if common.run(["nvidia-smi","-i",c["GPU"],"--query-compute-apps=pid","--format=csv,noheader"],True):
            raise ValueError("GPU occupied; no existing experiment was stopped")
        if "V100" not in common.run(["nvidia-smi","-i",c["GPU"],"--query-gpu=name","--format=csv,noheader"],True):
            raise ValueError("Select the V100")
        (Path(c["RUN_DIR"])/"compose.json").write_text(json.dumps(spec(c,common.docker_user(info)),indent=2))
        common.compose(c,"up","-d")
        for attempt in range(90):
            try:run_client(c,"experiments.reproduce.infaas_trace",["health"],capture=True);break
            except subprocess.CalledProcessError:
                if attempt==89:raise
                time.sleep(2)
        return
    if action=="register":
        names=",".join(json.loads(Path(__file__).with_name("infaas_models.json").read_text()))
        return run_client(c,"infaas.cli.register",["--controller","controller:50053","--models",names,"--hardware","v100","--reprofile",*extra])
    if action in ("trace","query","state"):
        return run_client(c,"experiments.reproduce.infaas_trace",[action,*extra],
            network=not(action=="trace" and "--dry-run" in extra))
    raise ValueError("Unsupported source-preserving INFaaS action: "+action)
