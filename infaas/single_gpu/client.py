"""CPU-only operator client. Doctor is the only command that accesses the GPU."""
import argparse
import json
from pathlib import Path
import time

import grpc
from infaas.proto import modelreg_pb2 as reg, modelreg_pb2_grpc
from infaas.proto import queryfe_pb2 as pb, queryfe_pb2_grpc, request_reply_pb2 as rr


def status(channel, timeout=10):
    call = channel.unary_unary("/infaas.single_gpu.Admin/Status",
        request_serializer=lambda value: json.dumps(value).encode(),
        response_deserializer=json.loads)
    return call({}, timeout=timeout)


def compact(state):
    fields = ("implementation", "application", "objective", "policy_ready", "policy_error", "enabled_models",
        "resident_model", "plan", "plan_details", "model_switches_after_initial_load", "batch_size_counts",
        "successful_requests_by_model", "queue_depth", "outstanding_requests", "busy", "maintenance",
        "processed", "failed", "deadline_drops", "server_slo_misses", "profile_scope", "preprocess_executor")
    result = {key: state[key] for key in fields}
    result["profile_summary"] = {model: {batch: {key: round(row[key], 3)
        for key in ("p50_ms", "p95_ms", "capacity_qps")} for batch, row in profile["batches"].items()}
        for model, profile in state["profiles"].items()}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="127.0.0.1:50052")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("state").add_argument("--full", action="store_true")
    subs.add_parser("health")
    subs.add_parser("doctor")
    registration = subs.add_parser("register")
    registration.add_argument("--image", required=True)
    registration.add_argument("--models", nargs="+")
    registration.add_argument("--reprofile", action="store_true")
    query = subs.add_parser("query")
    query.add_argument("--image", required=True)
    query.add_argument("--slo", type=float, default=200)
    query.add_argument("-n", type=int, default=11)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .settings import Settings
        from .backend import Backend
        settings = Settings()
        import os
        os.environ["MODEL_REVISIONS"] = ",".join(f"{name}:{entry['revision']}" for name, entry in settings.entries.items())
        backend = Backend(settings)
        import torch
        value = (torch.ones((2, 2), device="cuda") @ torch.ones((2, 2), device="cuda"))[0, 0].item()
        print(json.dumps({"ok": value == 2, "gpu": torch.cuda.get_device_name(0),
                          "fingerprint": backend.fingerprint}, indent=2))
        return 0 if value == 2 else 1
    with grpc.insecure_channel(args.controller) as channel:
        grpc.channel_ready_future(channel).result(timeout=10)
        before = status(channel)
        if args.command == "health":
            return 0
        if args.command == "state":
            print(json.dumps(before if args.full else compact(before), indent=2))
            return 0
        if args.command == "register":
            names = [name for part in args.models for name in part.split(",")] if args.models else before["enabled_models"]
            stub = modelreg_pb2_grpc.ModelRegStub(channel)
            image = Path(args.image).read_bytes()
            for name in names:
                reply = stub.RegisterModel(reg.ModelRegRequest(parent_model=name, profile_image=image,
                                                               reprofile=args.reprofile), timeout=600)
                if reply.status.status != rr.SUCCESS:
                    raise RuntimeError(f"Register {name}: {reply.status.msg}")
                print(f"registered {name}: {list(reply.variants)}", flush=True)
            after = status(channel)
            print(f"policy_ready={after['policy_ready']} enabled_models={after['enabled_models']}")
            return 0
        stub = queryfe_pb2_grpc.QueryStub(channel)
        image = Path(args.image).read_bytes()
        for _ in range(args.n):
            start = time.perf_counter()
            reply = stub.QueryOnline(pb.QueryOnlineRequest(parent_model=before["application"], raw_input=[image],
                slo=pb.QuerySLO(LatencyInUSec=int(args.slo * 1000))), timeout=30)
            elapsed = (time.perf_counter() - start) * 1000
            ok = reply.status.status == rr.SUCCESS
            print(f"{'OK' if ok else 'REJ'} e2e={elapsed:.2f}ms variant={reply.variant} "
                  f"batch={int(reply.timings.get('worker.batch_size', 0))} "
                  f"load={reply.timings.get('worker.load', 0):.2f}ms "
                  f"slo={'PASS' if ok and elapsed <= args.slo else 'MISS'} {reply.status.msg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
