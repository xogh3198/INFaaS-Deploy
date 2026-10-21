#!/usr/bin/env python3
"""CPU-only client for INFaaS QueryOnline; never changes the serving policy.

The common scheduler preserves the previous Proteus experiment's arrival-time
and metric definitions. No Proteus runtime, GPU access, reset or warmup is used.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from infaas.common.trace_replay import (
    count, empty_reply, percentile, positive, read_accuracy, read_trace,
    replay, summarize, write_results,
)


def model_for_variant(variant, entries):
    for name in sorted(entries, key=len, reverse=True):
        if variant == name or variant.startswith(name + "__"):
            return name
    # Do not silently award the requested model's accuracy to an unknown reply.
    return variant


def query_once(stub, image, model, slo_ms, timeout, entries):
    import grpc
    from infaas.proto import queryfe_pb2 as pb, request_reply_pb2 as rr

    try:
        response = stub.QueryOnline(pb.QueryOnlineRequest(
            raw_input=[image], parent_model=model,
            slo=pb.QuerySLO(LatencyInUSec=int(slo_ms * 1000))), timeout=timeout)
    except grpc.RpcError as error:
        return {**empty_reply(error.details() or str(error), error.code().name),
                "reply_type": "rpc_error", "server_variant": ""}
    ok = response.status.status == rr.SUCCESS
    timings = dict(response.timings)
    return {
        "ok": ok, "server_ms": response.latency_ms,
        "variant": model_for_variant(response.variant, entries),
        "server_variant": response.variant, "worker": response.worker,
        "gpu": response.hardware, "label": response.label, "path": response.path,
        "load_ms": timings.get("worker.load"),
        "queue_ms": timings.get("worker.ready_queue"),
        "service_ms": timings.get("worker.gpu_service"),
        "batch_size": int(timings.get("worker.batch_size", 1)) if ok else 0,
        "reply_type": "inference" if ok else "rejected",
        "reject_kind": response.reject_kind,
        "error": "" if ok else response.status.msg, "timings": timings,
    }


def extra_metrics(rows, elapsed):
    sent = [row for row in rows if row["sent"]]
    success = [row for row in sent if row["ok"]]
    good = [row for row in success if row["slo_met"]]
    breakdown = Counter(row.get("reply_type", "client_exception")
                        for row in sent if not row["ok"])
    breakdown["successful_but_late"] = len(success) - len(good)
    keys = (
        "worker.cpu_queue", "worker.cpu_service", "worker.decode",
        "worker.preprocess", "worker.ready_queue", "worker.gpu_service",
        "worker.forward", "worker.load", "srv.decide", "srv.total",
    )
    stages = {}
    for key in keys:
        values = [float(row["timings"][key]) for row in success
                  if key in row.get("timings", {})]
        values = [value for value in values if math.isfinite(value)]
        if values:
            stages[key] = {"n": len(values), "mean_ms": sum(values) / len(values),
                           "p50_ms": percentile(values, 50),
                           "p95_ms": percentile(values, 95)}
    return {
        "goodput_qps_including_drain": len(good) / max(elapsed, 1e-12),
        "slo_unmet_breakdown_sent": dict(breakdown),
        "failed_or_rejected_by_kind": dict(Counter(
            row.get("reject_kind") or row.get("reply_type") or "unknown"
            for row in sent if not row["ok"])),
        "successful_requests_by_server_variant": dict(Counter(
            row["server_variant"] for row in success)),
        "stage_timings_successful_replies": stages,
        "stage_note": "Only reported fields are summarized; stage intervals may overlap. "
                      "Do not add overlapping stages or compare generic queue_ms/service_ms "
                      "across implementations without checking their boundaries.",
        "batch_size_note": "Per-request worker-reported batch size; legacy replies without "
                           "the field use their fixed batch=1 contract. Actual execution counts "
                           "are stored separately from the single-GPU server's counter deltas.",
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--trace", required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--accuracy-registry", required=True)
    result.add_argument("--controller", default="127.0.0.1:50052")
    result.add_argument("--model", default="resnet-50")
    result.add_argument("--application", help="Task target for the single-GPU cost adapter")
    result.add_argument("--slo", type=positive, default=200.0)
    result.add_argument("--speed", type=positive, default=1.0)
    result.add_argument("--limit", type=count, default=100)
    result.add_argument("--concurrency", type=int, default=64)
    result.add_argument("--timeout", type=positive, default=10.0)
    result.add_argument("--max-send-lag-ms", type=positive, default=20.0)
    result.add_argument("--output", default="/results")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.concurrency <= 512:
        raise ValueError("concurrency must be in 1..512")
    arrivals, total, trace_hash = read_trace(args.trace, args.limit)
    entries, accuracy_hash = read_accuracy(args.accuracy_registry)
    if not args.application and args.model not in entries:
        raise ValueError("Requested model must be enabled in the accuracy registry")
    target = args.application or args.model
    image = Path(args.image).read_bytes()
    if not image or len(image) > 16 * 1024 * 1024:
        raise ValueError("Image must be nonempty and at most 16 MiB")
    config = {
        "backend": "infaas-native-queryonline", "controller": args.controller,
        "model": target, "application": args.application, "slo_override_ms": args.slo,
        "trace_path": str(args.trace), "trace_sha256": trace_hash,
        "trace_total_requests": total, "selected_requests": len(arrivals),
        "limit": args.limit, "speed": args.speed,
        "planned_duration_seconds": arrivals[-1].arrival_ms / (1000 * args.speed),
        "arrival_time_origin": "Original trace zero; first-arrival offset is preserved",
        "qos_note": "Trace QoS recorded, ignored in this single-SLO run",
        "image_path": str(args.image), "image_sha256": hashlib.sha256(image).hexdigest(),
        "accuracy_registry_sha256": accuracy_hash, "accuracy_metadata": entries,
        "checkpoint_note": "Registry is supplied reference metadata; current checkpoint "
                           "accuracy is not measured or independently verified by this client",
        "concurrency": args.concurrency, "rpc_timeout_seconds": args.timeout,
        "scope": "Repeated-JPEG open-loop trace replay; no server changes, reset or warmup",
        "latency_note": "Client e2e includes JPEG RPC, CPU preprocessing, queueing, "
                        "model load if needed, GPU execution and response transport",
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scheduler_sha256": hashlib.sha256(
            Path(__file__).resolve().parents[1].joinpath("common/trace_replay.py").read_bytes()).hexdigest(),
    }
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY RUN: no connection or inference requests were made.")
        return 0

    import grpc
    from infaas.proto import queryfe_pb2 as pb, queryfe_pb2_grpc, request_reply_pb2 as rr

    options = (("grpc.max_send_message_length", 20 * 1024 * 1024),
               ("grpc.max_receive_message_length", 20 * 1024 * 1024))
    with grpc.insecure_channel(args.controller, options=options) as channel:
        grpc.channel_ready_future(channel).result(timeout=args.timeout)
        stub = queryfe_pb2_grpc.QueryStub(channel)
        before = stub.QueryModelInfo(pb.QueryModelInfoRequest(model=target),
                                     timeout=args.timeout)
        if before.reply.status.status != rr.SUCCESS or not before.reply.all_models:
            raise ValueError("Model is not registered; run register before replay")
        config["registered_variants_before"] = list(before.reply.all_models)
        worker_before = worker_after = None
        if args.application:
            from infaas.single_gpu.client import status
            worker_before = status(channel, args.timeout)
            if (worker_before.get("implementation") != "infaas-single-gpu-cost-port"
                    or not worker_before.get("policy_ready")
                    or worker_before.get("application") != args.application
                    or worker_before.get("registry_sha256") != accuracy_hash
                    or set(worker_before.get("enabled_models", [])) != set(entries)):
                raise ValueError("Single-GPU task/registry/profile readiness mismatch")
            if worker_before["outstanding_requests"] or worker_before["maintenance"]:
                raise ValueError("Other requests or profiling are active; replay separately")
            if args.slo != worker_before["slo_ms"]:
                raise ValueError("Replay SLO differs from server configuration")
            if any(profile["image_sha256"] != config["image_sha256"]
                   for profile in worker_before["profiles"].values()):
                raise ValueError("Replay JPEG differs from the profiling input")
        rows, elapsed = replay(arrivals, lambda index: query_once(
            stub, image, target, args.slo, args.timeout, entries),
            speed=args.speed, slo_ms=args.slo, concurrency=args.concurrency)
        if args.application:
            worker_after = status(channel, args.timeout)

    summary = summarize(rows, entries, elapsed, args.max_send_lag_ms)
    summary.update(extra_metrics(rows, elapsed))
    if worker_before is not None:
        summary["actual_gpu_batch_execution_counts"] = {
            key: value - worker_before["batch_size_counts"].get(key, 0)
            for key, value in worker_after["batch_size_counts"].items()}
        summary["actual_model_switches_after_initial_load"] = (
            worker_after["model_switches_after_initial_load"] - worker_before["model_switches_after_initial_load"])
    csv_path, json_path = write_results(Path(args.output), rows,
        {"schema_version": 2, "config": config, "summary": summary,
         "worker_before": worker_before, "worker_after": worker_after})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"CSV: {csv_path}\nJSON: {json_path}", flush=True)
    # SLO misses are results, not a setup failure. Unsent/late client sends are not.
    return 0 if summary["trace_timing_valid"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}") from error
