"""Shared open-loop trace scheduling and result definitions; no serving policy."""
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Arrival:
    line: int
    arrival_ms: float
    qos: str
    original_slo_ms: float


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def count(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def read_trace(path, limit=100):
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    raw = Path(path).read_bytes()
    arrivals = []
    previous = -1.0
    for line, fields in enumerate(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), 1):
        if not fields or not any(part.strip() for part in fields):
            continue
        if fields[0].lstrip().startswith("#"):
            continue
        if len(fields) != 3:
            raise ValueError(f"Trace line {line}: expected arrival_ms,qos,slo_ms (no header)")
        arrival, original_slo = float(fields[0]), float(fields[2])
        if not math.isfinite(arrival) or arrival < 0 or arrival < previous:
            raise ValueError(f"Trace line {line}: nonnegative, nondecreasing finite times required")
        if not math.isfinite(original_slo) or original_slo <= 0:
            raise ValueError(f"Trace line {line}: invalid original SLO")
        previous = arrival
        arrivals.append(Arrival(line, arrival, fields[1].strip(), original_slo))
    if not arrivals:
        raise ValueError("Trace contains no requests")
    selected = arrivals[:limit] if limit else arrivals
    return selected, len(arrivals), hashlib.sha256(raw).hexdigest()


def read_accuracy(path):
    raw = Path(path).read_bytes()
    registry = json.loads(raw.decode("utf-8-sig"))
    entries = {name: entry for name, entry in registry["models"].items()
               if entry.get("enabled", True)}
    if not entries:
        raise ValueError("Enable at least one accuracy-registry model")
    datasets = set()
    for name, entry in entries.items():
        accuracy = entry.get("accuracy")
        if (isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
                or not math.isfinite(accuracy) or not 0 < accuracy <= 100):
            raise ValueError(f"Offline/reference accuracy is required for {name}")
        if not entry.get("accuracy_source") or not entry.get("accuracy_dataset"):
            raise ValueError(f"Accuracy source and dataset are required for {name}")
        datasets.add(entry["accuracy_dataset"])
    if len(datasets) != 1:
        raise ValueError("All model accuracies must have the same dataset and metric")
    return entries, hashlib.sha256(raw).hexdigest()


def empty_reply(error, kind):
    return {"ok": False, "e2e_ms": None, "server_ms": None, "slo_met": False,
            "variant": "", "batch_size": 0, "load_ms": None, "queue_ms": None,
            "service_ms": None, "path": "", "label": "", "gpu": "",
            "request_name": "", "reject_kind": kind, "error": error, "timings": {}}


def replay(arrivals, send, *, speed=1.0, slo_ms=200.0, concurrency=64,
           progress=True, clock=time.perf_counter, sleep=time.sleep):
    """Open-loop schedule, bounded active calls; never await one reply to send the next.

    Client saturation produces an explicit unsent row, not a hidden skipped request.
    Server SLO starts on actual sending; scheduled-deadline metrics also include lag.
    """
    if (not math.isfinite(speed) or speed <= 0 or not math.isfinite(slo_ms)
            or slo_ms <= 0 or concurrency < 1):
        raise ValueError("Positive speed, SLO and concurrency required")
    rows, futures = [], []
    slots = threading.BoundedSemaphore(concurrency)
    start = clock()
    last_progress = start

    def execute(index, arrival, due, offset):
        sent = clock()
        try:
            reply = dict(send(index))
        except Exception as error:
            reply = empty_reply(str(error), "client_exception")
        finished = clock()
        elapsed = (finished - sent) * 1000
        return {**reply, "index": index, "trace_line": arrival.line,
                "trace_arrival_ms": arrival.arrival_ms, "trace_qos": arrival.qos,
                "trace_original_slo_ms": arrival.original_slo_ms,
                "scheduled_offset_ms": offset * 1000, "slo_ms": slo_ms,
                "sent": True, "actual_send_offset_ms": (sent - start) * 1000,
                "client_send_lag_ms": max(0, (sent - due) * 1000),
                "completion_offset_ms": (finished - start) * 1000,
                "e2e_ms": elapsed,
                "scheduled_e2e_ms": (finished - due) * 1000,
                "slo_met": bool(reply["ok"] and elapsed <= slo_ms),
                "scheduled_slo_met": bool(reply["ok"] and (finished - due) * 1000 <= slo_ms)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index, arrival in enumerate(arrivals):
            offset = arrival.arrival_ms / (1000 * speed)
            due = start + offset
            delay = due - clock()
            if delay > 0:
                sleep(delay)
            if not slots.acquire(blocking=False):
                rows.append({**empty_reply("Client concurrency limit reached; request NOT sent",
                                          "client_not_sent"),
                    "index": index, "trace_line": arrival.line,
                    "trace_arrival_ms": arrival.arrival_ms, "trace_qos": arrival.qos,
                    "trace_original_slo_ms": arrival.original_slo_ms,
                    "scheduled_offset_ms": offset * 1000, "slo_ms": slo_ms,
                    "sent": False, "actual_send_offset_ms": None,
                    "client_send_lag_ms": None, "completion_offset_ms": None,
                    "scheduled_e2e_ms": None, "scheduled_slo_met": False})
            else:
                future = pool.submit(execute, index, arrival, due, offset)
                future.add_done_callback(lambda completed: slots.release())
                futures.append(future)
            now = clock()
            if progress and now - last_progress >= 10:
                print(f"scheduled={index + 1}/{len(arrivals)} submitted={len(futures)} "
                      f"client_not_sent={len(rows)} elapsed={now - start:.1f}s", flush=True)
                last_progress = now
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["index"])
    return rows, clock() - start


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize(rows, entries, elapsed, max_lag_ms=20.0):
    if not rows:
        raise ValueError("No scheduled requests")
    sent = [row for row in rows if row["sent"]]
    success = [row for row in sent if row["ok"]]
    good = [row for row in sent if row["slo_met"]]
    scheduled_good = [row for row in rows if row["scheduled_slo_met"]]
    latencies = [row["e2e_ms"] for row in success]
    counts = Counter(row.get("variant", "") for row in success)
    unknown = sorted(name for name in counts if name not in entries)

    def reference_mean(selected):
        if not selected or any(row.get("variant") not in entries for row in selected):
            return None
        return sum(entries[row["variant"]]["accuracy"] for row in selected) / len(selected)

    best = max(entry["accuracy"] for entry in entries.values())
    mean = reference_mean(success)
    lag = [row["client_send_lag_ms"] for row in sent]
    late = sum(value > max_lag_ms for value in lag)
    not_sent = len(rows) - len(sent)
    offering_seconds = rows[-1]["scheduled_offset_ms"] / 1000
    return {"scheduled_requests": len(rows), "sent_requests": len(sent),
            "client_not_sent": not_sent, "successful_replies": len(success),
            "failed_or_rejected_sent_requests": len(sent) - len(success),
            "slo_met_sent_requests": len(good),
            "slo_unmet_ratio_of_sent_requests": (1 - len(good) / len(sent)) if sent else None,
            "scheduled_deadline_unmet_ratio": 1 - len(scheduled_good) / len(rows),
            "scheduled_deadline_note": "Includes unsent requests and client send lag; "
                                       "not solely a server SLO metric",
            "successful_requests_by_model": dict(counts),
            "slo_met_requests_by_model": dict(Counter(row["variant"] for row in good)),
            "mean_reference_accuracy_of_successful_replies": mean,
            "mean_reference_accuracy_of_slo_met_replies": reference_mean(good),
            "normalized_reference_accuracy_of_successful_replies":
                100 * mean / best if mean is not None else None,
            "unknown_variants_for_accuracy": unknown,
            "accuracy_note": "Request-weighted supplied model scores, NOT measured image "
                             "correctness or verification of current checkpoint accuracy",
            "mean_e2e_ms": sum(latencies) / len(latencies) if latencies else None,
            "p50_e2e_ms": percentile(latencies, 50),
            "p95_e2e_ms": percentile(latencies, 95),
            "p99_e2e_ms": percentile(latencies, 99),
            "latency_population": "Successful replies only, including successful late replies",
            "max_client_send_lag_ms": max(lag, default=None),
            "p95_client_send_lag_ms": percentile(lag, 95),
            "send_lag_threshold_ms": max_lag_ms, "requests_above_send_lag_threshold": late,
            "trace_timing_valid": not_sent == 0 and late == 0,
            "offering_window_seconds": offering_seconds,
            "actual_sent_per_offering_second": len(sent) / offering_seconds if offering_seconds else None,
            "elapsed_seconds_including_drain": elapsed,
            "successful_reply_qps_including_drain": len(success) / max(elapsed, 1e-12)}


def write_results(output, rows, report):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "trace-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    csv_path, json_path = directory / (stem + ".csv"), directory / (stem + ".json")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "timings": json.dumps(row.get("timings", {}))})
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return csv_path, json_path
