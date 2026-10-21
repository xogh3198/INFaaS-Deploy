"""Ready-tensor batching, serialized GPU profiling and one-resident-model execution."""
from collections import Counter, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import hashlib
import json
import os
import threading
import time

from .policy import candidates, choose


class Rejected(RuntimeError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


@dataclass
class Job:
    arrival: float
    slo: float
    minimum_accuracy: float
    prepared: object = None
    done: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: object = None

    @property
    def deadline(self):
        return self.arrival + self.slo / 1000


class Executor:
    def __init__(self, settings, backend):
        self.settings, self.backend = settings, backend
        self.condition = threading.Condition(threading.RLock())
        self.queue, self.commands = deque(), deque()
        self.profiles = {}
        self.plan = None
        self.current = None
        self.plan_details = {}
        self.history, self.switches = deque(maxlen=256), deque(maxlen=512)
        self.batch_counts, self.model_counts = Counter(), Counter()
        self.seen = self.processed = self.failed = self.deadline_drops = self.slo_misses = 0
        self.actual_switch_count = 0
        self.outstanding = 0
        self.busy = self.maintenance = self.stopped = False
        self.rate = settings.initial_rate
        self.rate_start = time.monotonic()
        self.window_arrivals = 0
        self.profile_error = None
        self._restore()
        self.thread = threading.Thread(target=self._loop, name="infaas-single-gpu", daemon=True)
        self.thread.start()

    def ready(self):
        return set(self.profiles) == set(self.settings.entries) and not self.profile_error

    def _restore(self):
        if self.settings.profile_path.exists():
            try:
                data = json.loads(self.settings.profile_path.read_text(encoding="utf-8"))
                if data["fingerprint"] != self.backend.fingerprint:
                    raise ValueError("Profile fingerprint differs; run register --reprofile")
                self.profiles = data["profiles"]
                for profile in self.profiles.values():
                    if sorted(map(int, profile["batches"])) != self.settings.batches:
                        raise ValueError("Batch profile set differs")
            except Exception as error:
                self.profile_error = str(error)
                self.profiles = {}

    def profile(self, model, image, runs=None):
        if model not in self.settings.entries:
            raise ValueError(f"Unknown candidate: {model}")
        with self.condition:
            if self.maintenance or self.outstanding or self.busy:
                raise Rejected("busy", "Profile only when no inference requests are outstanding")
            self.maintenance = True
            future = Future()
            self.commands.append((model, image, runs or self.settings.profile_runs, future))
            self.condition.notify_all()
        return future.result(timeout=600)

    def _profile(self, model, image, runs):
        if runs < 2:
            raise ValueError("At least two profile trials required")
        sample = self.backend.prepare(image)
        load_ms, _ = self.backend.load(model)
        try:
            batches = {str(batch): self.backend.profile_batch(model, sample, batch, runs)
                       for batch in self.settings.batches}
            profile = {"model": model, "load_ms": load_ms, "batches": batches,
                "profile_scope": "prepared-batch-service", "image_sha256": hashlib.sha256(image).hexdigest(),
                "reference_accuracy": self.settings.entries[model]["accuracy"],
                "accuracy_measured": False, "profiled_at": time.time()}
        finally:
            self.backend.unload()
        with self.condition:
            self.profiles = {**self.profiles, model: profile}
            self.profile_error = None
            data = {"fingerprint": self.backend.fingerprint, "profiles": self.profiles}
            self.settings.profile_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.settings.profile_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(temporary, self.settings.profile_path)
            self.plan = None
            self.current = None
            self.rate_start, self.window_arrivals = time.monotonic(), 0
            self.rate = self.settings.initial_rate
        return profile

    def submit(self, image, slo, minimum_accuracy=0):
        if not image or len(image) > 16 << 20:
            raise Rejected("bad_request", "JPEG must be nonempty and at most 16 MiB")
        if not 0 < slo <= 600_000 or not 0 <= minimum_accuracy <= 100:
            raise Rejected("bad_request", "Invalid SLO/accuracy constraint")
        job = Job(time.monotonic(), slo, max(minimum_accuracy, self.settings.min_accuracy))
        with self.condition:
            if self.stopped or self.maintenance or not self.ready():
                raise Rejected("not_ready", "Register all enabled candidate profiles before replay")
            self.seen += 1
            self.window_arrivals += 1
            self.outstanding += 1
        try:
            # Preserve INFaaS request-handler preprocessing concurrency.
            job.prepared = self.backend.prepare(image)
        except Exception:
            with self.condition:
                self.outstanding -= 1
                self.failed += 1
            raise
        with self.condition:
            if self.stopped:
                self.outstanding -= 1
                raise Rejected("stopped", "Executor stopped during preprocessing")
            self.queue.append(job)
            self.condition.notify_all()
        job.done.wait()
        if job.error:
            raise job.error
        return job.result

    def _refresh(self, now, slo, minimum_accuracy):
        elapsed = now - self.rate_start
        if elapsed >= self.settings.interval:
            observed = self.window_arrivals / elapsed
            self.rate = self.settings.alpha * observed + (1 - self.settings.alpha) * self.rate
            self.rate_start, self.window_arrivals = now, 0
        options = candidates(self.profiles, self.settings.entries, self.settings.hardware, slo, minimum_accuracy)
        previous = self.plan
        selected, details = choose(options, self.rate, self.current, self.settings.slack)
        self.plan, self.plan_details = selected, details
        if previous is None or (previous.model, previous.batch) != (selected.model, selected.batch):
            self.history.append({"epoch_seconds": time.time(), **selected.as_dict(), **details})
        return selected

    def _fail(self, job, kind, message):
        job.error = Rejected(kind, message)
        self.failed += 1
        self.deadline_drops += kind == "deadline_drop"
        self.outstanding -= 1
        job.done.set()

    def _take(self):
        now = time.monotonic()
        while self.queue and self.queue[0].deadline <= now:
            self._fail(self.queue.popleft(), "deadline_drop", "Expired after CPU preparation/ready queue")
        if not self.queue:
            return None
        head = self.queue[0]
        try:
            plan = self._refresh(now, min(job.slo for job in self.queue),
                                 max(job.minimum_accuracy for job in self.queue))
        except ValueError as error:
            self._fail(self.queue.popleft(), "no_variant", str(error))
            return None
        table = self.profiles[plan.model]["batches"]
        available = min(len(self.queue), plan.batch)
        batch = max(size for size in self.settings.batches if size <= available)
        # Wait for one more configured batch size only until its latest safe start.
        larger = [size for size in self.settings.batches if available < size <= plan.batch]
        if larger:
            target = min(larger)
            latest = min(job.deadline for job in list(self.queue)[:available]) - (
                table[str(target)]["p50_ms"] + self.settings.guard) / 1000
            if now < latest:
                self.condition.wait(min(latest - now, self.settings.interval))
                return None
        sample = list(self.queue)[:batch]
        predicted = table[str(batch)]["p50_ms"] + self.settings.guard
        if (now + predicted / 1000) > min(job.deadline for job in sample):
            self._fail(self.queue.popleft(), "deadline_drop", "Predicted prepared batch exceeds remaining SLO")
            return None
        return plan, [self.queue.popleft() for _ in range(batch)]

    def _loop(self):
        while True:
            with self.condition:
                if self.stopped:
                    break
                if self.commands:
                    command = self.commands.popleft()
                    work = None
                elif self.queue:
                    command = None
                    work = self._take()
                    if work:
                        self.busy = True
                else:
                    self.condition.wait(self.settings.interval)
                    continue
            if command:
                model, image, runs, future = command
                result, failure = None, None
                try:
                    result = self._profile(model, image, runs)
                except BaseException as error:
                    failure = error
                    with self.condition:
                        self.profiles.pop(model, None)
                        self.profile_error = str(error)
                finally:
                    with self.condition:
                        self.maintenance = False
                        self.condition.notify_all()
                if failure:
                    future.set_exception(failure)
                else:
                    future.set_result(result)
                continue
            if not work:
                continue
            plan, jobs = work
            try:
                loading_start = time.monotonic()
                old = self.backend.resident
                load_ms, loaded = self.backend.load(plan.model)
                with self.condition:
                    self.current = plan
                if loaded:
                    with self.condition:
                        self.actual_switch_count += old is not None
                        self.switches.append({"epoch_seconds": time.time(), "from": old,
                            "to": plan.model, "load_ms": load_ms})
                labels, gpu_ms = self.backend.execute(plan.model, [job.prepared for job in jobs])
                if len(labels) != len(jobs):
                    raise RuntimeError("Batch result cardinality differs from request count")
                finish = time.monotonic()
                with self.condition:
                    self.batch_counts[len(jobs)] += 1
                    self.model_counts[plan.model] += len(jobs)
                    for job, label in zip(jobs, labels):
                        cpu = job.prepared
                        total = (finish - job.arrival) * 1000
                        timings = {"decode": cpu["decode"], "preprocess": cpu["preprocess"],
                            "cpu_queue": 0.0, "cpu_service": (cpu["cpu_end"] - cpu["cpu_start"]) * 1000,
                            "ready_queue": (loading_start - cpu["cpu_end"]) * 1000,
                            "gpu_service": gpu_ms, "load": load_ms, "batch_size": len(jobs),
                            "predicted_gpu_ms": self.profiles[plan.model]["batches"][str(len(jobs))]["p50_ms"],
                            "reference_accuracy": plan.accuracy, "model_loaded": float(loaded),
                            "wall": total, "total": total, "slo_missed": float(total > job.slo)}
                        job.result = {"label": label, "size": cpu["size"], "timings": timings,
                            "variant": f"{plan.model}__{self.settings.hardware}", "latency_ms": total,
                            "path": "inactive" if loaded else "active"}
                        self.processed += 1
                        self.slo_misses += total > job.slo
                        self.outstanding -= 1
                        job.done.set()
            except BaseException as error:
                with self.condition:
                    for job in jobs:
                        self._fail(job, "worker_error", str(error))
            finally:
                with self.condition:
                    self.busy = False
                    self.condition.notify_all()
        self.backend.unload()

    def status(self):
        with self.condition:
            return {"implementation": "infaas-single-gpu-cost-port", "application": self.settings.application,
                "policy_ready": bool(self.ready()), "policy_error": self.profile_error,
                "objective": "infaas-memory-loading-cost", "gpu_type": self.settings.hardware,
                "enabled_models": list(self.settings.entries), "registry_sha256": self.settings.registry_hash,
                "checkpoints": self.settings.entries, "resident_model": self.backend.resident,
                "profiles": self.profiles, "profile_fingerprint": self.backend.fingerprint,
                "plan": self.plan.as_dict() if self.plan else None, "plan_details": self.plan_details,
                "plan_history": list(self.history), "load_history": list(self.switches),
                "model_switches_after_initial_load": self.actual_switch_count,
                "batch_size_counts": dict(self.batch_counts), "successful_requests_by_model": dict(self.model_counts),
                "queue_depth": len(self.queue), "outstanding_requests": self.outstanding,
                "busy": self.busy, "maintenance": self.maintenance, "seen": self.seen,
                "processed": self.processed, "failed": self.failed, "deadline_drops": self.deadline_drops,
                "server_slo_misses": self.slo_misses, "slo_ms": self.settings.slo,
                "accuracy_measured": False, "profile_scope": "prepared-batch-service",
                "preprocess_executor": "request-handler", "worker_threads": self.settings.threads,
                "guard_ms": self.settings.guard, "allocation_interval": self.settings.interval,
                "note": "Single-device adapter; not the full original distributed INFaaS autoscaler"}

    def close(self):
        with self.condition:
            self.stopped = True
            while self.queue:
                self._fail(self.queue.popleft(), "stopped", "Executor closing")
            while self.commands:
                self.commands.popleft()[3].set_exception(Rejected("stopped", "Executor closing"))
            self.condition.notify_all()
        self.thread.join(timeout=10)
