"""Independent GPU implementation built on THIS repository's INFaaS executor.

No Proteus runtime or policy is imported. One GPU model is resident at a time;
all load/profile/forward/unload calls are made by the single scheduler thread.
CPU preparation stays on the existing request-handler threads.
"""
import hashlib
import io
import json
import os
import time


class Backend:
    def __init__(self, settings):
        import torch
        from infaas.common import preprocessing
        from infaas.vendor.lumina import models
        from infaas.worker.runtime import Runtime
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU-inference fallback")
        if preprocessing.backend() != "pil":
            raise ValueError("Single-GPU comparison requires PREPROCESS_BACKEND=pil")
        self.settings = settings
        self.runtime = Runtime(settings.hardware)
        self.runtime.warm_cuda()
        self.processors = {}
        contracts = set()
        for name, entry in settings.entries.items():
            if models.resolve(name) != entry["model_id"] or models.MODEL_AUTOCLASS[entry["model_id"]] != "AutoModelForImageClassification":
                raise ValueError(f"Unsupported classification candidate identity: {name}")
            expected = preprocessing.checkpoint_options(entry["model_id"])["revision"]
            if expected != entry["revision"]:
                raise ValueError(f"Runtime checkpoint revision differs from registry: {name}")
            processor = preprocessing.get_processor(entry["model_id"])
            contract = {"class": type(processor).__name__, "config": processor.to_dict()}
            contracts.add(json.dumps(contract, sort_keys=True))
            self.processors[name] = processor
        if len(contracts) != 1:
            raise ValueError("Candidates must have exactly the same CPU preprocessing contract")
        self.processor = next(iter(self.processors.values()))
        self.fingerprint = {
            "schema": "infaas-single-gpu-cost-v1", "registry_sha256": settings.registry_hash,
            "batch_sizes": settings.batches, "profile_scope": "prepared-batch-service",
            "profile_runs": settings.profile_runs,
            "warmup_queries": int(os.environ.get("WARMUP_QUERIES", "10")),
            "preprocessing_contract": hashlib.sha256(next(iter(contracts)).encode()).hexdigest(),
            "models": {name: preprocessing.measured_fingerprint(entry["model_id"], settings.hardware)
                       for name, entry in settings.entries.items()},
            "gpu": torch.cuda.get_device_name(0), "dtype": "fp32",
        }
        self.resident = None

    def prepare(self, jpeg):
        from PIL import Image
        start = time.monotonic()
        with Image.open(io.BytesIO(jpeg)) as source:
            image = source.convert("RGB")
        decoded = time.monotonic()
        inputs = dict(self.processor(images=[image], return_tensors="pt"))
        if any(value.device.type != "cpu" for value in inputs.values()):
            raise RuntimeError("Preprocessing must produce CPU tensors")
        finished = time.monotonic()
        return {"inputs": inputs, "size": image.size, "cpu_start": start, "cpu_end": finished,
                "decode": (decoded - start) * 1000,
                "preprocess": (finished - decoded) * 1000}

    def load(self, model):
        if self.resident == model:
            return 0.0, False
        for variant in self.runtime.loaded():
            self.runtime.unload(variant)
        self.resident = None
        variant = f"{model}__{self.settings.hardware}"
        _, elapsed, _ = self.runtime.load(variant)
        self.resident = model
        return elapsed, True

    def execute(self, model, prepared):
        import torch
        inst = self.runtime.get(f"{model}__{self.settings.hardware}")
        if inst is None or self.resident != model:
            raise RuntimeError("Selected model is not resident")
        start = time.perf_counter()
        inputs = {key: torch.cat([item["inputs"][key] for item in prepared], dim=0)
                  for key in prepared[0]["inputs"]}
        output, _, _ = inst._forward(inputs)
        if not hasattr(output, "logits") or output.logits.ndim != 2:
            raise RuntimeError("Classification logits required")
        indices = output.logits.argmax(-1).detach().cpu().tolist()
        torch.cuda.synchronize()
        labels = getattr(inst.model.config, "id2label", {}) or {}
        elapsed = (time.perf_counter() - start) * 1000
        return [str(labels.get(index, index)) for index in indices], elapsed

    def profile_batch(self, model, prepared, batch, runs):
        import torch
        from statistics import fmean
        # Repeated identical samples match the prior repeated-JPEG experiment.
        payload = [prepared] * batch
        for _ in range(3):
            self.execute(model, payload)
        torch.cuda.reset_peak_memory_stats()
        latencies = [self.execute(model, payload)[1] for _ in range(runs)]
        memory = int(torch.cuda.max_memory_allocated())
        from infaas.common.trace_replay import percentile
        return {"p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95),
                "mean_ms": fmean(latencies), "samples_ms": latencies,
                "capacity_qps": 1000 * batch / fmean(latencies),
                "memory_bytes": memory, "runs": runs,
                "capacity_note": "Measured repeated prepared-batch throughput; excludes CPU preprocessing"}

    def unload(self):
        for variant in self.runtime.loaded():
            self.runtime.unload(variant)
        self.resident = None
