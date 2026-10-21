"""Strict, torch-free configuration for the opt-in single-GPU experiment."""
import hashlib
import json
import math
import os
from pathlib import Path
import re


def registry(path):
    raw = Path(path).read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    if any(not isinstance(value.get("enabled", True), bool) for value in document["models"].values()):
        raise ValueError("Model enabled flags must be JSON booleans")
    entries = {key: value for key, value in document["models"].items() if value.get("enabled", True)}
    if len(entries) < 2:
        raise ValueError("Single-GPU comparison requires at least two enabled candidates")
    datasets = set()
    for name, entry in entries.items():
        score = entry.get("accuracy")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 < score <= 100:
            raise ValueError(f"Finite supplied reference accuracy is required for {name}")
        if not re.fullmatch(r"[0-9a-f]{40}", entry.get("revision", "")):
            raise ValueError(f"Pinned checkpoint commit is required for {name}")
        if not entry.get("accuracy_source") or not entry.get("accuracy_dataset"):
            raise ValueError(f"Accuracy provenance is required for {name}")
        if not entry.get("model_id"):
            raise ValueError(f"Model identity is required for {name}")
        datasets.add(entry["accuracy_dataset"])
    if len(datasets) != 1:
        raise ValueError("Candidate accuracy scores must use the same dataset/metric")
    return entries, hashlib.sha256(raw).hexdigest()


def positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class Settings:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.registry_path = env.get("MODEL_REGISTRY", "/config/models.json")
        self.entries, self.registry_hash = registry(self.registry_path)
        self.application = env.get("APPLICATION", "imagenet")
        if self.application in self.entries or not self.application:
            raise ValueError("APPLICATION must be a task name, not a candidate model name")
        self.hardware = env.get("GPU_TYPE", "v100")
        self.slo = positive(env.get("SLO_MS", 200), "SLO_MS")
        self.interval = positive(env.get("ALLOCATION_INTERVAL", 1), "ALLOCATION_INTERVAL")
        self.alpha = positive(env.get("EWMA_ALPHA", .7), "EWMA_ALPHA")
        if self.alpha > 1:
            raise ValueError("EWMA_ALPHA must be <= 1")
        self.initial_rate = positive(env.get("INITIAL_RATE", 1), "INITIAL_RATE")
        self.guard = float(env.get("DEADLINE_GUARD_MS", 2))
        self.slack = float(env.get("SLACK_THRESHOLD", 1.05))
        self.min_accuracy = float(env.get("MIN_ACCURACY", 0))
        if not math.isfinite(self.guard) or self.guard < 0 or not math.isfinite(self.slack) or self.slack < 1:
            raise ValueError("Invalid deadline guard or capacity slack")
        if not math.isfinite(self.min_accuracy) or not 0 <= self.min_accuracy <= 100:
            raise ValueError("MIN_ACCURACY must be between 0 and 100")
        self.batches = sorted(set(int(value) for value in env.get("BATCH_SIZES", "1,2,4,8").split(",")))
        if not self.batches or self.batches[0] != 1 or any(value < 1 or value > 64 for value in self.batches):
            raise ValueError("BATCH_SIZES must include 1 and stay within 1..64")
        self.profile_runs = int(env.get("PROFILE_RUNS", 30))
        self.threads = int(env.get("WORKER_THREADS", 32))
        if self.profile_runs < 2 or self.threads < 2:
            raise ValueError("PROFILE_RUNS and WORKER_THREADS must be >= 2")
        self.profile_path = Path(env.get("PROFILE_DIR", "/profiles")) / "single-gpu-cost-v1.json"
        self.name = env.get("WORKER_NAME", "infaas-single-gpu-worker")
