"""Shared preprocessing/checkpoint contract; usable without torch on the controller.

The CPU comparison path matches proteus_realtime.runtime.GpuRuntime. Serving,
placement, autoscaling, and batch size remain INFaaS's own implementation.
"""
from __future__ import annotations

import os
import re
import importlib.metadata

from infaas.vendor.lumina import models

RESNET50_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"
PINNED_VERSIONS = {
    "torch": "2.11.0+cu126", "torchvision": "0.26.0+cu126",
    "transformers": "5.19.0", "pillow": "12.1.1", "numpy": "2.2.6",
}


def backend() -> str:
    value = os.environ.get("PREPROCESS_BACKEND", "auto").strip().lower()
    if value not in {"pil", "auto", "dali"}:
        raise ValueError("PREPROCESS_BACKEND must be pil, auto or dali")
    return value


def cpu_threads() -> int:
    value = int(os.environ.get("CPU_THREADS", "2"))
    if value < 1:
        raise ValueError("CPU_THREADS must be a positive integer")
    return value


def revisions() -> dict:
    result = {"microsoft/resnet-50": RESNET50_REVISION}
    seen = set()
    for item in os.environ.get("MODEL_REVISIONS", "").split(","):
        if not item.strip():
            continue
        name, sep, revision = item.strip().partition(":")
        model_id = models.resolve(name.strip())
        if not sep or model_id is None or not re.fullmatch(r"[0-9a-fA-F]{40}", revision.strip()):
            raise ValueError("MODEL_REVISIONS requires supported-model:40-character-commit entries")
        if model_id in seen:
            raise ValueError(f"Duplicate MODEL_REVISIONS entry: {name}")
        seen.add(model_id)
        result[model_id] = revision.strip().lower()
    return result


def checkpoint_options(model_id: str, *, offline: bool = True) -> dict:
    options = {"local_files_only": offline}
    revision = revisions().get(model_id)
    if revision:
        options["revision"] = revision
    return options


def fingerprint(model_id: str, hardware: str = "") -> dict:
    versions = dict(PINNED_VERSIONS)
    versions["torch"] = os.environ.get("TORCH_VERSION", versions["torch"])
    versions["torchvision"] = os.environ.get("TORCHVISION_VERSION", versions["torchvision"])
    selected = backend()
    hw = hardware or os.environ.get("GPU_TYPE", "")
    modes = dict(item.split(":", 1) for item in os.environ.get("PREPROCESS_MODES", "").split(",") if ":" in item)
    return {
        "schema": "infaas-preprocessing-v1", "backend": selected,
        "pipeline": "cpu-ready-v1" if selected == "pil" else "legacy-dali",
        "preprocess_executor": "request-handler",
        "cpu_interop_threads": int(os.environ.get("CPU_INTEROP_THREADS", "1")),
        "dali_mode": None if selected == "pil" else modes.get(hw, "cpu_decode_gpu"),
        "cpu_threads": cpu_threads(), "processor_use_fast": False,
        "model_id": model_id, "revision": revisions().get(model_id, "main"),
        "ml_versions": versions,
    }


def get_processor(model_id: str, *, offline: bool = True):
    from transformers import AutoImageProcessor, AutoProcessor
    options = checkpoint_options(model_id, offline=offline)
    try:
        return AutoImageProcessor.from_pretrained(model_id, use_fast=False, **options)
    except (ValueError, KeyError):
        # Non-image processor catalog entries retain the existing fallback.
        # Missing/offline checkpoints must not be hidden by a broad retry.
        return AutoProcessor.from_pretrained(model_id, use_fast=False, **options)


def measured_fingerprint(model_id: str, hardware: str = "") -> dict:
    result = fingerprint(model_id, hardware)
    if result["backend"] != "pil":
        result["dali_mode"] = os.environ.get("PREPROCESS_MODE", result["dali_mode"])
    result["ml_versions"] = {name: importlib.metadata.version(name) for name in PINNED_VERSIONS}
    return result
