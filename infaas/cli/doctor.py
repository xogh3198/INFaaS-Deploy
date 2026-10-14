"""Verify the selected GPU, pinned CPU processor, and downloaded checkpoint."""
from __future__ import annotations

import argparse
import json

from infaas.common import preprocessing
from infaas.vendor.lumina import models


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="resnet-50")
    args = ap.parse_args()
    model_id = models.resolve(args.model)
    if model_id is None:
        ap.error(f"Unsupported model: {args.model}")
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU inference fallback is forbidden")
    torch.set_num_threads(preprocessing.cpu_threads())
    fingerprint = preprocessing.measured_fingerprint(model_id)
    expected = preprocessing.fingerprint(model_id)
    if preprocessing.backend() == "pil" and fingerprint != expected:
        raise RuntimeError(f"Worker ML versions differ from settings: actual={fingerprint}, expected={expected}")
    processor = preprocessing.get_processor(model_id)
    tensors = processor(images=[Image.new("RGB", (224, 224), (128, 128, 128))], return_tensors="pt")
    devices = {key: value.device.type for key, value in tensors.items()}
    if any(device != "cpu" for device in devices.values()):
        raise RuntimeError(f"Processor must run on CPU before H2D: {devices}")
    value = torch.ones((2, 2), device="cuda")
    cuda_test = (value @ value)[0, 0].item()
    torch.cuda.synchronize()
    if cuda_test != 2.0:
        raise RuntimeError("CUDA math check failed")
    print(json.dumps({"ok": True, "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "cuda_test": cuda_test, "processor_class": type(processor).__name__,
        "processor_backend": getattr(processor, "backend", None), "processor_device": devices,
        "cpu_threads": torch.get_num_threads(), "fingerprint": fingerprint,
        "note": "Processor/cache/CUDA check only; register measures GPU profiles; no model is made resident"
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
