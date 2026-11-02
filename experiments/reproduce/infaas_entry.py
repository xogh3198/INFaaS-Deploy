"""Run the user's docker/v1 source. Only the model catalogue is extended."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

BASE = "baa65befb0ee551d589926a6d44f7343072ec9eb"
CATALOGUE = Path(__file__).with_name("infaas_models.json")


def verify(root="/app"):
    actual = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if actual != BASE:
        raise RuntimeError("Wrong INFaaS source commit: "+actual)
    dirty = subprocess.check_output(["git","-C",str(root),"diff","--name-only",BASE,"--","infaas"],text=True)
    if dirty.strip():
        raise RuntimeError("docker/v1 source was modified: "+dirty)
    return {"repository":"https://github.com/xogh3198/INFaaS-Deploy.git",
            "branch":"docker/v1","commit":BASE,"policy_source_modified":False}


def catalogue():
    from infaas.vendor.lumina import models
    for name in json.loads(CATALOGUE.read_text()).keys():
        mid = "microsoft/"+name
        if mid not in models.SUPPORTED_MODELS:
            models.MODELS.append((mid,"image-classification"))
            models.SUPPORTED_MODELS.add(mid)
            models.MODEL_CATEGORY[mid] = "image-classification"
            models.MODEL_AUTOCLASS[mid] = "AutoModelForImageClassification"
            models.SHORT_TO_ID[name] = mid
    return models


def download():
    from huggingface_hub import snapshot_download
    for name,revision in json.loads(CATALOGUE.read_text()).items():
        mid = "microsoft/"+name
        path = Path(snapshot_download(mid,revision=revision,cache_dir=os.environ["HF_HOME"]+"/hub",
            allow_patterns=["*.json","*.safetensors","pytorch_model.bin"]))
        # Native from_pretrained(mid) remains unchanged; pin its offline main ref.
        refs = path.parents[1]/"refs"
        refs.mkdir(exist_ok=True)
        (refs/"main").write_text(revision)
        print(json.dumps({"model":mid,"revision":revision,"snapshot":str(path)}),flush=True)


def main():
    print(json.dumps(verify()),flush=True)
    catalogue()
    module,*args = sys.argv[1:]
    if module == "download":
        return download()
    if module == "infaas.worker.main":
        import torch
        if not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("A working selected V100 is required; CPU fallback is refused")
    sys.argv = [module,*args]
    runpy.run_module(module,run_name="__main__")


if __name__ == "__main__":
    main()
