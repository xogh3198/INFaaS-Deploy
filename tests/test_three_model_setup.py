"""Three-model setup validation; no GPU performance is claimed."""
import json
from pathlib import Path
from unittest.mock import patch

from infaas.single_gpu.deploy import ROOT, compose_document, read_env, settings
from infaas.single_gpu.settings import Settings, registry
from infaas.vendor.lumina import models


def test_three_pinned_candidates_use_the_same_reference_scores():
    path = ROOT / "deploy/docker/infaas-three-model.models.json"
    entries, _ = registry(path)
    assert {name: row["accuracy"] for name, row in entries.items()} == {
        "resnet-18": 70.93, "resnet-50": 77.36, "resnet-101": 78.34}
    for name, entry in entries.items():
        assert models.resolve(name) == entry["model_id"]
        assert models.MODEL_AUTOCLASS[entry["model_id"]] == "AutoModelForImageClassification"
    cfg = read_env(ROOT / "deploy/docker/infaas-three-model.env")
    cfg["MODEL_REGISTRY"] = str(path)
    runtime = Settings(cfg)
    assert len(runtime.entries) == 3
    assert runtime.threads == 32
    assert runtime.slo == 200


def test_preset_is_opt_in_and_keeps_request_handler_architecture():
    preset = ROOT / "deploy/docker/infaas-three-model.env"
    with patch.dict("os.environ", {"INFAAS_SINGLE_GPU_ENV": str(preset)}):
        cfg, selected = settings()
    assert selected == preset
    doc = compose_document(cfg)
    env = doc["services"]["worker"]["environment"]
    assert env["WORKER_THREADS"] == "32"
    assert env["CPU_THREADS"] == "2"
    assert env["CPU_INTEROP_THREADS"] == "1"
    assert "PREPROCESS_WORKERS" not in env
    assert env["PROFILE_RUNS"] == "30"
    assert "resnet-101:" in env["MODEL_REVISIONS"]
    assert doc["services"]["worker"]["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["2"]
    assert "three-model-v1" in cfg["DATA_DIR"]
