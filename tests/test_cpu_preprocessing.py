"""CPU comparison contract: config, checkpoints, profiles, and worker wiring.

No CUDA hardware is required. Worker calls use explicitly mocked tensors; the
server doctor + query commands perform the actual CUDA verification separately.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from infaas.common import config, preprocessing, profiles
from infaas.controller.registrar import import_profiles
from infaas.vendor import lumina

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_comparison_compose", ROOT / "deploy/docker/gen_compose.py")
compose = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compose
spec.loader.exec_module(compose)
SMI = [{"index": 2, "uuid": "GPU-fixture", "name": "Tesla V100-DGXS-32GB", "mem_mib": 32768}]


def test_comparison_preset_and_shared_worker_controller_contract():
    cfg = compose.read_env(ROOT / "deploy/docker/infaas-proteus-cpu.env")
    document, env, _, _ = compose.build(cfg, SMI)
    worker = document["services"]["infaas-worker-2"]
    assert worker["environment"]["PREPROCESS_BACKEND"] == env["PREPROCESS_BACKEND"] == "pil"
    assert worker["environment"]["CPU_THREADS"] == env["CPU_THREADS"] == "2"
    assert worker["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["2"]
    assert cfg["MODEL_REVISIONS"] == "resnet-50:" + preprocessing.RESNET50_REVISION
    assert env["PROFILE_REQUIRE_PREPROCESS_MATCH"] == "true"
    assert env["PROFILE_MODE"] == "service"
    assert env["WARMUP_QUERIES"] == "10"
    assert "proteus-cpu/profiles" in cfg["PROFILE_DIR"]
    assert "proteus-cpu/logs" in cfg["LOG_DIR"]
    assert cfg["WORKER_IMAGE"] != "infaas-worker:0.1.0"
    assert cfg["DECISION_MODE"] == "6" and cfg["MODEL_AUTOSCALER_INTERVAL_S"] == "1.0"


@pytest.mark.parametrize("key,value", [("PREPROCESS_BACKEND", "typo"), ("CPU_THREADS", "0"), ("CPU_THREADS", "two")])
def test_invalid_settings_stop_before_compose(key, value):
    with pytest.raises(compose.ConfigError):
        compose.build({"GPUS": "2:v100", "HW_COST": "v100:1", key: value}, SMI)


def test_dali_remains_selectable_not_overridden_by_compose():
    document, env, _, _ = compose.build({"GPUS": "2:v100", "HW_COST": "v100:1",
        "PREPROCESS_BACKEND": "dali", "INSTALL_DALI": "1"}, SMI)
    assert document["services"]["infaas-worker-2"]["environment"]["PREPROCESS_BACKEND"] == "dali"
    assert env["PREPROCESS_BACKEND"] == "dali"


def test_checkpoint_download_and_runtime_use_same_pin(monkeypatch):
    monkeypatch.delenv("MODEL_REVISIONS", raising=False)
    offline = preprocessing.checkpoint_options("microsoft/resnet-50")
    online = preprocessing.checkpoint_options("microsoft/resnet-50", offline=False)
    assert offline == {"revision": preprocessing.RESNET50_REVISION, "local_files_only": True}
    assert online == {**offline, "local_files_only": False}
    monkeypatch.setenv("MODEL_REVISIONS", "resnet-18:" + "a" * 40)
    assert preprocessing.checkpoint_options("microsoft/resnet-18")["revision"] == "a" * 40


@pytest.mark.parametrize("value", ["resnet-50:main", "unknown:" + "a" * 40,
    "resnet-18:" + "a" * 40 + ",resnet-18:" + "b" * 40])
def test_revision_must_be_pinned_supported_and_unique(monkeypatch, value):
    monkeypatch.setenv("MODEL_REVISIONS", value)
    with pytest.raises(ValueError):
        preprocessing.revisions()


def test_processor_is_explicitly_slow_and_missing_cache_is_not_hidden(monkeypatch):
    image = Mock()
    fallback = Mock()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoImageProcessor=image, AutoProcessor=fallback))
    preprocessing.get_processor("microsoft/resnet-50")
    image.from_pretrained.assert_called_once_with("microsoft/resnet-50", use_fast=False,
        revision=preprocessing.RESNET50_REVISION, local_files_only=True)
    image.from_pretrained.side_effect = OSError("checkpoint missing")
    with pytest.raises(OSError, match="checkpoint missing"):
        preprocessing.get_processor("microsoft/resnet-50")
    fallback.from_pretrained.assert_not_called()


def profile_row():
    return {"variant": "resnet-50__v100", "model": "resnet-50", "hardware": "v100",
        "inf_latency_ms": 80, "load_latency_ms": 500, "sat_qps": 12, "peak_memory_bytes": 100,
        "profile_mode": "service", "preprocessing_config": preprocessing.fingerprint("microsoft/resnet-50")}


def test_old_and_changed_profiles_require_reprofiling(monkeypatch, tmp_path):
    monkeypatch.setenv("PREPROCESS_BACKEND", "pil")
    monkeypatch.setattr(config, "PROFILE_MODE", "service")
    monkeypatch.setattr(config, "PROFILE_REQUIRE_PREPROCESS_MATCH", True)
    row = profile_row()
    assert profiles.compatible(row)
    old = {k: v for k, v in row.items() if k != "preprocessing_config"}
    assert not profiles.compatible(old)
    profiles.save(old, str(tmp_path))
    md = Mock()
    assert import_profiles(md, str(tmp_path)) == 0
    md.add_model.assert_not_called()
    assert profiles.load(old["variant"], str(tmp_path)) == old  # never deletes old results
    monkeypatch.setenv("CPU_THREADS", "4")
    assert not profiles.compatible(row)
    monkeypatch.setenv("CPU_THREADS", "2")
    monkeypatch.setenv("PREPROCESS_BACKEND", "dali")
    assert not profiles.compatible(row)
    monkeypatch.setenv("PREPROCESS_BACKEND", "pil")
    monkeypatch.setenv("MODEL_REVISIONS", "resnet-50:" + "a" * 40)
    assert not profiles.compatible(row)


def test_dali_mode_changes_invalidate_profiles(monkeypatch):
    monkeypatch.setenv("PREPROCESS_BACKEND", "dali")
    monkeypatch.setenv("PREPROCESS_MODES", "v100:cpu_decode_gpu")
    row = profile_row()
    assert profiles.matches_preprocessing(row)
    monkeypatch.setenv("PREPROCESS_MODES", "v100:cpu_only")
    assert not profiles.matches_preprocessing(row)


def load_mock_runtime(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, Stream=Mock()),
        set_num_threads=Mock(), get_num_threads=lambda: 2,
        backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, allow_tf32=True),
                                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))))
    dali = SimpleNamespace(available=Mock(return_value=False), status=lambda: "mock-dali")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "infaas.vendor.lumina.dali_preprocess", dali)
    monkeypatch.setattr(lumina, "dali_preprocess", dali, raising=False)
    spec = importlib.util.spec_from_file_location("mock_infaas_runtime_cpu", ROOT / "infaas/worker/runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime, fake_torch, dali


def test_cpu_runtime_never_calls_dali_and_keeps_cuda_execution(monkeypatch):
    monkeypatch.setenv("PREPROCESS_BACKEND", "pil")
    monkeypatch.setenv("CPU_THREADS", "2")
    runtime, torch, dali = load_mock_runtime(monkeypatch)
    instance = runtime.Runtime("v100")
    assert instance._probe is None
    dali.available.assert_not_called()
    torch.set_num_threads.assert_called_once_with(2)
    assert runtime.DEVICE == "cuda"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_dali_requested_but_not_installed_fails(monkeypatch):
    monkeypatch.setenv("PREPROCESS_BACKEND", "dali")
    runtime, _, _ = load_mock_runtime(monkeypatch)
    with pytest.raises(RuntimeError, match="INSTALL_DALI=1"):
        runtime.Runtime("v100")


def test_comparison_forbids_cpu_inference_fallback(monkeypatch):
    monkeypatch.setenv("PREPROCESS_BACKEND", "pil")
    monkeypatch.setenv("REQUIRE_CUDA", "true")
    runtime, _, _ = load_mock_runtime(monkeypatch)
    runtime.DEVICE = "cpu"
    with pytest.raises(RuntimeError, match="CPU-inference fallback"):
        runtime.Runtime("v100")


def test_profile_captures_checkpoint_before_unload(monkeypatch):
    _, fake_torch, _ = load_mock_runtime(monkeypatch)
    monkeypatch.setitem(sys.modules, "infaas.worker.runtime", SimpleNamespace(Runtime=object))
    spec = importlib.util.spec_from_file_location("mock_infaas_profiler_cpu", ROOT / "infaas/worker/profiler.py")
    profiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiler)
    tensor = SimpleNamespace(shape=(1, 3, 224, 224))
    inst = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash=preprocessing.RESNET50_REVISION)),
        processor=object(), dali_cfg=None, dali_note="PIL", ready_ms=25,
        forward_only=Mock(side_effect=[80, 82]))
    runtime = SimpleNamespace(hw="v100", get=lambda variant: None, is_loading=lambda variant: False,
        load=lambda variant: (inst, 50, True), unload=lambda variant: setattr(inst, "model", None))
    monkeypatch.setattr(profiler, "synthetic_inputs", lambda processor: {"pixel_values": tensor})
    monkeypatch.setattr(profiler, "_used", lambda gpu: 100)
    monkeypatch.setattr(preprocessing, "measured_fingerprint", preprocessing.fingerprint)
    row = profiler.profile(runtime, SimpleNamespace(name="V100"), "resnet-50", b"", runs=2, mode="original")
    assert inst.model is None
    assert row["checkpoint_commit"] == preprocessing.RESNET50_REVISION
    assert row["inf_latency_ms"] == 81 and row["dali"] is False
    assert row["preprocessing_config"] == preprocessing.fingerprint("microsoft/resnet-50")


def test_cpu_preprocess_uses_rgb_image_list_before_forward(monkeypatch):
    runtime, _, _ = load_mock_runtime(monkeypatch)
    tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    processor = Mock(return_value={"pixel_values": tensor})
    inst = runtime.Instance.__new__(runtime.Instance)
    inst.dali_cfg, inst.processor, inst.model = None, processor, object()
    inst._forward = Mock(side_effect=lambda inputs: (object(), runtime.time.perf_counter(), runtime.time.perf_counter()))
    monkeypatch.setattr(runtime, "_decode", lambda *args: ("Samoyed", []))
    buf = io.BytesIO()
    Image.new("L", (25, 19), 128).save(buf, format="JPEG")
    label, _, shape, timings = inst._infer(buf.getvalue())
    images = processor.call_args.kwargs["images"]
    assert len(images) == 1 and images[0].mode == "RGB"
    assert processor.call_args.kwargs["return_tensors"] == "pt"
    inst._forward.assert_called_once_with({"pixel_values": tensor})
    assert label == "Samoyed" and shape == (25, 19) and timings["dali"] == 0
    tensor.device.type = "cuda"
    with pytest.raises(RuntimeError, match="CPU tensors"):
        inst._infer(buf.getvalue())


def test_docker_pins_and_build_download_doctor_contract():
    recipe = (ROOT / "docker/worker.Dockerfile").read_text()
    requirements = (ROOT / "docker/requirements-worker.txt").read_text()
    for name in ("transformers", "pillow", "numpy"):
        assert f"{name}=={preprocessing.PINNED_VERSIONS[name]}" in requirements
    for name in ("torch", "torchvision"):
        assert f"ARG {name.upper()}_VERSION={preprocessing.PINNED_VERSIONS[name]}" in recipe
    assert "python:3.11-slim-bookworm" in recipe and "cu126" in recipe
    script = (ROOT / "deploy/docker/infaas-docker.sh").read_text()
    assert '"INSTALL_DALI=$(get INSTALL_DALI 0)"' in script
    assert '-e "MODEL_REVISIONS=$(get MODEL_REVISIONS)"' in script
    assert "TRANSFORMERS_OFFLINE=0" in script and "infaas.cli.doctor" in script
    assert "export DOCKER_CONTEXT" not in script
    ignore = (ROOT / ".dockerignore").read_text()
    for entry in ("data", "inputs", "deploy/docker/*.local.env", ".git"):
        assert entry in ignore.splitlines()


def test_cpu_guide_links_and_keeps_policy_scope_explicit():
    text = (ROOT / "CPU_PREPROCESSING.md").read_text(encoding="utf-8")
    assert text.count("```") % 2 == 0
    assert "register --reprofile" in text
    assert "batch 1" in text and "INSTALL_DALI=1" in text
    assert "CPU_PREPROCESSING.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "/deploy/docker/*.local.env" in (ROOT / ".gitignore").read_text()
