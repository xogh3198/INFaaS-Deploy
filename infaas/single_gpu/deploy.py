"""Standard-library-only Docker launcher; safe to run on the host without torch."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .settings import registry

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "deploy/docker/infaas-single-gpu.env"
OUT = ROOT / "deploy/docker/generated/single-gpu-cost-v1"


def read_env(path):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, equal, value = line.partition("=")
        if not equal:
            raise ValueError(f"Invalid settings line: {line}")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def settings():
    defaults = read_env(DEFAULT)
    local = DEFAULT.with_name("infaas-single-gpu.local.env")
    selected = Path(os.environ.get("INFAAS_SINGLE_GPU_ENV", str(local if local.exists() else DEFAULT))).resolve()
    cfg = {**defaults, **read_env(selected)}
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown single-GPU settings: {sorted(unknown)}")
    for key in ("MODEL_REGISTRY", "MODEL_DIR", "DATA_DIR", "INPUT_IMAGE", "TRACE_FILE"):
        cfg[key] = str((ROOT / Path(cfg[key]).expanduser()).resolve())
    entries, _ = registry(cfg["MODEL_REGISTRY"])
    cfg["MODEL_REVISIONS"] = ",".join(f"{name}:{entry['revision']}" for name, entry in entries.items())
    return cfg, selected


def run(args, capture=False):
    return subprocess.run(list(map(str, args)), check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def gpu_guard(cfg):
    # Never stop somebody else's experiment, change daemon settings, or hide a
    # missing GPU query. Permit an existing instance of this exact container only.
    selected = run(["nvidia-smi", "-i", cfg["GPU"], "--query-gpu=uuid,name", "--format=csv,noheader"], True).strip()
    rows = list(csv.reader(io.StringIO(selected)))
    if len(rows) != 1:
        raise ValueError("Exactly one GPU must be selected")
    uuid, name = (value.strip() for value in rows[0])
    if cfg["GPU_TYPE"] == "v100" and "V100" not in name:
        raise ValueError(f"Selected device is not the requested V100: {name}")
    processes = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], True)
    pids = {row[1].strip() for row in csv.reader(io.StringIO(processes)) if len(row) == 2 and row[0].strip() == uuid}
    own = set()
    existing = subprocess.run(["docker", "top", cfg["CONTAINER_NAME"], "-eo", "pid"],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if existing.returncode == 0:
        own = set(existing.stdout.split()) - {"PID"}
    if pids - own and cfg["ALLOW_GPU_SHARING"] != "1":
        raise ValueError(f"Selected GPU is occupied (PIDs {sorted(pids - own)}). Stop your other experiment first.")


def compose_document(cfg):
    worker_keys = ("APPLICATION", "GPU_TYPE", "HW_COST", "PREPROCESS_BACKEND", "CPU_THREADS",
        "CPU_INTEROP_THREADS", "WORKER_THREADS", "WARMUP_QUERIES", "SLO_MS", "BATCH_SIZES",
        "PROFILE_RUNS", "ALLOCATION_INTERVAL", "EWMA_ALPHA", "INITIAL_RATE", "DEADLINE_GUARD_MS",
        "MIN_ACCURACY", "SLACK_THRESHOLD", "LAMBDA", "TORCH_VERSION", "TORCHVISION_VERSION",
        "MODEL_REVISIONS")
    env = {key: cfg[key] for key in worker_keys}
    env.update(MODEL_REGISTRY="/config/models.json", PROFILE_DIR="/profiles", HF_HOME="/models",
        TORCH_HOME="/models", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", REQUIRE_CUDA="true",
        WORKER_NAME=cfg["CONTAINER_NAME"], PYTHONUNBUFFERED="1")
    # JSON is valid YAML. Dollar escaping prevents accidental Compose substitution.
    esc = lambda value: str(value).replace("$", "$$")
    return {"name": cfg["PROJECT"], "services": {"worker": {
        "image": esc(cfg["WORKER_IMAGE"]), "container_name": cfg["CONTAINER_NAME"],
        "command": ["python", "-m", "infaas.single_gpu.server"],
        "environment": {key: esc(value) for key, value in env.items()},
        "volumes": [f"{esc(cfg['MODEL_DIR'])}:/models", f"{esc(cfg['MODEL_REGISTRY'])}:/config/models.json:ro",
                    f"{esc(Path(cfg['DATA_DIR']) / 'profiles')}:/profiles"],
        "ports": [f"{cfg['PUBLISH_ADDR']}:{cfg['QUERY_PORT']}:50052", f"{cfg['PUBLISH_ADDR']}:{cfg['REGISTER_PORT']}:50053"],
        "deploy": {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "device_ids": [cfg["GPU"]], "capabilities": ["gpu"]}]}}},
        "healthcheck": {"test": ["CMD", "python", "-m", "infaas.single_gpu.client", "health"],
                        "interval": "5s", "timeout": "10s", "retries": 36, "start_period": "20s"},
    }}}


def generate(cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (cfg["MODEL_DIR"], str(Path(cfg["DATA_DIR"]) / "profiles"), str(Path(cfg["DATA_DIR"]) / "results")):
        Path(path).mkdir(parents=True, exist_ok=True)
    (OUT / "docker-compose.yml").write_text(json.dumps(compose_document(cfg), indent=2), encoding="utf-8")
    return ["docker", "compose", "-f", str(OUT / "docker-compose.yml")]


def client(cfg, command, extra):
    image = run(["docker", "inspect", "--format", "{{.Image}}", cfg["CONTAINER_NAME"]], True).strip()
    base = ["docker", "run", "--rm", "--pull=never", "--network", "container:" + cfg["CONTAINER_NAME"]]
    if command in {"register", "query"}:
        if not Path(cfg["INPUT_IMAGE"]).is_file():
            raise ValueError("INPUT_IMAGE is missing")
        base += ["-v", cfg["INPUT_IMAGE"] + ":/inputs/image.jpg:ro"]
    if command == "trace":
        from argparse import ArgumentParser
        parser = ArgumentParser(add_help=False)
        parser.add_argument("--trace", default=cfg["TRACE_FILE"])
        known, remaining = parser.parse_known_args(extra)
        trace = (ROOT / known.trace).resolve()
        if not trace.is_file() or not Path(cfg["INPUT_IMAGE"]).is_file():
            raise ValueError("Trace or JPEG missing; reuse the original Proteus trace")
        results = Path(cfg["DATA_DIR"]) / "results/trace"
        results.mkdir(parents=True, exist_ok=True)
        security = json.loads(run(["docker", "info", "--format", "{{json .SecurityOptions}}"], True))
        uid = "0:0" if "name=rootless" in security else f"{os.getuid()}:{os.getgid()}"
        base += ["--user", uid, "-e", "PYTHONUNBUFFERED=1", "-v", str(trace) + ":/inputs/trace.txt:ro",
            "-v", cfg["INPUT_IMAGE"] + ":/inputs/image.jpg:ro", "-v", cfg["MODEL_REGISTRY"] + ":/config/models.json:ro",
            "-v", str(results) + ":/results"]
        print(f"Host results: {results}", flush=True)
        return run(base + [image, "python", "-m", "infaas.cli.trace_replay", "--application", cfg["APPLICATION"],
            "--slo", cfg["SLO_MS"], "--trace", "/inputs/trace.txt", "--image", "/inputs/image.jpg",
            "--accuracy-registry", "/config/models.json", "--output", "/results", *remaining])
    extra = ["--image", "/inputs/image.jpg", *extra] if command in {"query", "register"} else extra
    if command == "query":
        extra = ["--slo", cfg["SLO_MS"], *extra]
    return run(base + [image, "python", "-m", "infaas.single_gpu.client", command, *extra])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    command, *extra = argv or ["help"]
    if command == "help":
        print("infaas-single-gpu.sh settings|build|download|doctor|up|register|query|trace|state|logs|down")
        return 0
    cfg, selected = settings()
    if command == "settings":
        print(json.dumps({"settings_file": str(selected), **cfg}, indent=2))
        return 0
    print("Docker data-root:", run(["docker", "info", "--format", "{{.DockerRootDir}}"], True).strip(), flush=True)
    if command == "build":
        run(["docker", "build", "-f", ROOT / "docker/worker.Dockerfile", "-t", cfg["WORKER_IMAGE"],
            "--build-arg", "PYTHON_IMAGE=" + cfg["PYTHON_IMAGE"], "--build-arg", "TORCH_VERSION=" + cfg["TORCH_VERSION"],
            "--build-arg", "TORCHVISION_VERSION=" + cfg["TORCHVISION_VERSION"], "--build-arg", "INSTALL_DALI=0", ROOT, *extra])
    elif command == "download":
        Path(cfg["MODEL_DIR"]).mkdir(parents=True, exist_ok=True)
        entries, _ = registry(cfg["MODEL_REGISTRY"])
        run(["docker", "run", "--rm", "-v", cfg["MODEL_DIR"] + ":/models", "-e", "HF_HOME=/models",
            "-e", "HF_HUB_OFFLINE=0", "-e", "TRANSFORMERS_OFFLINE=0", "-e", "MODEL_REVISIONS=" + cfg["MODEL_REVISIONS"],
            cfg["WORKER_IMAGE"], "python", "-m", "infaas.cli.download_models", "--models", ",".join(entries), *extra])
    elif command in {"doctor", "up"}:
        gpu_guard(cfg)
        compose = generate(cfg)
        if command == "doctor":
            run(compose + ["run", "--rm", "--no-deps", "worker", "python", "-m", "infaas.single_gpu.client", "doctor"])
        else:
            run(compose + ["up", "-d"])
            for _ in range(60):
                state = json.loads(run(["docker", "inspect", "--format", "{{json .State}}", cfg["CONTAINER_NAME"]], True))
                if state.get("Health", {}).get("Status") == "healthy":
                    client(cfg, "state", [])
                    break
                if not state.get("Running") or state.get("Health", {}).get("Status") == "unhealthy":
                    raise RuntimeError("Container failed readiness; inspect logs")
                time.sleep(5)
            else:
                raise RuntimeError("Readiness timeout; inspect logs")
    elif command in {"register", "query", "state", "trace"}:
        client(cfg, command, extra)
    elif command in {"down", "logs"}:
        if not (OUT / "docker-compose.yml").exists():
            print("No single-GPU stack generated; legacy containers were not touched.")
        else:
            compose = ["docker", "compose", "-f", str(OUT / "docker-compose.yml")]
            run(compose + (["down"] if command == "down" else ["logs", "--tail=100", *extra]))
    else:
        raise ValueError(f"Unknown command: {command}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
