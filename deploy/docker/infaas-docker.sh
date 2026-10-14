#!/usr/bin/env bash
# INFaaS on docker (one host, one worker container per GPU). See setting_readme.md.
#
#   deploy/docker/infaas-docker.sh gpus                 GPU indices + a suggested GPUS line
#   deploy/docker/infaas-docker.sh build                build controller/worker images
#   deploy/docker/infaas-docker.sh doctor               CPU processor/cache + selected GPU check
#   deploy/docker/infaas-docker.sh download [--models a,b]   fetch weights into MODEL_DIR
#   deploy/docker/infaas-docker.sh gen                  write generated/docker-compose.yml
#   deploy/docker/infaas-docker.sh up                   gen + start everything, wait for workers
#   deploy/docker/infaas-docker.sh down                 stop and remove the containers
#   deploy/docker/infaas-docker.sh ps | logs [service]  status / logs (docker compose)
#   deploy/docker/infaas-docker.sh state                workers, variants, states (Metadata Store)
#   deploy/docker/infaas-docker.sh register [--all|--models a,b] [--image F]   profile + register
#   deploy/docker/infaas-docker.sh query MODEL IMAGE SLO_MS [-n N]  one request, native API
#
# INFAAS_ENV=<file> selects another settings file (default: infaas-docker.env here).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${INFAAS_ENV:-$HERE/infaas-docker.env}"
if [[ -z "${INFAAS_ENV:-}" && -f "$HERE/infaas-docker.local.env" ]]; then
  ENV_FILE="$HERE/infaas-docker.local.env"
fi
cd "$ROOT"
OUT="$HERE/generated"
COMPOSE=(docker compose -f "$OUT/docker-compose.yml")

get() {  # get KEY [default] — last KEY=VALUE in the settings file
  local v
  v="$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
  v="${v%\"}"; v="${v#\"}"
  echo "${v:-${2:-}}"
}
CONTROLLER_IMAGE="$(get CONTROLLER_IMAGE infaas-controller:0.1.0)"
WORKER_IMAGE="$(get WORKER_IMAGE infaas-worker:0.1.0)"
NETWORK="$(get NETWORK bridge)"

# how a one-off container reaches the controller
if [ "$NETWORK" = "host" ]; then
  NET_ARGS=(--network host); CTRL=127.0.0.1; REDIS=127.0.0.1
else
  NET_ARGS=(--network infaas); CTRL=infaas-controller; REDIS=infaas-redis
fi

gen() { python3 "$HERE/gen_compose.py" --env "$ENV_FILE" --out "$OUT" "$@"; }

cmd="${1:-help}"; shift || true
case "$cmd" in
  settings)
    printf 'Settings file: %s\n' "$ENV_FILE"
    for key in GPUS HW_COST MODEL_NAME MODEL_REVISIONS INPUT_IMAGE SLO_MS MODEL_DIR PROFILE_DIR LOG_DIR PROFILE_MODE PREPROCESS_BACKEND CPU_THREADS TORCH_VERSION TORCHVISION_VERSION INSTALL_DALI; do
      printf '%s=%s\n' "$key" "$(get "$key")"
    done
    echo 'BATCH_SIZE=1 (this deployment does not implement configurable batching)' ;;
  gpus)
    python3 "$HERE/gen_compose.py" --suggest ;;
  build)
    docker build -f "$ROOT/docker/controller.Dockerfile" -t "$CONTROLLER_IMAGE" "$ROOT"
    docker build -f "$ROOT/docker/worker.Dockerfile" -t "$WORKER_IMAGE" \
      --build-arg "PYTHON_IMAGE=$(get PYTHON_IMAGE python:3.11-slim-bookworm)" \
      --build-arg "TORCH_VERSION=$(get TORCH_VERSION 2.11.0+cu126)" \
      --build-arg "TORCHVISION_VERSION=$(get TORCHVISION_VERSION 0.26.0+cu126)" \
      --build-arg "INSTALL_DALI=$(get INSTALL_DALI 0)" "$ROOT" ;;
  doctor)
    gen
    services=$(python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads("\n".join(p.read_text().splitlines()[1:])); print("\n".join(n for n,s in d["services"].items() if "WORKER_NAME" in s.get("environment", {})))' "$OUT/docker-compose.yml")
    [[ -n "$services" ]] || { echo 'No local GPU workers selected.' >&2; exit 1; }
    while IFS= read -r service; do
      "${COMPOSE[@]}" run --rm --no-deps "$service" python -m infaas.cli.doctor --model "$(get MODEL_NAME resnet-50)"
    done <<< "$services" ;;
  download)
    MODEL_DIR="$(get MODEL_DIR /data/podexec-models)"
    MODEL_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$MODEL_DIR")"
    mkdir -p "$MODEL_DIR"
    download_args=(); selected=0; all_models=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --all) all_models=1; shift ;;
        --models|--models=*) selected=1; download_args+=("$1"); shift ;;
        *) download_args+=("$1"); shift ;;
      esac
    done
    if [[ "$all_models" -eq 1 && "$selected" -eq 1 ]]; then
      echo 'Choose --all or --models, not both.' >&2; exit 1
    fi
    if [[ "$all_models" -eq 0 && "$selected" -eq 0 ]]; then
      download_args=(--models "$(get MODEL_NAME resnet-50)" "${download_args[@]}")
    fi
    docker run --rm -v "$MODEL_DIR:/models" -e HF_HOME=/models -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
      -e "MODEL_REVISIONS=$(get MODEL_REVISIONS)" \
      "$WORKER_IMAGE" python -m infaas.cli.download_models "${download_args[@]}" ;;
  gen)
    gen "$@" ;;
  up)
    gen "$@"
    "${COMPOSE[@]}" up -d --remove-orphans
    echo ">> waiting for workers to report healthy (library preload takes a while)..."
    for _ in $(seq 1 90); do
      total=0; ok=0
      while IFS= read -r id; do
        [[ -n "$id" ]] || continue
        row=$(docker inspect --format '{{.Name}} {{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$id")
        if [[ "$row" == *-worker-* ]]; then
          total=$((total + 1))
          if [[ "$row" == *' healthy' ]]; then ok=$((ok + 1)); fi
        fi
      done < <("${COMPOSE[@]}" ps --all --quiet)
      echo "   workers healthy: $ok/$total"
      [ "$total" -gt 0 ] && [ "$ok" -eq "$total" ] && break
      sleep 5
    done
    if [[ "$total" -eq 0 || "$ok" -ne "$total" ]]; then
      echo 'ERROR: workers did not become healthy; inspect logs/state.' >&2
      "${COMPOSE[@]}" ps
      exit 1
    fi
    "${COMPOSE[@]}" ps ;;
  down)
    "${COMPOSE[@]}" down --remove-orphans ;;
  ps)
    "${COMPOSE[@]}" ps "$@" ;;
  logs)
    "${COMPOSE[@]}" logs --tail=100 "$@" ;;
  state)
    docker run --rm "${NET_ARGS[@]}" "$CONTROLLER_IMAGE" \
      python -m infaas.cli.state --redis "$REDIS:16379" "$@" ;;
  register)
    # PROFILE_MODE=original needs no image; service needs --image FILE
    vol=(); img_arg=(); args=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --image) vol=(-v "$(realpath "$2"):/in/image.jpg:ro"); img_arg=(--image /in/image.jpg); shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    selected=0
    for option in "${args[@]}"; do
      case "$option" in --all|--models|--models=*|--import-dir|--import-dir=*) selected=1 ;; esac
    done
    if [[ "$selected" -eq 0 ]]; then args=(--models "$(get MODEL_NAME resnet-50)" "${args[@]}"); fi
    if [[ ${#img_arg[@]} -eq 0 && "$(get PROFILE_MODE original)" == service ]]; then
      image="$(get INPUT_IMAGE inputs/test-dog.jpg)"
      [[ -f "$image" ]] || { echo "Input missing: $image" >&2; exit 1; }
      vol=(-v "$(realpath "$image"):/in/image.jpg:ro"); img_arg=(--image /in/image.jpg)
    fi
    docker run --rm "${NET_ARGS[@]}" "${vol[@]}" "$CONTROLLER_IMAGE" \
      python -m infaas.cli.register --controller "$CTRL:50053" "${img_arg[@]}" "${args[@]}" ;;
  query)
    if [[ $# -gt 0 && "$1" != -* ]]; then
      [[ $# -ge 3 ]] || { echo 'Usage: query [MODEL IMAGE SLO_MS] [-n N]' >&2; exit 1; }
      model="$1"; img="$2"; slo="$3"; shift 3
    else
      model="$(get MODEL_NAME resnet-50)"; img="$(get INPUT_IMAGE inputs/test-dog.jpg)"; slo="$(get SLO_MS 200)"
    fi
    [[ -f "$img" ]] || { echo "Input missing: $img" >&2; exit 1; }
    docker run --rm "${NET_ARGS[@]}" -v "$(realpath "$img"):/in/image.jpg:ro" "$CONTROLLER_IMAGE" \
      python -m infaas.cli.online_query --controller "$CTRL:50052" \
      --model "$model" --image /in/image.jpg --slo "$slo" "$@" ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
