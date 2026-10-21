# CPU 전처리 → 텐서 큐 → GPU 실행

배포 브랜치: `docker/v4` (`docker/v2`에서 분기, main 변경 없음).
`docker/v4` 기준 보정 패치: 기존 설정을 유지하고 미지정 inter-op만 1로 명시합니다.
전체 실험 설정과 유지/변경 범위는 [EXPERIMENT_SETTINGS.md](EXPERIMENT_SETTINGS.md)에 정리했습니다.

## GPU 서버에서 코드 받기

기존 실행을 해당 env로 `down`한 뒤 진행합니다. `git status`에 기존 수정이 있으면 먼저 보존하고,
충돌 시 강제 체크아웃하지 마세요. 개인 `.local.env`·가중치·결과는 삭제하지 않습니다.

```bash
cd /home/xsailor4/taeho/serving-repos/infaas_deploy
git status --short
git fetch origin refs/heads/docker/v4
git checkout -b docker/v4 FETCH_HEAD
```

이미 브랜치를 만들었다면 `git checkout docker/v4` 후
`git pull --ff-only origin docker/v4`로 갱신합니다.

INFaaS의 PIL 경로는 이미 전처리 뒤 모델별 GPU 실행 큐에 텐서를 넣습니다.
전처리는 기존처럼 요청 처리 스레드에서 수행합니다. 추가했던 단일 CPU 풀은 제거했습니다.
요청 분배·모델 선택·autoscaler·batch 1은 그대로입니다.

```text
JPEG → 기존 요청 처리 스레드의 PIL RGB + HF slow processor
 → 준비된 CPU 텐서의 모델별 실행 큐 → H2D·GPU forward → 응답
```

전처리 후 SLO를 다시 시작하지 않습니다. 첫 모델 로딩은 기존대로 요청 처리에 포함됩니다.
현재 INFaaS 경로는 모델을 먼저 준비한 뒤 해당 processor로 전처리하므로,
콜드 요청 자체의 전처리/모델 로딩을 동시에 시작하는 기능을 추가한 것은 아닙니다.

## 갱신 순서

1. 기존 설정으로 해당 INFaaS를 `down`합니다. 다른 사람의 서비스는 종료하지 않습니다.
2. `infaas-proteus-cpu.local.env`의 모델/가중치 경로는 보존하고 아래 값을 갱신합니다.

```dotenv
PREPROCESS_BACKEND=pil
CPU_THREADS=2
CPU_INTEROP_THREADS=1
FRONTEND_THREADS=64
WORKER_THREADS=32
PROFILE_MODE=service
PROFILE_REQUIRE_PREPROCESS_MATCH=true
CONTROLLER_IMAGE=infaas-controller:cpu-ready-v1
WORKER_IMAGE=infaas-worker:cpu-ready-v1
PROFILE_DIR=data/infaas/proteus-cpu/preserved-settings-v1/profiles
LOG_DIR=data/infaas/proteus-cpu/preserved-settings-v1/logs
```

`CPU_THREADS`는 intra-op입니다. 전처리는 원래의 요청 처리 스레드에서 실행되며,
별도 전처리 작업자 수로 제한하지 않습니다. 기존 개인 env의 `PREPROCESS_WORKERS`는 삭제하세요.
남겨 두더라도 INFaaS에서는 무시하고 안내를 출력합니다. 64/32는 기존 코드 기본값을 명시한 것입니다.
새 env를 만들 때만 기본 preset을 복사하고, 기존 개인 env 전체를 덮어쓰지 마세요.

```bash
export INFAAS_ENV="$PWD/deploy/docker/infaas-proteus-cpu.local.env"
docker info --format '{{.DockerRootDir}}'
bash deploy/docker/infaas-docker.sh settings
bash deploy/docker/infaas-docker.sh build
bash deploy/docker/infaas-docker.sh doctor
bash deploy/docker/infaas-docker.sh up
bash deploy/docker/infaas-docker.sh register --reprofile
bash deploy/docker/infaas-docker.sh query -n 11
```

가중치가 없을 때만 `download`를 먼저 실행합니다. 기존 캐시/결과는 삭제하지 않습니다.
비교 preset의 프로파일은 기존 `PROFILE_MODE=service`로 복구했습니다.
실제 JPEG 디코딩·전처리·H2D·forward·후처리를 포함한 평균 지연과 실측 포화 QPS입니다.
기본 infaas-docker.env의 `PROFILE_MODE=original` 옵션 자체는 기존대로 남아 있습니다.
Proteus의 GPU 준비 텐서 프로파일과 측정 범위가 다르며, 이를 같다고 해석하면 안 됩니다.
새 fingerprint와 별도 디렉터리로 잘못 변경했던 original 모드 결과의 혼용을 막습니다.

요청 단계 기록: `cpu_service`, `ready_queue`, `gpu_service`, `postprocess`.
`cpu_queue=0`은 별도 전처리 큐가 없다는 뜻이지, RPC 접수 대기까지 없다는 뜻은 아닙니다.
CPU 동시성은 RPC 스레드 수와 구분됩니다. RPC 처리량·모델별 GPU 스트림 등 서빙 구조 차이는 유지합니다.
선택 옵션인 DALI 경로는 이 CPU 비교 조건이 아니므로 별도 실험으로 취급합니다.
