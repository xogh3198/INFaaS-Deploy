# CPU 전처리 → 텐서 큐 → GPU 실행

배포 브랜치: `docker/cpu-ready-v1` (`docker/v2`에서 분기, main 변경 없음).

## GPU 서버에서 코드 받기

기존 실행을 해당 env로 `down`한 뒤 진행합니다. `git status`에 기존 수정이 있으면 먼저 보존하고,
충돌 시 강제 체크아웃하지 마세요. 개인 `.local.env`·가중치·결과는 삭제하지 않습니다.

```bash
cd /home/xsailor4/taeho/serving-repos/infaas_deploy
git status --short
git fetch origin refs/heads/docker/cpu-ready-v1
git checkout -b docker/cpu-ready-v1 FETCH_HEAD
```

이미 브랜치를 만들었다면 `git checkout docker/cpu-ready-v1` 후
`git pull --ff-only origin docker/cpu-ready-v1`로 갱신합니다.

INFaaS의 PIL 경로는 이미 전처리 뒤 모델별 GPU 실행 큐에 텐서를 넣습니다.
이번 변경은 독립 CPU 전처리 풀을 두어 다른 비교 구현과 단계·동시성 기본값을 맞춘 것입니다.
요청 분배·모델 선택·autoscaler·batch 1은 그대로입니다.

```text
JPEG → CPU 준비 대기 → PIL RGB + HF slow processor
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
PREPROCESS_WORKERS=1
PROFILE_MODE=original
PROFILE_REQUIRE_PREPROCESS_MATCH=true
CONTROLLER_IMAGE=infaas-controller:cpu-ready-v1
WORKER_IMAGE=infaas-worker:cpu-ready-v1
PROFILE_DIR=data/infaas/proteus-cpu/cpu-ready-v1/profiles
LOG_DIR=data/infaas/proteus-cpu/cpu-ready-v1/logs
```

`CPU_THREADS`는 intra-op, `PREPROCESS_WORKERS`는 CPU 준비 동시 작업 수입니다. 기본은 작업자 1입니다.
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
프로파일은 `PROFILE_MODE=original`의 준비된 입력 H2D·forward 시간과 포화 QPS입니다.
Proteus의 pack·결과 D2H까지 포함하는 프로파일과 세부 범위까지 동일하다고 해석하면 안 됩니다.
새 fingerprint와 별도 디렉터리로 이전 전처리 포함 프로파일의 혼용을 막습니다.

요청 단계 기록: `cpu_queue`, `cpu_service`, `ready_queue`, `gpu_service`, `postprocess`.
CPU 동시성은 RPC 스레드 수와 구분됩니다. RPC 처리량·모델별 GPU 스트림 등 서빙 구조 차이는 유지합니다.
선택 옵션인 DALI 경로는 이 CPU 비교 조건이 아니므로 별도 실험으로 취급합니다.
