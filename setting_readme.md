# INFaaS Docker 설정 및 실행

저장소 루트에서 실행합니다. 사전 학습 가중치로 실제 GPU 추론을 수행합니다.

**Proteus와 전처리를 통일하는 V100·ResNet-50 실험은 [CPU_PREPROCESSING.md](CPU_PREPROCESSING.md)**에
별도 전체 env와 재실행 명령을 정리했습니다. 기존 DALI 결과는 보존합니다.

## 1. 코드와 Docker 확인

```bash
git clone --branch docker/v2 --single-branch https://github.com/xogh3198/INFaaS-Deploy.git infaas_deploy
cd infaas_deploy
docker info --format '{{.DockerRootDir}}'
bash deploy/docker/infaas-docker.sh gpus
```

현재 터미널에서 선택한 GPU 사용 가능 Docker를 사용합니다. 동일 GPU의 다른 실험은 먼저 종료하세요.

## 2. 개인 설정 만들기

```bash
cp -n deploy/docker/infaas-docker.env deploy/docker/infaas-docker.local.env
vim deploy/docker/infaas-docker.local.env
```

INFaaS local.env는 **전체 기본 파일을 복사한 설정**입니다. 필요한 줄만 수정하세요.
주석은 별도 줄에 작성합니다.

```dotenv
GPUS=2:v100
HW_COST=v100:1
MODEL_NAME=resnet-50
INPUT_IMAGE=inputs/test-dog.jpg
SLO_MS=200
MODEL_DIR=data/infaas/models
PROFILE_DIR=data/infaas/profiles
LOG_DIR=data/infaas/logs
PUBLISH_ADDR=127.0.0.1
PROFILE_MODE=original
PROFILE_RUNS=30
PREPROCESS_BACKEND=pil
CPU_THREADS=2
MODEL_REVISIONS=resnet-50:34c2154c194f829b11125337b98c8f5f9965ff19
```

- `GPUS`: 사용할 호스트 GPU와 유형. 2번 한 장이면 `2:v100`.
- `MODEL_NAME`: 지원 목록의 모델 이름. 예: `resnet-50`, `resnet-18`, `resnet-34`.
  목록은 [infaas/vendor/lumina/models.py](infaas/vendor/lumina/models.py)에 있습니다.
- `PROFILE_MODE=original`: 원본 방식의 random tensor 추론 프로파일.
  `service`: `INPUT_IMAGE`의 디코딩·전처리를 포함한 실측. 모델 교체와 함께 선택하세요.
- 현재 이 배포는 **batch 1 고정**입니다. `BATCH_SIZE`를 env에 추가해도 배치는 바뀌지 않습니다.
- `PREPROCESS_BACKEND=pil`: CPU PIL RGB + HF slow processor. GPU에는 전처리가 끝난 텐서를 전송합니다.
- `CPU_THREADS=2`: PyTorch intra-op 설정. CPU pinning과는 다릅니다.
- `MODEL_REVISIONS`: `모델이름:40자리커밋` 목록. 다운로드와 워커 로딩에 동일하게 적용됩니다.
  ResNet-50 기본값은 Proteus와 같은 pinned revision입니다. 다른 모델은 원하는 revision을 명시하세요.
- 새 worker ML 버전은 Proteus와 맞춥니다. 기존 이미지로 실행하지 말고 첫 사용 시 재빌드하세요.

local.env는 자동 선택됩니다. 다른 전체 설정 파일은 `INFAAS_ENV=/절대경로/파일.env`로 선택합니다.

## 3. 최초 실행

`INPUT_IMAGE`에 사용할 기존 JPEG를 준비한 뒤 실행합니다.
각 명령이 성공한 뒤 다음 단계로 진행하세요.

```bash
bash deploy/docker/infaas-docker.sh settings
bash deploy/docker/infaas-docker.sh build
bash deploy/docker/infaas-docker.sh download
bash deploy/docker/infaas-docker.sh doctor
bash deploy/docker/infaas-docker.sh up
bash deploy/docker/infaas-docker.sh state
bash deploy/docker/infaas-docker.sh register
bash deploy/docker/infaas-docker.sh query -n 11
bash deploy/docker/infaas-docker.sh logs infaas-worker-2
bash deploy/docker/infaas-docker.sh down
```

`download`와 `register`는 기본적으로 env의 모델 하나를 사용합니다.
전체 모델은 `download --all`·`register --all`로 별도 선택합니다.
`query -n 11`은 env의 모델·입력·SLO로 순차 요청을 보냅니다.

기존 명령 형식도 그대로 사용할 수 있습니다.

```bash
bash deploy/docker/infaas-docker.sh download --models resnet-18
bash deploy/docker/infaas-docker.sh register --models resnet-18 --reprofile
bash deploy/docker/infaas-docker.sh query resnet-18 inputs/test-dog.jpg 200 -n 11
```

## 4. 모델·환경 변경 후 재실험

**기존 설정으로 `down` → local.env 수정 → `settings` → 필요한 준비 → `up`** 순서입니다.

| 변경 | 추가로 할 일 |
|---|---|
| 모델 | `download` → `up` → `register --reprofile` |
| GPU·프로파일 방식·전처리 | `up` → `register --reprofile` |
| DALI 사용 여부 | `INSTALL_DALI=1`로 설치 필요 시 `build`; backend 변경 뒤 재프로파일링 |
| SLO·입력 | `query`에 적용. `PROFILE_MODE=service`에서 입력도 바꿨다면 재프로파일링 |
| 코드·Dockerfile·라이브러리 버전 | `build` → `up` → `register --reprofile` |

모델·GPU·요청 SLO 변경만으로는 **재빌드가 필요 없습니다**.
`GPUS`를 바꾸면 `HW_COST`에도 해당 유형을 적고 로그의 워커 번호를 맞추세요.
모델 캐시·프로파일·로그는 각 설정 경로에 남습니다. 새 실험의 cold 요청을 확실히 확인하려면
다른 `PROFILE_DIR`·`LOG_DIR`로 분리하고 `up` → `register --reprofile` → `query` 순서로 측정하세요.
첫 요청이 이미 active라면 cold 측정으로 기록하지 마세요.

## 5. 결과 확인

- `path=inactive`, `load>0`: 모델 준비 비용이 포함된 요청.
- `path=active`, `load=0`: 이미 GPU에 올라간 모델로 처리한 요청.
- `e2e`: 요청 왕복 시간. `decide`: 컨트롤러 선택 시간. `OK`는 SLO 만족을 뜻하지 않습니다.
- `dali=0`: 실제 PIL 경로. `slo=PASS/MISS`: 클라이언트 e2e 기준.
- 상태와 로그는 `state`·`logs`, 저장 파일은 `PROFILE_DIR`·`LOG_DIR`에서 확인합니다.
- 학습은 하지 않습니다. 이미지 반복 요청은 정확도 평가가 아닙니다.

기존 API 예시는 [docker.md](docker.md)를 참고하세요.
