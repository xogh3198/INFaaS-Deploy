# Proteus와 동일한 CPU 전처리로 INFaaS 재실험

`docker/v2`에서 실행합니다. **전처리·ML 버전을 맞추되 INFaaS의 요청 분배,
워커 실행 구조, 모델/VM autoscaler와 batch 1은 유지**합니다. Proteus 서비스에 요청을 보내지 않습니다.

## 1. 통일한 조건

| 항목 | 비교 설정 |
|---|---|
| GPU | 호스트 2번 V100 한 대, 컨테이너에서는 CUDA 0 |
| 모델 | microsoft/resnet-50, FP32 |
| 가중치·processor revision | `34c2154c194f829b11125337b98c8f5f9965ff19` |
| 요청 입력·SLO | 기존 `inputs/test-dog.jpg`, 200ms |
| 전처리 | CPU PIL 디코딩·RGB → HF AutoImageProcessor `use_fast=False` → CPU 텐서 |
| 추론 | CPU 텐서 H2D → GPU forward. CPU 추론 대체 금지 |
| CPU 스레드 | PyTorch intra-op 2. CPU 코어 독점/서비스 전체 스레드 제한은 아님 |
| ML 버전 | Python 3.11, Torch 2.11.0+cu126, torchvision 0.26.0+cu126, CUDA 12.6 |
| 전처리 라이브러리 | Transformers 5.19.0, Pillow 12.1.1, NumPy 2.2.6 |
| 모델 준비 | WARMUP_QUERIES=10, TF32·cuDNN benchmark 비활성 |
| 프로파일 | `PROFILE_MODE=service`: 실제 JPEG의 전처리 포함, 평균 지연·포화 QPS 측정 |

Proteus의 배치별 p50 프로파일과 INFaaS의 평균/포화 QPS 프로파일은 정책이 요구하는
통계가 달라 그대로 유지합니다. 프로파일 알고리즘·콜드 준비 절차까지 동일해진 것은 아닙니다.

## 2. 코드 갱신과 개인 설정

서버의 INFaaS 저장소에서 실행합니다. `git status --short`에 기존 수정이 나오면
먼저 보존하고 진행합니다. Docker context/data-root는 스크립트가 변경하지 않습니다.

```bash
cd /home/xsailor4/taeho/infaas_deploy
git status --short
git fetch origin
git checkout docker/v2
git pull --ff-only origin docker/v2
docker info --format '{{.DockerRootDir}}'
nvidia-smi -i 2
```

기존 INFaaS가 실행 중이면 **기존 설정으로** 먼저 종료합니다. 다른 실험이 같은 GPU를
사용 중이면 본인이 소유한 해당 실험에서 종료하고 GPU가 비었는지 확인합니다.

```bash
bash deploy/docker/infaas-docker.sh down
```

위 명령은 기존 generated Compose 파일이 있을 때 실행합니다. 모델/결과 디렉터리는 삭제하지 않습니다.

```bash
cp -n deploy/docker/infaas-proteus-cpu.env deploy/docker/infaas-proteus-cpu.local.env
export INFAAS_ENV="$(pwd)/deploy/docker/infaas-proteus-cpu.local.env"
bash deploy/docker/infaas-docker.sh settings
```

GPU·입력 경로 변경이 필요하면 `infaas-proteus-cpu.local.env`를 수정합니다.
이 파일은 Git에 올리지 않습니다. 별도 이미지 태그와 프로파일/로그 디렉터리를 사용하므로
이전 DALI 프로파일/로그는 보존됩니다. 프런트엔드 포트·컨테이너 이름은 기존 INFaaS와 같아서
둘을 동시에 띄우는 설정은 아닙니다. 새 SSH 터미널에서는 `INFAAS_ENV`를 다시 선택합니다.

## 3. 빌드 → 다운로드 → 확인 → 등록 → 요청

기존 worker 이미지는 CUDA/라이브러리가 다르므로 반드시 재빌드합니다.
각 명령 성공을 확인한 뒤 다음 명령을 실행합니다.

```bash
bash deploy/docker/infaas-docker.sh build
bash deploy/docker/infaas-docker.sh download
bash deploy/docker/infaas-docker.sh doctor
bash deploy/docker/infaas-docker.sh up
bash deploy/docker/infaas-docker.sh state
bash deploy/docker/infaas-docker.sh register --reprofile
bash deploy/docker/infaas-docker.sh query -n 11
bash deploy/docker/infaas-docker.sh logs infaas-worker-2
```

기존 JPEG를 그대로 사용합니다. 입력이 없다면 새 이미지를 임의로 만들지 말고 이전
Proteus/INFaaS에서 사용한 파일을 가져오세요. 입력 확인:

```bash
sha256sum inputs/test-dog.jpg
```

기존 비교 JPEG의 SHA256은 `f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a`입니다.

## 4. 정상 동작 확인

- `doctor`: `ok: true`, V100, CUDA 12.6, CPU 텐서, `cpu_threads: 2`, pinned revision/ML 버전 확인.
  processor/cache/CUDA 검사이며 모델 추론 성공을 대신하지 않습니다.
- 워커 로그: `PIL RGB + HF slow AutoImageProcessor (CPU)`, 모델 로딩 `[PIL]`.
- 새 프로파일: `profile_mode: service`, `dali: false`, `preprocessing_config` 기록.
- 첫 요청: `path=inactive`, `load>0`이면 cold. 이미 active면 cold로 보고하지 않습니다.
- 이후 요청: `path=active`, `load=0`, 같은 분류 결과. `dali=0`이면 실제 PIL 경로.
- `OK`는 요청 처리 성공, `slo=PASS/MISS`는 **클라이언트 e2e 기준** SLO 판정입니다.
  첫 로딩 요청은 200ms를 넘을 수 있습니다.

```bash
bash deploy/docker/infaas-docker.sh down
```

요청 11회는 cold/warm 기능 확인이며 정확도·부하 처리량 비교는 아닙니다.
GPU 실행 전·후 모델 미상주/상주 상태를 구분하고, 이전 DALI 수치와 새 CPU 수치를 따로 기록합니다.

## 5. DALI 옵션으로 되돌리기

개인 설정에서 `PREPROCESS_BACKEND=dali`, `INSTALL_DALI=1`로 바꿉니다.
`PROFILE_DIR`·`LOG_DIR`·이미지 태그도 별도로 정하고 `build → download → up → register --reprofile`합니다.
`PREPROCESS_MODES`는 DALI 선택 시에만 사용합니다. DALI 미설치인데 `dali`를 요청하면
워커가 실패하며 CPU 경로를 DALI라고 표시하지 않습니다. 지원되지 않는 모델은 기존의
모델별 PIL 대체가 가능하므로 실제 `dali` 플래그도 확인해야 합니다.

여기서 DALI 옵션은 새 ML 버전 위의 A/B 경로입니다. 이전 CUDA 12.8 이미지의 완전한 복원은
아니며, 기존 `docker/v1` 브랜치와 이전 이미지/결과는 따로 보존합니다.
