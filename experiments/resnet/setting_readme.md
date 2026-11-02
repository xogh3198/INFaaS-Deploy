# v1: 사용자가 제공한 INFaaS docker/v1 실행·계측

기준은 Stanford C++ 저장소 전체가 아니라 사용자가 지정한
xogh3198/INFaaS-Deploy docker/v1 @baa65befb0ee551d589926a6d44f7343072ec9eb 이다.
Docker 이미지가 이 커밋을 직접 checkout하고, 시작할 때 커밋과 infaas/ 변경 여부를 확인한다.

기존 Controller·Redis·Worker·Model-Autoscaler·VM-Autoscaler를 실행한다.
이후 추가했던 infaas/single_gpu 비용 순위·SLO/2 배치 선택·배치 대기 실행기는 호출하지 않는다.
선택·배치·큐·마감·프로파일·전처리 코드는 제공된 버전과 바이트 단위로 동일하다.
추가하는 것은 ResNet-101·152 모델 목록, 고정 HF snapshot, V100/경로/포트 배포 설정과 외부 지표 수집이다.
전처리는 제공된 DALI/PIL 흐름을 그대로 실행한다. 미리 만든 입력 텐서로 건너뛰지 않는다.
Torch/CUDA 패키지는 이 서버 V100 실행을 위해 기존 사용 버전인 cu126으로 지정하고 컨테이너에 기록한다.

## 모델·배치 의미

제공된 docker/v1은 모델 아키텍처별 parent_model과 GPU 종류별 variant를 사용한다.
V100 한 종류에서는 ResNet-18→50→152를 자동 선택하는 원래 기능이 없다. 그 기능을 새로 추가하지 않는다.
배치는 제공된 실행기 그대로 1이다. 모델 5개는 각각 등록하고 REQUEST_MODEL로 요청 대상을 선택한다.

기본 REQUEST_MODEL=resnet-50. 각 모델의 기준선을 측정하려면 local.env의 REQUEST_MODEL만 바꾸고 같은 trace를 반복한다.
다른 모델로 바꾸기 전에 down/up으로 같은 초기 상태를 만들며 모든 후보를 GPU에 강제로 함께 올리지 않는다.
PROFILE_MODE=original과 기본 30회 측정은 제공 버전의 프로파일 방식 그대로다.
실제 요청에는 전처리가 남아 있고, 원래 응답의 worker.decode/preprocess/lock_wait/forward/postprocess/load를 계측에 사용한다.

START_STATE 값으로 원래 정책에 reset/warmup 명령을 끼워 넣지 않는다. 실제 시작 상주 상태를 기록한다.
프로파일 뒤 down/up 하면 새 Redis·프로세스가 시작하고 저장된 프로파일을 다시 읽으며 모델은 요청 시 원래 경로에서 로딩한다.

## 실행 순서

기존 수정 파일이 있는 폴더는 그대로 두고 v1 커밋을 별도 폴더에서 실행한다.
기존 저장소 폴더에서 아래를 실행하며, 새 폴더 이름은 저장소별로 다르게 지정한다.

```bash
git fetch origin refs/heads/v1
git worktree add --detach ../INFaaS-source-v1 FETCH_HEAD
cd ../INFaaS-source-v1

systemctl --user start docker.service
unset DOCKER_CONTEXT
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"
docker info --format '{{.DockerRootDir}}'
nvidia-smi -i 2
```

저장 위치는 /data4tb/taeho-proteus/docker-data. V100에 다른 작업이 있으면 해당 작업을 먼저 종료한다.
스크립트가 다른 실험을 종료하거나 Docker daemon 설정을 바꾸지 않는다.

```bash
# 실행할 구현을 아래에서 하나 지정한다.
ENGINE=infaas

bash deploy/docker/source-docker.sh "$ENGINE" settings
bash deploy/docker/source-docker.sh "$ENGINE" build &&
bash deploy/docker/source-docker.sh "$ENGINE" setup &&
bash deploy/docker/source-docker.sh "$ENGINE" up &&
bash deploy/docker/source-docker.sh "$ENGINE" register &&
bash deploy/docker/source-docker.sh "$ENGINE" state

# 프로파일링 종료 후 측정용 프로세스·정책 상태를 새로 시작한다.
bash deploy/docker/source-docker.sh "$ENGINE" down &&
bash deploy/docker/source-docker.sh "$ENGINE" up

bash deploy/docker/source-docker.sh "$ENGINE" trace --dry-run
bash deploy/docker/source-docker.sh "$ENGINE" trace --limit 0 --speed 2 --concurrency 256 --timeout 10
bash deploy/docker/source-docker.sh "$ENGINE" state
bash deploy/docker/source-docker.sh "$ENGINE" down
```

설정 변경은 experiments/resnet/local.env에 필요한 KEY=VALUE만 쓴다. 없으면 settings.env 기본값을 사용한다.
기존 local.env가 있으면 그것이 우선이므로 settings 출력으로 적용 값을 확인한다.
GPU=2, SLO_MS=200, WORKLOAD=twitter, TRACE_SPEED=2가 현재 비교 조건이다.
공개 ResNet trace의 원래 SLO는 300ms이며 200ms는 기존 실험에서 선택한 조건이다. 논문의 SLO를 그대로 재현한 값이라고 쓰지 않는다.
Twitter 목록은 3,776건이고 2배속이면 145.154초·평균 26.01 QPS다. 파일의 도착 시각·QoS·원 SLO와 파일 해시를 기록한다.
배속으로 요청 수가 늘어나지는 않는다.

## 결과 해석

JSON/CSV에 원래 클라이언트 E2E, 성공/늦은 성공/실패·거절/미전송, 요청별 모델·배치·원래 stage timings를 저장한다.
SLO 위반율은 실제 클라이언트 E2E와 실패를 기준으로 계산한다. 성공 응답 지연 분위수는 성공 응답만의 분포다.

measurement_boundaries에는 전처리 wall time, 전처리 완료 뒤 worker 결과까지의 구간, E2E−전처리 wall time을 별도로 저장한다.
DALI 전처리는 GPU 연산을 포함할 수 있으므로 전처리 wall time 전체를 CPU 연산 시간이라고 부르지 않는다.
E2E−전처리는 같은 실행의 산술 분해이며, 전처리를 실제로 제거했을 때의 결과나 원래 SLO 위반율을 대신하지 않는다.
전처리 앞에 있던 큐 대기·모델 로딩은 after_preprocessing_tail에 포함되지 않는다. 원래 배치/전처리 순서가 달라 이 값만으로 전체 서빙을 비교하지 않는다.
모델 로딩은 원래 timings/이벤트에 별도로 기록한다. 측정 구간을 맞추려고 원래 deadline이나 거절 조건을 바꾸지 않는다.

실측 정확도는 측정하지 않는다. 표시되는 모델 점수는 supplied Proteus reference이고 현재 공개 체크포인트의 정답률과 구분한다.
정규화 점수는 다섯 후보의 최고 기준 점수 79.22를 공통 분모로 사용한다. 성공 응답 전체와 SLO 내 성공 응답을 나누어 기록한다.
단일 V100·ResNet 부분 실험이며, 논문 전체 클러스터 결과의 수치 재현이라고 주장하지 않는다.
로컬 CPU 연결·소스 차이 검사는 수행하지만 Docker 빌드와 GPU 실행은 GPU 서버에서 확인해야 한다.

결과 기본 경로:

```text
/data4tb/taeho-proteus/experiments/paired-resnet/infaas-docker-v1/v1/results/
```
