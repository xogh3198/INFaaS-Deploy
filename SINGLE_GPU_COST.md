# 단일 GPU 모델·배치 선택 — INFaaS 비용 포트

기존 고정 모델 실험은 유지합니다. 이 문서의 `infaas-single-gpu.sh`는 **추가 구현된 단일 장치 비교 모드**입니다.
Proteus의 GPU 코드나 MILP를 호출하지 않습니다. 이 저장소의 INFaaS GPU 실행기와 비용 함수를 사용합니다.

## 1. 무엇이 달라지는가

| 항목 | 기존 스택 | 새 비용 포트 |
|---|---|---|
| 실행 스크립트 | infaas-docker.sh | infaas-single-gpu.sh |
| 요청 대상 | resnet-50 고정 | imagenet 작업 |
| 선택 후보 | resnet-50__v100 | ResNet-18·50 × 배치 1·2·4·8 |
| GPU | V100 한 장 | 동일, GPU 상주 모델 최대 1개 |
| CPU 전처리 | 요청 처리 스레드의 PIL RGB + HF slow | 유지, 전처리 후 CPU 텐서를 준비 큐로 전달 |
| GPU 실행 | 요청별 batch 1 | 준비 텐서 결합·H2D·배치 추론·결과 회수 |
| 선택 기준 | 기존 배포의 get_variant/분산 확장 경로 | 아래 비용 함수의 단일-GPU 제약 포트 |
| 구조 | 프런트엔드·Redis·워커·VM autoscaler | 동일 native API를 제공하는 GPU 워커 한 개 |

새 모드에서 다중 워커 증설·VM 제어·원본 분산 확장 알고리즘 전체를 재현하지 않습니다.
결과 표의 이름은 **INFaaS single-GPU cost port**로 구분하세요. 기존 스택 결과와 같은 구현이라고 합치지 않습니다.
서빙 구조 축약에 따른 RPC 경로 차이도 있으므로 과거 e2e 수치와 알고리즘 효과만으로 직접 비교하지 않습니다.

## 2. 선택 기준

기존 `infaas/policy/costs.py`의 `instance_cost`, `action_cost`를 직접 사용합니다.

```text
C = HW_COST[GPU] × max(프로파일 peak GPU memory / 1e9, memory floor)
Cost(delta) = C × (delta + LAMBDA × load_seconds × max(delta, 0))

현재 모델 유지/배치 변경: score = C_candidate
다른 모델로 교체/최초 로딩: score = Cost(+1 candidate)
```

단일 상주 모델 교체에서는 `C_current + Cost(+1 new) + Cost(-1 current)`의 기존 모델 비용이 상쇄됩니다.
따라서 로딩 비용을 포함한 위 점수로 후보를 비교합니다. **정확도 최대화 목적함수로 교체하지 않았습니다.**

1. `MIN_ACCURACY` 이상이고, 준비된 배치의 프로파일 P50이 `SLO_MS/2`보다 작은 후보만 허용합니다.
2. `capacity_qps >= EWMA 요청률 × SLACK_THRESHOLD`인 후보 중 비용 점수가 가장 낮은 구성을 고릅니다.
3. 처리량 조건을 모두 만족하지 못하면 가장 높은 GPU 처리 용량 후보를 택하고 `capacity_sufficient=false`로 기록합니다.
4. 정확도는 최소 조건 및 비용 동률 해소에만 사용합니다. ResNet-18이 모든 부하에서 더 유리하면 전환하지 않아도 정상입니다.

배치 프로파일은 전처리가 끝난 입력의 결합·H2D·forward·결과 회수 구간이며 GPU 처리량이지 전체 JPEG 서빙 처리량은 아닙니다.
CPU 전처리·로딩·전송 시간은 실제 e2e에 포함됩니다. CPU 처리 후 요청 마감시각을 새로 설정하지 않습니다.
다음 배치를 기다리는 마지막 시각은 `원래 마감시각 − 다음 배치 예상 실행시간 − DEADLINE_GUARD_MS`입니다.
모델 최초 로딩/교체는 실제로 수행하므로 해당 요청이 SLO를 위반할 수 있습니다. 모델 전환을 강제로 만들지 않습니다.

## 3. 설정

```bash
cp -n deploy/docker/infaas-single-gpu.env deploy/docker/infaas-single-gpu.local.env
export INFAAS_SINGLE_GPU_ENV="$PWD/deploy/docker/infaas-single-gpu.local.env"
bash deploy/docker/infaas-single-gpu.sh settings
```

기본값: GPU=2, GPU_TYPE=v100, SLO_MS=200, BATCH_SIZES=1,2,4,8, PROFILE_RUNS=30,
CPU_THREADS=2, CPU_INTEROP_THREADS=1, WORKER_THREADS=32, WARMUP_QUERIES=10,
ALLOCATION_INTERVAL=1, EWMA_ALPHA=0.7, INITIAL_RATE=1, DEADLINE_GUARD_MS=2,
SLACK_THRESHOLD=1.05, LAMBDA=1.0.

새 CPU 풀을 1개로 제한하지 않고 기존 INFaaS 요청 처리 스레드의 전처리 동시성을 유지합니다.
이는 Proteus의 PREPROCESS_WORKERS=1과 다른 구현 조건입니다.

모델은 `deploy/docker/infaas-single-gpu.models.json`에 두 개 모두 enabled=true로 포함합니다.
모델·revision·accuracy 변경은 이 JSON을 수정하고 settings → download → up → register --reprofile 순서로 적용합니다.
배치·CPU·GPU·전처리·가중치를 바꾼 뒤에는 재프로파일링합니다. 같은 설정의 서버 재시작은 호환 프로파일을 재사용합니다.
모델/배치 등의 런타임 설정만 바꾸면 일반적으로 재빌드가 필요 없고, 코드·ML 패키지·베이스 이미지를 바꾼 경우에 build합니다.

77.36/70.93은 원 Proteus의 ResNet-50/18 **참조 정확도**입니다. 현재 HF 가중치에서 직접 검증한 Top-1이 아닙니다.

## 4. 기존 실험을 종료한 뒤 실행

현재 터미널이 개인 Docker를 선택했는지 먼저 확인합니다. 스크립트가 Docker context/data-root를 바꾸지 않습니다.

```bash
docker info --format '{{.DockerRootDir}}'

# 현재 실행 중인 기존 INFaaS만 종료. 가중치와 결과는 삭제하지 않습니다.
INFAAS_ENV="$PWD/deploy/docker/infaas-proteus-cpu.local.env" \
  bash deploy/docker/infaas-docker.sh down

bash deploy/docker/infaas-single-gpu.sh build &&
bash deploy/docker/infaas-single-gpu.sh download &&
bash deploy/docker/infaas-single-gpu.sh doctor &&
bash deploy/docker/infaas-single-gpu.sh up &&
bash deploy/docker/infaas-single-gpu.sh register --reprofile &&
bash deploy/docker/infaas-single-gpu.sh query -n 11
```

포트 50052·50053을 재사용하므로 기존 스택과 동시에 띄우지 않습니다.
다른 실험의 GPU 점유는 기본 거부하며 다른 컨테이너를 강제로 종료하지 않습니다.

## 5. 같은 트레이스로 측정

기존 Proteus 트레이스를 읽습니다. 새 트레이스를 생성하거나 수정하지 않습니다.
이전 루트의 `replay-infaas-trace.sh`는 고정 모델 클라이언트이므로 **새 모드에서는 아래 명령을 사용**합니다.

```bash
bash deploy/docker/infaas-single-gpu.sh trace --dry-run
bash deploy/docker/infaas-single-gpu.sh trace --limit 100 --speed 1 --concurrency 64 --timeout 10
# 100건 결과 확인 후 전체 요청
bash deploy/docker/infaas-single-gpu.sh trace --limit 0 --speed 1 --concurrency 64 --timeout 10
bash deploy/docker/infaas-single-gpu.sh state
```

TRACE_FILE 기본값은 `../Proteus/workloads/twitter-resnet-slo200.txt`입니다. 필요하면
`trace --trace /absolute/path/trace.txt ...`로 같은 파일을 명시합니다.
QoS는 기록만 하고 모든 요청에 env의 SLO 200ms를 적용합니다. 같은 JPEG를 반복하며 자동 reset/warmup은 하지 않습니다.

CSV/JSON: `data/infaas/single-gpu-cost-v1/results/trace/`.
프로파일: `data/infaas/single-gpu-cost-v1/profiles/single-gpu-cost-v1.json`.
이전 고정 모델 실험의 프로파일·로그·결과는 그대로 둡니다.

확인할 값:

- `successful_requests_by_model`: 실제 응답 모델별 요청 수.
- `actual_gpu_batch_execution_counts`: 재생 구간에 실행한 실제 GPU 배치 크기별 횟수. 요청별 배치 크기 개수와 구분합니다.
- `actual_model_switches_after_initial_load`: 최초 로딩을 제외한 실제 GPU 상주 모델 교체 횟수.
- `worker_before/after.plan_history`, `load_history`: 선택 이유·비용·요청률·실제 로딩 내역.
- SLO 미충족률, P50/P95/P99, Goodput, 기준 정확도, 클라이언트 미전송/전송 지연.

부하가 GPU 후보의 용량 경계를 넘지 않거나 한 모델이 계속 최저 비용이면 전환 0회일 수 있습니다.
그때는 `state`의 배치별 실측 capacity와 plan_details를 확인합니다. 배속 실험은 다른 서빙에도 같은 배속을 적용하고 별도 조건으로 기록합니다.

```bash
bash deploy/docker/infaas-single-gpu.sh state --full
bash deploy/docker/infaas-single-gpu.sh logs
bash deploy/docker/infaas-single-gpu.sh down
```

로컬 CPU 단위/가짜 GPU·실제 gRPC 연결 테스트는 수행하지만, V100의 가중치 로드·성능·전환 결과는 서버에서 위 순서로 검증해야 합니다.
