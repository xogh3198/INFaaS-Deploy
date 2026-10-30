# 독립 INFaaS: 세 모델과 동일 2배속 트레이스

기존 infaas-single-gpu.sh 비용 정책을 유지합니다. ResNet-101을 카탈로그에 추가하고
별도 설정·레지스트리를 제공했습니다. 기존 기본 env·두 모델 결과는 수정하지 않습니다.
일반 분산 infaas-docker.sh와 별개의 단일 GPU 비용 어댑터입니다.

## 설정

- V100 GPU 2 한 대, FP32, ResNet-18·50·101의 checkpoint pin과 사전 점수는 Proteus와 동일.
- 사전 점수는 70.93 / 77.36 / 78.34이며 현재 HF 체크포인트 정답률을 측정한 값은 아닙니다.
- PIL RGB + HF slow CPU 전처리 후 GPU 준비 큐, intra-op 2 / inter-op 1.
- 기존 요청 핸들러 32개에서 전처리합니다. 별도 풀이나 PREPROCESS_WORKERS는 추가하지 않습니다.
  Proteus/INFaaS-Accuracy의 CPU 풀 8개와 동시성 구조가 다릅니다.
- 비용 기준·HW_COST·LAMBDA·slack 1.05·배치 1,2,4,8·기존 요청률 계산 유지.
- GPU 구간 프로파일 30회. CPU 전처리는 개별 요청 마감에서 차감하며 GPU 프로파일에
  CPU 큐 대기나 클라이언트 시간을 합산하지 않습니다.
- 동일 JPEG·Twitter 3,776건·speed 2·SLO 200ms·동시 요청 256·RPC 타임아웃 10초.
  재생 시간 145.154초, 평균 26.014 QPS.

## 실행

다른 실험을 종료한 개인 rootless Docker 터미널에서 실행합니다.

```bash
cd /home/xsailor4/taeho/serving-repos/infaas_deploy
export INFAAS_SINGLE_GPU_ENV="$PWD/deploy/docker/infaas-three-model.env"
bash deploy/docker/infaas-single-gpu.sh settings
bash deploy/docker/infaas-single-gpu.sh build &&
bash deploy/docker/infaas-single-gpu.sh download &&
bash deploy/docker/infaas-single-gpu.sh up &&
bash deploy/docker/infaas-single-gpu.sh register --reprofile
```

등록은 GPU 모델을 해제하므로 한 번 요청해서 active 상태로 만듭니다.
이 한 요청은 트레이스에 포함하지 않습니다. 트레이스의 before/after 카운터 차이를 사용합니다.

```bash
bash deploy/docker/infaas-single-gpu.sh query -n 1
bash deploy/docker/infaas-single-gpu.sh trace \
  --limit 0 --speed 2 --concurrency 256 --timeout 10
bash deploy/docker/infaas-single-gpu.sh state
```

결과: /data4tb/taeho-proteus/experiments/infaas-three-model-v1/results/trace/

```bash
bash deploy/docker/infaas-single-gpu.sh down
```

비용이 가장 작은 후보가 부하를 감당한다고 판단되면 ResNet-18이 유지될 수 있습니다.
모델 변경을 강제하지 않습니다. 실제 V100 로딩·SLO 검증은 서버 결과로 확인합니다.
