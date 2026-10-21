# INFaaS-Deploy

GPU·모델·SLO 설정과 단계별 실행은 **[setting_readme.md](setting_readme.md)**를 참고하세요.

Proteus와 동일한 CPU 전처리로 재실험하려면 **[CPU_PREPROCESSING.md](CPU_PREPROCESSING.md)**를 사용하세요.

`docker/v4` 브랜치에서 독립 INFaaS Docker 배포를 실행합니다.
기존 요청 API 예시는 [docker.md](docker.md)에 있습니다.
기존 `infaas-docker.sh` 배포는 batch 1이며, 모델 학습은 수행하지 않습니다.

단일 V100에서 ResNet-18·50 및 배치 1·2·4·8을 비용 기준으로 선택하는 **추가 구현 모드**는
[SINGLE_GPU_COST.md](SINGLE_GPU_COST.md)를 사용하세요. 기존 스택과 별개이며,
INFaaS의 분산 오토스케일러 전체를 그대로 재현했다는 의미는 아닙니다.
