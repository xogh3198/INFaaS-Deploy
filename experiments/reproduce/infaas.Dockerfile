FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git redis-server libgomp1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0+cu126 torchvision==0.26.0+cu126
RUN python -m pip install --no-cache-dir transformers==5.19.0 accelerate safetensors Pillow==12.1.1 numpy==2.2.6 grpcio==1.84.0 grpcio-health-checking==1.84.0 protobuf==7.36.2 redis==5.2.1 nvidia-ml-py kubernetes
RUN python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120==1.50.0
RUN git clone https://github.com/xogh3198/INFaaS-Deploy.git /app && git -C /app checkout baa65befb0ee551d589926a6d44f7343072ec9eb
RUN git config --system --add safe.directory /app
COPY experiments/reproduce /audit/experiments/reproduce
COPY experiments/resnet /audit/experiments/resnet
ENV PYTHONPATH=/audit:/app PYTHONUNBUFFERED=1 HF_HOME=/models TORCH_HOME=/models HF_HUB_OFFLINE=1 CUDA_MODULE_LOADING=EAGER
WORKDIR /app
CMD ["python","-m","experiments.reproduce.infaas_entry","infaas.worker.main"]
