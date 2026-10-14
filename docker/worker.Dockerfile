# Independent INFaaS worker, with the same CPU image/ML versions as Proteus.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}
ARG TORCH_VERSION=2.11.0+cu126
ARG TORCHVISION_VERSION=0.26.0+cu126
ARG INSTALL_DALI=0
LABEL org.infaas.preprocessing="proteus-cpu-v1"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
COPY docker/requirements-worker.txt /app/requirements-worker.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-worker.txt
# Optional A/B backend; selecting CPU never imports/runs DALI.
RUN if [ "$INSTALL_DALI" = "1" ]; then \
      python -m pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120; \
    elif [ "$INSTALL_DALI" != "0" ]; then exit 1; fi
ENV HF_HOME=/models TORCH_HOME=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PREPROCESS_BACKEND=pil CPU_THREADS=2 GPU_TYPE=unknown WORKER_PORT=9000 \
    TORCH_VERSION=${TORCH_VERSION} TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=EAGER
COPY infaas /app/infaas
EXPOSE 9000
CMD ["python", "-m", "infaas.worker.main"]
