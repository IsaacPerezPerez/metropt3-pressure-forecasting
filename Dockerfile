FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow==2.22.0 pandas==2.2.3
WORKDIR /workspace
ENV PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
