# Discovery training smoke image: a2-cann + LlamaFactory (platform tfevents, no HF tensorboard)
FROM registry2.d.pjlab.org.cn/lepton-trainingjob/a2-cann:8.3.rc2-910b-ubuntu22.04-py3.11

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.cache/huggingface \
    RESULTS_DIR=/data/data \
    TB_LOGDIR=/var/log/training \
    PLATFORM_TRAIN_SMOKE=1

WORKDIR /app

# Copy trimmed LlamaFactory tree (see platform/README.md)
COPY . /app

RUN if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then \
      source /usr/local/Ascend/ascend-toolkit/set_env.sh; \
    fi && \
    pip install -U pip setuptools wheel packaging editables "hatchling>=1.18.0" && \
    pip uninstall -y torch torchvision torchaudio || true && \
    pip install --no-cache-dir -r requirements/npu.txt && \
    pip install --no-cache-dir -e . --no-build-isolation && \
    chmod +x platform/run_train.sh

VOLUME ["/var/log/training", "/data", "/train-events"]

ENTRYPOINT ["/app/platform/run_train.sh"]
