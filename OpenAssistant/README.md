# Open-Assistant（离线用例）

本目录**不是** Open-Assistant 完整源码，只放 **Discovery 平台对接层**（`platform/`）。

## 为什么文件这么少？

| 对比 | LlamaFactory 用例 | Open-Assistant 用例 |
|------|-------------------|---------------------|
| 训练代码在哪 | `Offline-task-test-cases/LlamaFactory/src/` 内嵌整份 LlamaFactory（用于打 NPU 镜像） | **训练镜像内已内置**整份 Open-Assistant（`OPEN_ASSISTANT_ROOT`，约 1500+ 文件） |
| 本仓库放什么 | `platform/` 入口脚本 + 完整 `src/` | 仅 `platform/` 入口脚本、TB callback、patch |
| 完整源码仓库 | 同目录 `src/` | 独立仓库 [`Open-Assistant`](../../Open-Assistant/) |

Open-Assistant 原始项目包含 `model/`、`website/`、`inference/`、`docs/` 等大量目录；平台冒烟只需要：

- 调用镜像里的 `trainer_sft.py`（`model/model_training/`）
- 云盘上传本目录的 `platform/run_train.sh` 作为 Job `start_cmd`

因此**不必**把整份 Open-Assistant 再复制进 `Offline-task-test-cases`（避免与镜像、`AILab/Open-Assistant` 三份重复维护）。

## 目录结构

```text
OpenAssistant/
├── README.md                 ← 本说明
└── platform/
    ├── run_train.sh          ← 平台 Job 入口（上传到云盘）
    ├── platform_callback.py  ← 周期性 TB + train_result.json
    ├── platform_tfevents.py
    ├── trainer_sft_platform.patch
    └── README.md             ← 路径、Apifox E2E
```

## 相关路径

| 位置 | 路径 |
|------|------|
| 完整源码（开发/改 trainer） | `AILab/Open-Assistant/` |
| P 集群 GPU 训练镜像 | `training-gpu-a100:latest`（内含 `OPEN_ASSISTANT_ROOT`） |
| K8s 直连冒烟脚本 | `discovery-k8s-resource/p-cluster/kubeflow/trainer/examples/run-oa-sft-smoke-gpu1.sh` |

若需要像 LlamaFactory 一样在本仓库内嵌完整源码（例如自建 GPU 镜像 Dockerfile），可以说一下，再补 `src/` 子目录或 submodule 方案。
