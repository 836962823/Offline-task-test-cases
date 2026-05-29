# Offline-task-test-cases

Discovery **离线训练任务**端到端冒烟用例集合，与 `discovery-ml-be` 的 TrainJob / TensorBoard / 数据集 PVC 能力配套使用。

## 用例列表

| 用例 | 路径 | 说明 |
|------|------|------|
| **Neural（Iris）** | 仓库 [`neural-networks-from-scratch`](../neural-networks-from-scratch/) | NumPy 小网络；**不需 NPU**|
| **LlamaFactory NPU SFT** | [`LlamaFactory/`](./LlamaFactory/) | 昇腾 LoRA 冒烟；`platform/run_train.sh` + `train_npu_platform.yaml` |

统一运行镜像：[`discovery-train-image`](../discovery-train-image/) → `registry2.d.pjlab.org.cn/ccr-discovery/train-npu-smoke:v1-a2cann-arm64`。


## LlamaFactory 子目录

| 文件 | 作用 |
|------|------|
| `platform/run_train.sh` | 平台入口（`TRAIN_RUN_ID`、数据集解析、`output_dir`） |
| `platform/train_npu_platform.yaml` | LoRA SFT 最小配置 |
| `platform/README.md` | 路径与数据集说明 |
| `data/dataset_info.json` + `data/alpaca_en_demo.json` | 演示数据集 |

## 快速命令（容器内）

```bash
# Neural（无需 source CANN）
bash /usr/local/bin/run-neural-smoke.sh

# LlamaFactory（建议 train-entrypoint 加载 Ascend）
export HF_ENDPOINT=https://hf-mirror.com #没有该命令从HF拉取权重时可能会出现超时问题
export DATASET_DIR=/mnt/datasets/<user>/<dataset> 
bash /app/platform/run_train.sh
```

##  输出结果示例
```
/data/
├── data/
│   ├── 20260528-120001/          ← Neural 一次运行
│   │   ├── train_result.json
│   │   └── loss_curve.png
│   └── 20260528-143052/          ← Llama 一次运行
│       └── train_result.json
├── sft_output/
│   └── 20260528-143052/          ← Llama 独有：checkpoint
│       ├── adapter_config.json
│       └── ...
└── .cache/huggingface/           ← Llama 独有：下载的基座模型
```