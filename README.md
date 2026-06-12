# Offline-task-test-cases

Discovery **离线训练任务**端到端冒烟用例集合，与 `discovery-ml-be` 的 TrainJob / TensorBoard / 数据集 PVC 能力配套使用。

## 用例列表

| 用例 | 路径 | 说明 |
|------|------|------|
| **Neural（Iris）** | 仓库 [`neural-networks-from-scratch`](../neural-networks-from-scratch/) | NumPy 小网络；**不需 NPU**|
| **LlamaFactory NPU SFT** | [`LlamaFactory/`](./LlamaFactory/) | 昇腾 LoRA 冒烟；`platform/run_train.sh` + `train_npu_platform.yaml` |
| **Open-Assistant GPU SFT** | [`OpenAssistant/`](./OpenAssistant/) | P 集群 GPU 冒烟；`galactica-125m` + `webgpt`；TB `/var/log/training`，结果 `/data/result/{RUN_ID}` |

选用镜像：`cann 8.3rc2 / pytorch 2.8.0 / torch_npu 2.8.0 / python 3.11 arrm64(训练任务测试用)`。


## Open-Assistant 子目录（P 集群 GPU）

| 文件 | 作用 |
|------|------|
| `platform/run_train.sh` | 平台入口（`TRAIN_RUN_ID`、TB 软链、`/data/result/{RUN_ID}`） |
| `platform/platform_callback.py` | 周期性写 tfevents + `train_result.json` |
| `platform/README.md` | 路径、TB 刷新、Apifox E2E 清单 |

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash /data/projects/open-assistant/platform/run_train.sh
```

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
python3 /workspace/neural/simple_neural_network_iris_example.py --epochs 100

# LlamaFactory（建议 train-entrypoint 加载 Ascend）
export HF_ENDPOINT=https://hf-mirror.com #没有该命令从HF拉取权重时可能会出现超时问题
export DATASET_DIR=/mnt/datasets/<user>/<dataset> 
bash /app/platform/run_train.sh
# 训练结束会自动清理 $HF_HOME 缓存，避免云盘残留损坏权重影响下次运行（CLEAN_HF_CACHE_ON_EXIT=0 可关闭）
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
├── result/
│   └── 20260609-120001/          ← Open-Assistant 一次运行
│       ├── train_result.json
│       └── checkpoints/
├── sft_output/
│   └── 20260528-143052/          ← Llama 独有：checkpoint
│       ├── adapter_config.json
│       └── ...
└── .cache/huggingface/           ← Llama / OA：下载的基座模型
```

TB 事件（train-events PVC，非云盘）：

```text
/train-events/jobs/{train_job_name}/tb/var/log/training/events.out.tfevents.*
# 训练容器内别名：/var/log/training -> 同上
```