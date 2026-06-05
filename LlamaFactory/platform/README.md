# LlamaFactory — Discovery 训练任务冒烟

面向 **a2-cann / 昇腾 NPU** 的最小 SFT 冒烟：镜像/app路径下已内置LLamaFactory文件夹，运行代码以该路径为标准，如果是通过云盘或文件上传等方式运行需要修改运行路径

## 平台路径约定

| 用途 | 路径 | 说明 |
|------|------|------|
| TensorBoard events | `$TENSORBOARD_LOGDIR` 或 `/var/log/training` | 由 `PlatformTfeventsCallback` 写 `events.out.tfevents.*`，**不用** HF `report_to: tensorboard` |
| 训练结果摘要 | `$RESULTS_DIR`（默认 `/data/data/{RUN_ID}`） | `train_result.json`；`RUN_ID`=`TRAIN_RUN_ID` 或 UTC `YYYYMMDD-HHMMSS` |
| LoRA / checkpoint | `output_dir`（默认 `/data/sft_output/{RUN_ID}`） | `run_train.sh` 按时间戳建子目录 |
| 数据集 | yaml `dataset_dir`（默认 `/data/datasets`） | 需含 `dataset_info.json` + 数据 json |
| 模型缓存 | `$HF_HOME`（默认 `/data/.cache/huggingface`） | 首次从 HF 拉权重；**每次训练结束自动清理**（见下） |


## 运行（训练 Job command 可设为）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 可选，国内拉 HF 权重
/app/platform/run_train.sh
```

### 模型缓存清理

云盘 PVC 上若中断下载，会留下损坏的 Hugging Face 缓存（如 `config.json` 解析失败）。`run_train.sh` 在**训练结束**（成功或失败）后默认删除 `$HF_HOME` 下全部内容，避免影响下次任务。

| 变量 | 默认 | 说明 |
|------|------|------|
| `CLEAN_HF_CACHE_ON_EXIT` | `1` | 设为 `0` 可保留缓存（加速重复跑同一模型，但有脏缓存风险） |
| `HF_HOME` | `/data/.cache/huggingface` | 与 `HF_HUB_CACHE` / `TRANSFORMERS_CACHE` 对齐 |

LoRA checkpoint（`/data/sft_output/{RUN_ID}`）与 `train_result.json` **不会被清理**。

## 数据集挂载示例

平台默认挂载点 `/data/datasets/`；若实际挂在其它路径（如 `/mnt/datasets/user/name`），任选其一：

```bash
export DATASET_DIR=/mnt/datasets/a123/llama-test
/app/platform/run_train.sh
# 或
/app/platform/run_train.sh dataset_dir=/mnt/datasets/a123/llama-test
```

```
/data/datasets/   # 或你的 DATASET_DIR / dataset_dir= 路径
  dataset_info.json   # 至少包含 alpaca_en_demo
  alpaca_en_demo.json
```

`run_train.sh` 会检测 `dataset_info.json`：找到则用该目录；否则回退镜像内 `/app/data/` 并打印明确提示。