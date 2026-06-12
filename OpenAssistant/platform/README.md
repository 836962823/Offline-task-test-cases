# Open-Assistant — Discovery 训练任务冒烟（P 集群 GPU）

面向 **P 集群 / CUDA GPU** 的最小 SFT 冒烟：`galactica-125m` + `webgpt_dataset_only`。训练镜像内已内置 Open-Assistant（`OPEN_ASSISTANT_ROOT`）；通过云盘上传本目录后，将 Job command 指向 `run_train.sh` 即可。

## 平台路径约定

| 用途 | 路径 | 说明 |
|------|------|------|
| TensorBoard events | `$TENSORBOARD_LOGDIR`（平台注入） | PVC `/train-events` 下 `jobs/{job}/tb/var/log/training` |
| TB 别名（容器内） | `/var/log/training` | 平台 create 时注入软链 → `$TENSORBOARD_LOGDIR` |
| 训练结果摘要 | `$RESULTS_DIR`（默认 `/data/result/{RUN_ID}`） | `train_result.json`；`RUN_ID` = `TRAIN_RUN_ID` 或 UTC `YYYYMMDD-HHMMSS` |
| Checkpoint | `output_dir`（默认 `/data/result/{RUN_ID}/checkpoints`） | 每次运行独立子目录 |
| HF 缓存 | `$HF_HOME`（默认 `/data/.cache/huggingface`） | 训练结束默认清理（`CLEAN_HF_CACHE_ON_EXIT=0` 可关闭） |

## 周期性 TensorBoard 刷新

- `PlatformTfeventsCallback` 在每次 `on_log`（由 `--logging_steps` 控制，默认 **5**）写入并 **flush** `events.out.tfevents.*`。
- TensorBoard ksvc 使用 `--reload_interval` 时会自动拉取新事件，训练过程中曲线可持续更新。
- 可通过 `PLATFORM_LOGGING_STEPS=10` 调整刷新频率。

## 运行（训练 Job command）

将本 `platform/` 目录放到云盘，例如 `/data/projects/open-assistant/platform/`，然后：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 可选，国内拉 HF
bash /data/projects/open-assistant/platform/run_train.sh
```

或镜像内若已打包：

```bash
bash "${OPEN_ASSISTANT_ROOT}/platform/run_train.sh"
```

### 常用环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `TRAIN_RUN_ID` | UTC 时间戳 | 区分每次 `/data/result/{RUN_ID}` |
| `PLATFORM_LOGGING_STEPS` | `5` | TB 曲线刷新步长 |
| `OA_NUM_EPOCHS` | `1` | 冒烟 epoch 数 |
| `OA_MAX_LENGTH` | `128` | 序列长度（缩短以加快冒烟） |
| `CLEAN_HF_CACHE_ON_EXIT` | `1` | 结束后清理 HF 缓存 |

## Apifox E2E 验证清单（P 集群）

1. **create training**：`code_from: cloud_disk`，`start_cmd` 指向上面的 `run_train.sh`；`tensorboard_events.inject_log_dir_export: true`（平台默认）。
2. **detail**：状态变为 running / succeeded。
3. **detail-monitor**：DCGM GPU 指标有数据。
4. **tensorboard_url**：打开 TB 页面，训练中每 ~`logging_steps` 步应看到 `train/loss` 曲线增长。
5. **云盘结果**：`/data/result/{RUN_ID}/train_result.json` 与 `checkpoints/` 存在。

create 请求示例（字段按实际环境调整）：

```json
{
  "resource_id": 18,
  "mirror_id": 16,
  "training_dataset": "[]",
  "validate_dataset": "[]",
  "code_from": "cloud_disk",
  "start_cmd": "bash /data/projects/open-assistant/platform/run_train.sh",
  "hyper_parameters": {}
}
```

## 目录文件

| 文件 | 作用 |
|------|------|
| `run_train.sh` | 平台入口：路径、补丁、调用 `trainer_sft.py` |
| `platform_tfevents.py` | 轻量 tfevents 写入（无 HF tensorboard 依赖） |
| `platform_callback.py` | 周期性 TB + `train_result.json` |

`trainer_sft.py` 在 `PLATFORM_TRAIN_SMOKE=1` 时自动加载上述 callback（需 `PLATFORM_CALLBACK_DIR` 指向本目录）。若训练镜像尚未内置该改动，`run_train.sh` 会用 `trainer_sft_platform.patch` 在 Job 启动时打补丁。
