# Platform training callbacks: periodic tfevents flush + train_result.json.

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any, Optional

from transformers import TrainerCallback

from platform_tfevents import PlatformTfeventsWriter, resolve_platform_tb_logdir

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments


def _is_rank_zero() -> bool:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank == 0


def _results_dir() -> str:
    default = "/data/result"
    return os.path.realpath(os.environ.get("RESULTS_DIR", default))


class PlatformTfeventsCallback(TrainerCallback):
    """Write train metrics as events.out.tfevents.*; flush each on_log for live TB refresh."""

    def __init__(self) -> None:
        self._writer: Optional[PlatformTfeventsWriter] = None
        self._tb_dir = ""
        self._tb_source = ""

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not _is_rank_zero():
            return
        self._tb_dir, self._tb_source = resolve_platform_tb_logdir()
        self._writer = PlatformTfeventsWriter(self._tb_dir)
        print(f"[platform] TB logdir: {self._tb_dir} ({self._tb_source})")

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self._writer is None or not state.log_history:
            return
        last = state.log_history[-1]
        metrics: dict[str, float] = {}
        if last.get("loss") is not None:
            metrics["train/loss"] = float(last["loss"])
        if last.get("learning_rate") is not None:
            metrics["train/learning_rate"] = float(last["learning_rate"])
        if last.get("epoch") is not None:
            metrics["train/epoch"] = float(last["epoch"])
        if last.get("eval_loss") is not None:
            metrics["eval/loss"] = float(last["eval_loss"])
        if metrics:
            self._writer.add_scalars(int(state.global_step), metrics)

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self._writer is not None:
            path = self._writer.close()
            print(f"[platform] TB events -> {path}")


class PlatformResultCallback(TrainerCallback):
    """Write train_result.json under RESULTS_DIR (/data/result/{RUN_ID} by default)."""

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not _is_rank_zero():
            return

        results_dir = _results_dir()
        os.makedirs(results_dir, exist_ok=True)

        last: dict[str, Any] = state.log_history[-1] if state.log_history else {}
        tb_dir, tb_source = resolve_platform_tb_logdir()
        run_id = (os.environ.get("TRAIN_RUN_ID") or "").strip() or os.path.basename(results_dir.rstrip("/"))
        payload = {
            "run_id": run_id,
            "global_step": int(state.global_step),
            "epochs": float(last.get("epoch", 0) or 0),
            "loss": last.get("loss"),
            "output_dir": args.output_dir,
            "tb_logdir": tb_dir,
            "tb_logdir_source": tb_source,
            "results_dir": results_dir,
            "written_at": int(time.time()),
        }
        manifest = os.path.join(results_dir, "train_result.json")
        with open(manifest, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
            fp.write("\n")
        print(f"[platform] train result -> {manifest}")
