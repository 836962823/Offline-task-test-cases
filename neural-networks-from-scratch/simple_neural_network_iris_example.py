# -*- coding: utf-8 -*-
"""Train a simple NumPy neural network and emit TB/results artifacts.

This script aligns with current training path conventions:
- TensorBoard events are written to /var/log/training by default.
- Training artifacts are written under /data/data/{run_id} by default.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import time
from typing import BinaryIO

import matplotlib.pyplot as plt
import numpy as np

TB_LOGDIR_DEFAULT = "/var/log/training"
RESULTS_DIR_DEFAULT = "/data/data"

_CRC32C_TABLE = None


def _crc32c_table():
  poly = 0x82F63B78
  table = []
  for i in range(256):
    crc = i
    for _ in range(8):
      crc = (crc >> 1) ^ poly if crc & 1 else crc >> 1
    table.append(crc & 0xFFFFFFFF)
  return table


def _crc32c(data, crc=0):
  global _CRC32C_TABLE
  if _CRC32C_TABLE is None:
    _CRC32C_TABLE = _crc32c_table()
  crc = crc ^ 0xFFFFFFFF
  for b in data:
    crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
  return crc ^ 0xFFFFFFFF


def _masked_crc(data):
  x = _crc32c(data) & 0xFFFFFFFF
  return (((x >> 15) | ((x << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


def _encode_varint(n):
  out = bytearray()
  while True:
    b = n & 0x7F
    n >>= 7
    if n:
      out.append(b | 0x80)
    else:
      out.append(b)
      break
  return bytes(out)


def _key(field, wire):
  return _encode_varint((field << 3) | wire)


def _encode_double(field, v):
  return _key(field, 1) + struct.pack("<d", v)


def _encode_int64(field, v):
  return _key(field, 0) + _encode_varint(v)


def _encode_string(field, s):
  b = s.encode("utf-8")
  return _key(field, 2) + _encode_varint(len(b)) + b


def _encode_float(field, v):
  return _key(field, 5) + struct.pack("<f", v)


def _encode_bytes(field, b):
  return _key(field, 2) + _encode_varint(len(b)) + b


def _scalar_value(tag, value):
  return _encode_string(1, tag) + _encode_float(2, value)


def _summary(values):
  return b"".join(_encode_bytes(1, _scalar_value(tag, value)) for tag, value in values)


def _event(step, wall_time, values):
  summary = _summary(values)
  return _encode_double(1, wall_time) + _encode_int64(2, step) + _encode_bytes(5, summary)


def _write_record(fp, data):
  length = len(data)
  len_bytes = struct.pack("<Q", length)
  fp.write(len_bytes)
  fp.write(struct.pack("<I", _masked_crc(len_bytes)))
  fp.write(data)
  fp.write(struct.pack("<I", _masked_crc(data)))


class MinimalTBWriter:
  """TensorBoard writer fallback when SummaryWriter is unavailable."""

  def __init__(self, logdir):
    os.makedirs(logdir, exist_ok=True)
    host = socket.gethostname()[:50]
    filename = f"events.out.tfevents.{int(time.time())}.{host}.{os.getpid()}"
    self.path = os.path.join(logdir, filename)
    self.fp: BinaryIO = open(self.path, "wb")

  def add_scalars(self, step, metrics):
    _write_record(self.fp, _event(step, time.time(), list(metrics.items())))
    self.fp.flush()

  def close(self):
    self.fp.close()

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
    self.weights_input_to_hidden = 0.01 * np.random.rand(input_size, hidden_size)
    self.weights_hidden_to_output = 0.01 * np.random.rand(hidden_size, output_size)

    self.bias_hidden = np.zeros((1, hidden_size))
    self.bias_output = np.zeros((1, output_size))

    self.learning_rate = learning_rate

  def relu(self, x):
    return np.maximum(0, x)

  def relu_derivative(self, x):
    return (x>0).astype(float)

  def softmax(self, x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

  def compute_loss(self, y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss

  def forward(self, x):
    self.z_hidden = np.dot(x, self.weights_input_to_hidden) + self.bias_hidden
    self.a_hidden = self.relu(self.z_hidden)

    self.z_output = np.dot(self.a_hidden, self.weights_hidden_to_output) + self.bias_output
    self.a_output = self.softmax(self.z_output)

    return self.a_output

  def backward(self, x, y_true):
    m = y_true.shape[0]

    # Gradients for output layer
    dz_output = self.a_output - y_true
    dw_hidden_to_output = np.dot(self.a_hidden.T, dz_output) / m
    db_output = np.sum(dz_output, axis=0, keepdims=True) / m

    # Gradients for hidden layer
    da_hidden = np.dot(dz_output, self.weights_hidden_to_output.T)
    dz_hidden = da_hidden * self.relu_derivative(self.z_hidden)
    dw_input_to_hidden = np.dot(x.T, dz_hidden) / m
    db_hidden = np.sum(dz_hidden, axis=0, keepdims=True) / m

    return dw_input_to_hidden, db_hidden, dw_hidden_to_output, db_output

  def update_parameters(self, dw_input_to_hidden, db_hidden, dw_hidden_to_output, db_output):
    self.weights_input_to_hidden -= self.learning_rate * dw_input_to_hidden
    self.bias_hidden -= self.learning_rate * db_hidden

    self.weights_hidden_to_output -= self.learning_rate * dw_hidden_to_output
    self.bias_output -= self.learning_rate * db_output

  def train(self, x, y_true, epochs, tb_writer=None, log_every=100):
    loss_history = []  # List to store loss values
    for epoch in range(epochs):
      # forward pass
      y_pred = self.forward(x)

      # compute loss
      loss = self.compute_loss(y_true, y_pred)
      loss_history.append(loss)  # Store the loss value

      # backward pass
      dw_input_to_hidden, db_hidden, dw_hidden_to_output, db_output = self.backward(x, y_true)

      # update parameters
      self.update_parameters(dw_input_to_hidden, db_hidden, dw_hidden_to_output, db_output)

      train_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
      if tb_writer is not None:
        tb_writer.add_scalars(epoch, {"train/loss": float(loss), "train/accuracy": float(train_acc)})

      # print loss
      if epoch % log_every == 0:
        print(f"Epoch {epoch}, loss: {loss}")

    return loss_history

  def predict(self, x):
    y_pred = self.forward(x)
    return np.argmax(y_pred, axis=1)

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def _to_abs(path):
  return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def _run_id(explicit=None):
  run_id = (explicit or os.environ.get("TRAIN_RUN_ID") or "").strip()
  if not run_id:
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
  return run_id


def _resolve_results_dir(results_dir_arg, run_id_arg=None):
  """Use RESULTS_DIR env if set; else {results_dir_arg}/{run_id} (skip duplicate suffix)."""
  env_dir = (os.environ.get("RESULTS_DIR") or "").strip()
  if env_dir:
    path = _to_abs(env_dir)
    rid = (os.environ.get("TRAIN_RUN_ID") or run_id_arg or os.path.basename(path.rstrip("/"))).strip()
    return path, _run_id(rid)
  base = _to_abs(results_dir_arg)
  rid = _run_id(run_id_arg)
  if os.path.basename(base.rstrip("/")) == rid:
    return base, rid
  return os.path.join(base, rid), rid


def _build_tb_writer(logdir):
  return MinimalTBWriter(logdir), "minimal-writer"


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--epochs", type=int, default=5000)
  parser.add_argument("--learning-rate", type=float, default=0.025)
  parser.add_argument("--hidden-size", type=int, default=20)
  parser.add_argument(
    "--tb-logdir",
    default=os.environ.get("TENSORBOARD_LOGDIR", TB_LOGDIR_DEFAULT),
    help=f"TensorBoard log dir (default: env TENSORBOARD_LOGDIR or {TB_LOGDIR_DEFAULT})",
  )
  parser.add_argument(
    "--results-dir",
    default=RESULTS_DIR_DEFAULT,
    help=f"Base directory for results; a run subdir is appended unless RESULTS_DIR is set (default base: {RESULTS_DIR_DEFAULT})",
  )
  parser.add_argument(
    "--run-id",
    default="",
    help="Run subdirectory name (default: TRAIN_RUN_ID env or UTC timestamp YYYYMMDD-HHMMSS)",
  )
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def main():
  args = parse_args()
  np.random.seed(args.seed)

  tb_logdir = _to_abs(args.tb_logdir)
  results_dir, run_id = _resolve_results_dir(args.results_dir, args.run_id or None)
  os.makedirs(tb_logdir, exist_ok=True)
  os.makedirs(results_dir, exist_ok=True)

  # Load and prepare the Iris dataset.
  iris_data = load_iris()
  x = iris_data.data
  y_ = iris_data.target.reshape(-1, 1)
  encoder = OneHotEncoder(sparse_output=False)
  y = encoder.fit_transform(y_)
  train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=args.seed)

  # Initialize and train the network.
  nn = NeuralNetwork(
    input_size=4,
    hidden_size=args.hidden_size,
    output_size=3,
    learning_rate=args.learning_rate,
  )
  tb_writer, writer_backend = _build_tb_writer(tb_logdir)
  print(f"TensorBoard logdir: {tb_logdir} (backend={writer_backend})")
  print(f"TRAIN_RUN_ID: {run_id}")
  print(f"Results dir: {results_dir}")
  print("Training the network...")
  loss_history = nn.train(train_x, train_y, epochs=args.epochs, tb_writer=tb_writer, log_every=100)
  tb_writer.close()

  # Evaluate on the test split.
  print("Testing the network...")
  y_pred = nn.forward(test_x)
  y_pred_labels = np.argmax(y_pred, axis=1)
  test_y_labels = np.argmax(test_y, axis=1)
  accuracy = float(np.mean(y_pred_labels == test_y_labels))
  print(f"Accuracy: {accuracy * 100:.2f}%")

  # Save loss curve and run summary under /data (or overridden results dir).
  loss_plot_path = os.path.join(results_dir, "loss_curve.png")
  plt.figure()
  plt.plot(range(0, args.epochs, 100), loss_history[::100])
  plt.xlabel("Epochs (every 100 steps)")
  plt.ylabel("Loss")
  plt.title("Loss over Epochs")
  plt.tight_layout()
  plt.savefig(loss_plot_path)
  plt.close()

  metrics_path = os.path.join(results_dir, "train_result.json")
  metrics = {
    "run_id": run_id,
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "hidden_size": args.hidden_size,
    "accuracy": accuracy,
    "final_loss": float(loss_history[-1]) if loss_history else None,
    "tb_logdir": tb_logdir,
    "results_dir": results_dir,
    "written_at": int(time.time()),
  }
  with open(metrics_path, "w", encoding="utf-8") as fp:
    json.dump(metrics, fp, indent=2)
    fp.write("\n")

  print(f"Saved: {loss_plot_path}")
  print(f"Saved: {metrics_path}")


if __name__ == "__main__":
  main()