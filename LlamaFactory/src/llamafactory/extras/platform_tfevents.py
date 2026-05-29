# Copyright 2025 the LlamaFactory team.
#
# Minimal TensorBoard event writer for Discovery training platform.
# Writes events.out.tfevents.* under TENSORBOARD_LOGDIR (or /var/log/training alias).

from __future__ import annotations

import os
import socket
import struct
import time
from typing import BinaryIO

_CRC32C_TABLE: list[int] | None = None

# Platform alias (discovery-ml-be tensorboardprep/constants.go)
TB_LOGDIR_ALIAS = "/var/log/training"


def _crc32c_table() -> list[int]:
    poly = 0x82F63B78
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ poly if crc & 1 else crc >> 1
        table.append(crc & 0xFFFFFFFF)
    return table


def _crc32c(data: bytes, crc: int = 0) -> int:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        _CRC32C_TABLE = _crc32c_table()
    crc = crc ^ 0xFFFFFFFF
    for b in data:
        crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> int:
    x = _crc32c(data) & 0xFFFFFFFF
    return (((x >> 15) | ((x << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


def _encode_varint(n: int) -> bytes:
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


def _key(field: int, wire: int) -> bytes:
    return _encode_varint((field << 3) | wire)


def _encode_double(field: int, v: float) -> bytes:
    return _key(field, 1) + struct.pack("<d", v)


def _encode_int64(field: int, v: int) -> bytes:
    return _key(field, 0) + _encode_varint(v)


def _encode_string(field: int, s: str) -> bytes:
    b = s.encode("utf-8")
    return _key(field, 2) + _encode_varint(len(b)) + b


def _encode_float(field: int, v: float) -> bytes:
    return _key(field, 5) + struct.pack("<f", v)


def _encode_bytes(field: int, b: bytes) -> bytes:
    return _key(field, 2) + _encode_varint(len(b)) + b


def _scalar_value(tag: str, value: float) -> bytes:
    return _encode_string(1, tag) + _encode_float(2, value)


def _summary(values: list[tuple[str, float]]) -> bytes:
    return b"".join(_encode_bytes(1, _scalar_value(tag, value)) for tag, value in values)


def _event(step: int, wall_time: float, values: list[tuple[str, float]]) -> bytes:
    summary = _summary(values)
    return _encode_double(1, wall_time) + _encode_int64(2, step) + _encode_bytes(5, summary)


def _write_record(fp: BinaryIO, data: bytes) -> None:
    length = len(data)
    len_bytes = struct.pack("<Q", length)
    fp.write(len_bytes)
    fp.write(struct.pack("<I", _masked_crc(len_bytes)))
    fp.write(data)
    fp.write(struct.pack("<I", _masked_crc(data)))


def _to_abs_path(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser((path or "").strip())))


def resolve_platform_tb_logdir() -> tuple[str, str]:
    """Resolve TB directory: TENSORBOARD_LOGDIR (platform) > TB_LOGDIR > /var/log/training."""
    env_logdir = (os.environ.get("TENSORBOARD_LOGDIR") or "").strip()
    if env_logdir:
        return _to_abs_path(env_logdir), "TENSORBOARD_LOGDIR"

    alias = (os.environ.get("TB_LOGDIR") or TB_LOGDIR_ALIAS).strip()
    return _to_abs_path(alias), "TB_LOGDIR alias"


class PlatformTfeventsWriter:
    """Writes TensorBoard 2.x compatible events.out.tfevents.* files."""

    def __init__(self, logdir: str) -> None:
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        host = socket.gethostname()[:50]
        fname = f"events.out.tfevents.{int(time.time())}.{host}.{os.getpid()}"
        self.path = os.path.join(logdir, fname)
        self._fp: BinaryIO = open(self.path, "wb")

    def add_scalars(self, step: int, metrics: dict[str, float]) -> None:
        items = [(k, float(v)) for k, v in metrics.items() if v is not None]
        if not items:
            return
        _write_record(self._fp, _event(step, time.time(), items))
        self._fp.flush()

    def close(self) -> str:
        self._fp.close()
        return self.path
