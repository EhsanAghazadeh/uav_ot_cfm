"""Minimal copy of the user's canonical-format helper.
Only what the TOES adapter / evaluator need to read+write trajectory.npz.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class TrajectoryRecord:
    pos: np.ndarray
    times: np.ndarray
    domain: Tuple[float, float, float, float]
    method: str
    meta: Dict[str, float]

    @classmethod
    def from_npz(cls, path: str) -> "TrajectoryRecord":
        data = np.load(path, allow_pickle=True)
        pos = np.asarray(data["pos"], dtype=np.float64)
        times = np.asarray(data["times"], dtype=np.float64)
        domain = tuple(float(x) for x in np.asarray(data["domain"]).ravel())
        if len(domain) != 4:
            raise ValueError(f"Expected domain of length 4, got {domain}")
        method = str(data["method"])
        meta: Dict[str, float] = {}
        if "meta" in data.files:
            try:
                raw = data["meta"]
                if raw.dtype.kind in ("U", "S"):
                    meta = json.loads(str(raw))
                elif raw.dtype == object:
                    item = raw.item()
                    meta = item if isinstance(item, dict) else {}
            except Exception as err:
                print(f"[warn] could not read meta from {path}: {err!r}")
                meta = {}
        return cls(pos=pos, times=times, domain=domain, method=method, meta=meta)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            pos=self.pos,
            times=self.times,
            domain=np.asarray(self.domain, dtype=np.float64),
            method=str(self.method),
            meta=json.dumps(self.meta, default=float),
        )


def save_trajectory_npz(path, pos, times, domain, method, meta=None):
    rec = TrajectoryRecord(
        pos=np.asarray(pos, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        domain=tuple(float(x) for x in domain),
        method=str(method),
        meta=meta or {},
    )
    rec.save(path)
