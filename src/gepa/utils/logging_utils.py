from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional
from dataclasses import dataclass

import h5py
from torch.utils.tensorboard import SummaryWriter
import numpy as np


@dataclass
class Logger:
    log_dir: str
    csv_file: Optional[str] = None
    jsonl_file: Optional[str] = None
    hdf5_file: Optional[str] = None
    enable_tb: bool = True

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_fh = open(os.path.join(self.log_dir, self.csv_file or "metrics.csv"), "w", newline="")
        self._csv_writer = None
        self._jsonl_fh = open(os.path.join(self.log_dir, self.jsonl_file or "events.jsonl"), "w")
        self._h5 = h5py.File(os.path.join(self.log_dir, self.hdf5_file or "trajectories.h5"), "w")
        self._tb = SummaryWriter(self.log_dir) if self.enable_tb else None

    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=["step", *metrics.keys()], extrasaction="ignore")
            self._csv_writer.writeheader()
        else:
            # Expand CSV columns dynamically if new metric keys appear
            existing = set(self._csv_writer.fieldnames or [])
            incoming = set(["step", *metrics.keys()])
            if not incoming.issubset(existing):
                new_fieldnames = sorted(existing.union(incoming), key=lambda k: (k != "step", k))
                self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=new_fieldnames, extrasaction="ignore")
                self._csv_writer.writeheader()
        row = {"step": step, **metrics}
        self._csv_writer.writerow(row)
        self._csv_fh.flush()
        self._jsonl_fh.write(json.dumps({"step": step, **metrics}) + "\n")
        self._jsonl_fh.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                try:
                    self._tb.add_scalar(k, float(v), global_step=step)
                except Exception:
                    pass

    def log_trajectory(self, name: str, observations: Iterable, actions: Iterable, rewards: Iterable):
        grp = self._h5.create_group(name)
        grp.create_dataset("observations", data=list(observations))
        grp.create_dataset("actions", data=list(actions))
        grp.create_dataset("rewards", data=list(rewards))
        self._h5.flush()

    def close(self):
        self._csv_fh.close()
        self._jsonl_fh.close()
        self._h5.close()
        if self._tb is not None:
            self._tb.close()

    def log_image(self, tag: str, img: np.ndarray, step: int):
        if self._tb is None:
            return
        try:
            # Expect HxWxC in [0,255] uint8 or [0,1] float
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            self._tb.add_image(tag, img.transpose(2, 0, 1), global_step=step)
        except Exception:
            pass
