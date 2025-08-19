from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional
from dataclasses import dataclass

import h5py
from torch.utils.tensorboard import SummaryWriter


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
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=["step", *metrics.keys()])
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
