from __future__ import annotations

import os
from typing import Any, Dict
import torch


def save_checkpoint(state: Dict[str, Any], checkpoint_dir: str, name: str = "ckpt.pt") -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, name)
    torch.save(state, path)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")
