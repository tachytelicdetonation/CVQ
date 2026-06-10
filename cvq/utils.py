"""Shared helpers: config loading, device pick, checkpointing, image grids."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_image_grid(tensor: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """tensor (B, C, H, W) in [-1, 1] -> PNG grid."""
    from torchvision.utils import save_image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor.clamp(-1, 1) * 0.5 + 0.5, str(path), nrow=nrow)


def build_tokenizer_from_cfg(model_cfg: dict, data_cfg: dict):
    from .models.tokenizer import CVQTokenizer

    return CVQTokenizer(
        resolution=data_cfg["resolution"],
        in_channels=data_cfg.get("in_channels", 3),
        ch=model_cfg["ch"],
        ch_mult=tuple(model_cfg["ch_mult"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        z_channels=model_cfg["z_channels"],
        num_codes=model_cfg["num_codes"],
        beta=model_cfg.get("beta", 0.25),
        use_mid_attn=model_cfg.get("use_mid_attn", True),
        dropout_ratio=model_cfg.get("nested_dropout_ratio", 0.25),
    )


def load_tokenizer_checkpoint(ckpt_path: str, device: torch.device):
    """Returns (tokenizer, full checkpoint dict)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tok = build_tokenizer_from_cfg(ckpt["config"]["model"], ckpt["config"]["data"])
    tok.load_state_dict(ckpt["model"])
    return tok.to(device).eval(), ckpt
