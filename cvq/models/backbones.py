"""Decoder-only transformer backbones for CAR.

Two interchangeable implementations behind one interface
(`forward(inputs_embeds) -> last hidden states`, causal attention):

- TinyGPT: a from-scratch pre-norm transformer so the MNIST config can
  train on a laptop (CPU/MPS) with zero extra dependencies.
- HFBackbone: wraps any Hugging Face causal trunk (the paper uses
  Qwen3-4B/8B) via AutoModel; requires `pip install cvq[llm]`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGPTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyGPT(nn.Module):
    """Minimal causal transformer with learned positional embeddings."""

    def __init__(self, dim: int = 384, depth: int = 6, heads: int = 6, max_seq_len: int = 512):
        super().__init__()
        self.hidden_size = dim
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([TinyGPTBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        B, T, _ = inputs_embeds.shape
        x = inputs_embeds + self.pos_emb[:, :T]
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm_out(x)


class HFBackbone(nn.Module):
    """Hugging Face trunk (e.g. Qwen/Qwen3-4B) exposed as embeds -> hidden."""

    def __init__(self, model_name: str, dtype: str = "bfloat16", gradient_checkpointing: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=getattr(torch, dtype))
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.hidden_size = self.model.config.hidden_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.model(inputs_embeds=inputs_embeds).last_hidden_state


def build_backbone(cfg: dict) -> nn.Module:
    kind = cfg.get("kind", "tiny")
    if kind == "tiny":
        return TinyGPT(
            dim=cfg.get("dim", 384),
            depth=cfg.get("depth", 6),
            heads=cfg.get("heads", 6),
            max_seq_len=cfg.get("max_seq_len", 512),
        )
    if kind == "hf":
        return HFBackbone(
            cfg["model_name"],
            dtype=cfg.get("dtype", "bfloat16"),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        )
    raise ValueError(f"unknown backbone kind: {kind}")
