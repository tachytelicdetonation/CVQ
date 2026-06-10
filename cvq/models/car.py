"""CAR: Channel-wise AutoRegressive generation (Sec. 3.2).

Sequence layout per sample:

    [text tokens] [BOI] [x_1] [x_2] ... [x_c]

where x_k are channel-token indices from the CVQ tokenizer. The model is
trained with next-token cross-entropy on the channel positions only
(Eq. 6: p(X) = prod_k p(x_k | x_<k, text)).

Channel tokens enter the backbone as MLP(codeword) where the codeword is
the (frozen) h*w-dim CVQ codebook entry; the head classifies over the
codebook. Classifier-free guidance is supported by dropping the text
prefix during training and mixing cond/uncond logits at sampling time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HFBackbone, TinyGPT, build_backbone


class CARModel(nn.Module):
    def __init__(
        self,
        backbone_cfg: dict,
        num_codes: int,
        code_dim: int,
        num_channels: int,
        text_vocab_size: int | None = None,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone_cfg)
        d = self.backbone.hidden_size
        self.num_codes = num_codes
        self.num_channels = num_channels
        self.cond_drop_prob = cond_drop_prob

        # Frozen copy of the tokenizer codebook; channel tokens embed as
        # projector(codeword) (two-layer MLP, Sec. 3.2 / Stage I).
        self.register_buffer("codebook", torch.zeros(num_codes, code_dim))
        self.projector = nn.Sequential(nn.Linear(code_dim, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Linear(d, num_codes, bias=False)
        self.boi = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # TinyGPT has no token embeddings of its own; HF backbones reuse theirs.
        self.text_embed = None
        if not isinstance(self.backbone, HFBackbone):
            assert text_vocab_size is not None, "tiny backbone needs text_vocab_size"
            self.text_embed = nn.Embedding(text_vocab_size, d)

    @torch.no_grad()
    def load_codebook(self, weight: torch.Tensor) -> None:
        self.codebook.copy_(weight.to(self.codebook.dtype))

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        if self.text_embed is not None:
            return self.text_embed(text_ids)
        return self.backbone.embed_tokens(text_ids)

    def embed_channels(self, indices: torch.Tensor) -> torch.Tensor:
        return self.projector(self.codebook[indices])

    def set_stage(self, stage: int) -> None:
        """Stage I: freeze the LLM backbone, train projector + head (+BOI).
        Stage II: end-to-end (Sec. 4.2 Training Recipe)."""
        self.backbone.requires_grad_(stage != 1)

    def forward(self, text_ids: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Training loss: next-channel cross-entropy.

        text_ids (B, T_text), indices (B, C) -> scalar loss.
        """
        B, C = indices.shape
        text_emb = self.embed_text(text_ids)
        if self.training and self.cond_drop_prob > 0:
            # CFG: zero the text prefix for a random subset of samples.
            keep = (torch.rand(B, 1, 1, device=indices.device) > self.cond_drop_prob).to(text_emb.dtype)
            text_emb = text_emb * keep
        chan_emb = self.embed_channels(indices).to(text_emb.dtype)
        seq = torch.cat([text_emb, self.boi.expand(B, -1, -1).to(text_emb.dtype), chan_emb], dim=1)
        hidden = self.backbone(seq)
        # Positions [T_text + k] predict channel token k+1 (k=0 is BOI).
        T_text = text_ids.shape[1]
        logits = self.head(hidden[:, T_text : T_text + C].float())
        return F.cross_entropy(logits.reshape(B * C, -1), indices.reshape(B * C))

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        cfg_scale: float = 1.0,
        num_channels: int | None = None,
    ) -> torch.Tensor:
        """Sample channel-token sequences (B, C) autoregressively."""
        self.eval()
        B = text_ids.shape[0]
        C = num_channels or self.num_channels
        device = text_ids.device

        text_emb = self.embed_text(text_ids)
        prefix_cond = torch.cat([text_emb, self.boi.expand(B, -1, -1).to(text_emb.dtype)], dim=1)
        prefix_uncond = torch.cat([torch.zeros_like(text_emb), self.boi.expand(B, -1, -1).to(text_emb.dtype)], dim=1)

        indices = torch.zeros(B, 0, dtype=torch.long, device=device)
        for _ in range(C):
            chan_emb = self.embed_channels(indices).to(prefix_cond.dtype) if indices.shape[1] else None

            def step(prefix: torch.Tensor) -> torch.Tensor:
                seq = torch.cat([prefix, chan_emb], dim=1) if chan_emb is not None else prefix
                return self.head(self.backbone(seq)[:, -1].float())

            logits = step(prefix_cond)
            if cfg_scale != 1.0:
                uncond = step(prefix_uncond)
                logits = uncond + cfg_scale * (logits - uncond)
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                kth = logits.topk(top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = logits.softmax(dim=-1)
            nxt = torch.multinomial(probs, 1)
            indices = torch.cat([indices, nxt], dim=1)
        return indices
