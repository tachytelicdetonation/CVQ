"""Text conditioning for CAR.

The paper conditions on free-form text via the Qwen3 tokenizer. To keep
the *same pipeline* at MNIST scale, class labels are rendered as prompts
("a handwritten digit 7") and tokenized with a tiny word-level vocab, so
both tiers go through the identical text-tokens -> embeddings -> prefix
code path.
"""

from __future__ import annotations

import torch

MNIST_PROMPT = "a handwritten digit {label}"


class WordTokenizer:
    """Word-level tokenizer over a fixed corpus of prompt templates."""

    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, vocab: list[str]):
        self.itos = [self.PAD, self.UNK] + sorted(set(vocab))
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = 0

    @classmethod
    def for_mnist(cls) -> "WordTokenizer":
        words = set()
        for d in range(10):
            words.update(MNIST_PROMPT.format(label=d).split())
        return cls(sorted(words))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(w, 1) for w in text.split()][:max_len]
        return ids + [self.pad_id] * (max_len - len(ids))


class HFTextTokenizer:
    """Wraps a Hugging Face tokenizer (paper: Qwen3's)."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.pad_id = self.tok.pad_token_id or self.tok.eos_token_id

    def __len__(self) -> int:
        return len(self.tok)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = self.tok(text, truncation=True, max_length=max_len)["input_ids"]
        return ids + [self.pad_id] * (max_len - len(ids))


def build_text_tokenizer(cfg: dict):
    kind = cfg.get("kind", "word")
    if kind == "word":
        return WordTokenizer.for_mnist()
    if kind == "hf":
        return HFTextTokenizer(cfg["model_name"])
    raise ValueError(f"unknown text tokenizer kind: {kind}")


def encode_batch(tokenizer, texts: list[str], max_len: int, device) -> torch.Tensor:
    return torch.tensor([tokenizer.encode(t, max_len) for t in texts], dtype=torch.long, device=device)
