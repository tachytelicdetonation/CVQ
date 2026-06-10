"""Datasets for both tiers.

Every dataset yields (image, prompt) pairs so the tokenizer and CAR
training loops are dataset-agnostic. Images are in [-1, 1].

- mnist: torchvision MNIST, padded 28->32, prompts "a handwritten digit N".
- imagefolder: ImageNet-style class folders (tokenizer training; prompts
  are the folder names).
- jsonl: text-to-image pairs, one {"image": path, "text": caption} per
  line — the stand-in for the paper's 80M-pair mixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .text import MNIST_PROMPT


class MNISTPrompts(Dataset):
    def __init__(self, root: str, train: bool = True, resolution: int = 32):
        tf = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.ds = datasets.MNIST(root, train=train, download=True, transform=tf)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        img, label = self.ds[i]
        return img, MNIST_PROMPT.format(label=label)


class ImageFolderPrompts(Dataset):
    def __init__(self, root: str, resolution: int = 256, train: bool = True):
        ops = [transforms.Resize(resolution), transforms.CenterCrop(resolution)]
        if train:
            ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        self.ds = datasets.ImageFolder(root, transform=transforms.Compose(ops))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        img, label = self.ds[i]
        return img, self.ds.classes[label].replace("_", " ")


class JsonlTextImage(Dataset):
    def __init__(self, jsonl_path: str, resolution: int = 256):
        from PIL import Image

        self._open = Image.open
        self.records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        self.root = Path(jsonl_path).parent
        self.tf = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        img = self._open(self.root / rec["image"]).convert("RGB")
        return self.tf(img), rec["text"]


def build_dataset(cfg: dict, train: bool = True) -> Dataset:
    kind = cfg["kind"]
    if kind == "mnist":
        return MNISTPrompts(cfg.get("root", "data"), train=train, resolution=cfg["resolution"])
    if kind == "imagefolder":
        root = cfg["train_root" if train else "val_root"]
        return ImageFolderPrompts(root, resolution=cfg["resolution"], train=train)
    if kind == "jsonl":
        return JsonlTextImage(cfg["train_jsonl" if train else "val_jsonl"], resolution=cfg["resolution"])
    raise ValueError(f"unknown dataset kind: {kind}")


def build_loader(cfg: dict, train: bool = True) -> DataLoader:
    ds = build_dataset(cfg, train=train)
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=train,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=cfg.get("num_workers", 2) > 0,
    )
