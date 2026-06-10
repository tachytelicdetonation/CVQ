"""Train the CAR next-channel-prediction model on a frozen CVQ tokenizer.

    python -m cvq.train_car --config configs/car_mnist.yaml

Paper recipe (Sec. 4.2 + Appendix E): two stages, AdamW(b1=0.9, b2=0.96,
wd=1e-3). Stage I trains only the MLP projector + LM head (lr 1e-4) with
the LLM frozen; Stage II trains everything (lr 2e-5). Set
train.stage: 0 to train end-to-end in one pass (MNIST tier).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .data import build_loader
from .models.car import CARModel
from .text import build_text_tokenizer, encode_batch
from .utils import load_config, load_tokenizer_checkpoint, pick_device, set_seed


def build_car(cfg: dict, tokenizer, text_tokenizer) -> CARModel:
    return CARModel(
        backbone_cfg=cfg["backbone"],
        num_codes=tokenizer.quantizer.num_codes,
        code_dim=tokenizer.code_dim,
        num_channels=tokenizer.num_channels,
        text_vocab_size=len(text_tokenizer),
        cond_drop_prob=cfg.get("cond_drop_prob", 0.1),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", default=None, help="CAR checkpoint to continue from (e.g. stage II)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(cfg.get("seed", 0))
    device = pick_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    tokenizer, _ = load_tokenizer_checkpoint(cfg["tokenizer_ckpt"], device)
    tokenizer.requires_grad_(False)
    text_tokenizer = build_text_tokenizer(cfg["text_tokenizer"])

    model = build_car(cfg, tokenizer, text_tokenizer).to(device)
    model.load_codebook(tokenizer.quantizer.embedding.weight)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=False)["model"])

    stage = tcfg.get("stage", 0)
    model.set_stage(stage)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], betas=(0.9, 0.96), weight_decay=tcfg.get("weight_decay", 1e-3))
    n_train = sum(p.numel() for p in params)
    print(f"stage {stage}: training {n_train/1e6:.1f}M params")

    loader = build_loader({**cfg["data"], "batch_size": tcfg["batch_size"]}, train=True)
    max_text_len = cfg.get("max_text_len", 32)
    max_steps = args.max_steps or tcfg.get("max_steps") or tcfg["epochs"] * len(loader)

    step, t0 = 0, time.time()
    model.train()
    pbar = tqdm(total=max_steps, desc=f"car-stage{stage}")
    while step < max_steps:
        for img, prompts in loader:
            if step >= max_steps:
                break
            img = img.to(device)
            with torch.no_grad():
                indices = tokenizer.encode_indices(img)
            text_ids = encode_batch(text_tokenizer, list(prompts), max_text_len, device)

            loss = model(text_ids, indices)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            step += 1
            pbar.update(1)
            if step % tcfg.get("log_every", 50) == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
            if step % tcfg.get("ckpt_every", 5000) == 0 or step == max_steps:
                torch.save({"model": model.state_dict(), "config": cfg, "step": step}, out_dir / "car.pt")
    pbar.close()
    print(f"done in {time.time() - t0:.0f}s -> {out_dir / 'car.pt'}")


if __name__ == "__main__":
    main()
