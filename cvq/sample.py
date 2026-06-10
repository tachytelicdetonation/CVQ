"""Generate images with a trained CAR model.

    python -m cvq.sample --config configs/car_mnist.yaml \
        --prompts "a handwritten digit 3" "a handwritten digit 7" \
        --progressive

--progressive also renders each sample decoded from its first
1, 2, 4, ..., c channels (the coarse-to-fine sweep of Fig. 5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models.car import CARModel
from .text import build_text_tokenizer, encode_batch
from .train_car import build_car
from .utils import load_config, load_tokenizer_checkpoint, pick_device, save_image_grid, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--car-ckpt", default=None, help="defaults to <out_dir>/car.pt")
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--out", default=None, help="defaults to <out_dir>/samples.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = pick_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["out_dir"])

    tokenizer, _ = load_tokenizer_checkpoint(cfg["tokenizer_ckpt"], device)
    text_tokenizer = build_text_tokenizer(cfg["text_tokenizer"])
    model = build_car(cfg, tokenizer, text_tokenizer).to(device).eval()
    ckpt_path = args.car_ckpt or out_dir / "car.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])

    prompts = [p for p in args.prompts for _ in range(args.samples_per_prompt)]
    text_ids = encode_batch(text_tokenizer, prompts, cfg.get("max_text_len", 32), device)
    indices = model.generate(
        text_ids, temperature=args.temperature, top_k=args.top_k, cfg_scale=args.cfg_scale
    )

    images = tokenizer.decode_indices(indices)
    out = Path(args.out) if args.out else out_dir / "samples.png"
    save_image_grid(images, out, nrow=args.samples_per_prompt)
    print(f"wrote {out}")

    if args.progressive:
        c = tokenizer.num_channels
        keeps = [max(1, c // 2**i) for i in range(c.bit_length() - 1, -1, -1)]
        frames = [tokenizer.decode_indices(indices[:1], c_keep=k) for k in keeps]
        prog_out = out.with_name(out.stem + "_progressive.png")
        save_image_grid(torch.cat(frames), prog_out, nrow=len(keeps))
        print(f"wrote {prog_out} (channels: {keeps})")


if __name__ == "__main__":
    main()
