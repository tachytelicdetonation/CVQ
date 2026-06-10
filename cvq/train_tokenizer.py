"""Train the CVQ tokenizer.

    python -m cvq.train_tokenizer --config configs/tokenizer_mnist.yaml

Paper recipe (Sec. 4.1): Adam(b1=0.5, b2=0.9), lr 1e-4, wd 1e-4,
batch 256, 100 epochs on ImageNet-1K at 256x256. Nested channel dropout
is sampled per iteration inside the loop (Appendix B.1, Eq. 9).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .data import build_loader
from .losses import VQGANLoss
from .utils import build_tokenizer_from_cfg, load_config, pick_device, save_image_grid, set_seed


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> dict[str, float]:
    model.eval()
    mse_sum, n = 0.0, 0
    used = torch.zeros(model.quantizer.num_codes, dtype=torch.bool, device=device)
    for b, (img, _) in enumerate(loader):
        if b >= max_batches:
            break
        img = img.to(device)
        recon, _, idx = model(img)
        mse_sum += torch.mean((recon.clamp(-1, 1) - img) ** 2).item() * img.shape[0]
        used[idx.unique()] = True
        n += img.shape[0]
    mse = mse_sum / max(n, 1)
    psnr = 10.0 * torch.log10(torch.tensor(4.0 / max(mse, 1e-10))).item()  # range [-1,1] -> peak^2 = 4
    model.train()
    return {"val/psnr": psnr, "val/codebook_usage": used.float().mean().item()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="override config (smoke tests)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(cfg.get("seed", 0))
    device = pick_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    model = build_tokenizer_from_cfg(cfg["model"], cfg["data"]).to(device)
    loss_fn = VQGANLoss(
        in_channels=cfg["data"].get("in_channels", 3),
        perceptual_weight=tcfg.get("perceptual_weight", 0.0),
        gan_weight=tcfg.get("gan_weight", 0.0),
        gan_eta=cfg["model"].get("gan_eta", 0.05),
        disc_start=tcfg.get("disc_start", 10000),
    ).to(device)

    opt_g = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], betas=(0.5, 0.9), weight_decay=tcfg.get("weight_decay", 1e-4)
    )
    opt_d = (
        torch.optim.Adam(loss_fn.discriminator.parameters(), lr=tcfg["lr"], betas=(0.5, 0.9))
        if loss_fn.discriminator is not None
        else None
    )

    train_loader = build_loader({**cfg["data"], "batch_size": tcfg["batch_size"]}, train=True)
    val_loader = build_loader({**cfg["data"], "batch_size": tcfg["batch_size"]}, train=False)

    max_steps = args.max_steps or tcfg.get("max_steps") or tcfg["epochs"] * len(train_loader)
    step, t0 = 0, time.time()
    model.train()
    pbar = tqdm(total=max_steps, desc="tokenizer")
    while step < max_steps:
        for img, _ in train_loader:
            if step >= max_steps:
                break
            img = img.to(device)
            c_keep = model.sample_c_keep()
            recon, vq_loss, _ = model(img, c_keep=c_keep)

            loss, logs = loss_fn.generator_loss(
                recon, img, vq_loss, c_keep, model.num_channels, global_step=step
            )
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()

            d_loss = loss_fn.discriminator_loss(recon, img, global_step=step)
            if d_loss is not None and opt_d is not None:
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()
                logs["loss/d"] = d_loss.item()

            step += 1
            pbar.update(1)
            if step % tcfg.get("log_every", 100) == 0:
                pbar.set_postfix({k.split("/")[-1]: f"{v:.3f}" for k, v in logs.items()})
            if step % tcfg.get("eval_every", 2000) == 0 or step == max_steps:
                metrics = evaluate(model, val_loader, device)
                print(f"\nstep {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                save_image_grid(torch.cat([img[:8], recon[:8].detach()]), out_dir / f"recon_{step:07d}.png")
            if step % tcfg.get("ckpt_every", 5000) == 0 or step == max_steps:
                torch.save(
                    {"model": model.state_dict(), "config": cfg, "step": step},
                    out_dir / "tokenizer.pt",
                )
    pbar.close()
    print(f"done in {time.time() - t0:.0f}s -> {out_dir / 'tokenizer.pt'}")


if __name__ == "__main__":
    main()
