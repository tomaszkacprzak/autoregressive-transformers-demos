#!/usr/bin/env python3
"""Toy conditional autoencoder + small decoder adapters.

Foundation pairs:
    X ~ N(0, diag(sigma_1^2, sigma_2^2)) as a 128x128 image
    y = [sigma_1^2, sigma_2^2]

Adapter quads:
    X       uses cov_xy = 0
    X_prime uses cov_xy = b[0]
    y = [sigma_1^2, sigma_2^2], b = [cov_xy]

During adapter training the foundation model is frozen and only small residual
modules inserted in its decoder are optimized. The final edit is

    X_pred = X + D_adapt(E(X, y), y, b) - D_frozen(E(X, y), y).

The adapter uses A(h,y,b)-A(h,y,0), so b=0 gives X_pred=X exactly.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import wandb
import torchinfo


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


class GaussianDataset(Dataset):
    def __init__(
        self,
        n: int,
        image_size: int,
        sigma_min: float,
        sigma_max: float,
        rho_max: float,
        seed: int,
        extended: bool,
    ):
        rng = np.random.default_rng(seed)
        self.s1 = rng.uniform(sigma_min, sigma_max, n).astype(np.float32)
        self.s2 = rng.uniform(sigma_min, sigma_max, n).astype(np.float32)
        self.rho = (
            rng.uniform(-rho_max, rho_max, n).astype(np.float32)
            if extended
            else np.zeros(n, np.float32)
        )
        self.extended = extended

        axis = torch.arange(image_size, dtype=torch.float32) - (image_size - 1) / 2
        self.yy, self.xx = torch.meshgrid(axis, axis, indexing="ij")

    def __len__(self):
        return len(self.s1)

    def image(self, v1: float, v2: float, cov_xy: float):
        det = v1 * v2 - cov_xy**2
        q = (
            v2 * self.xx.square()
            - 2 * cov_xy * self.xx * self.yy
            + v1 * self.yy.square()
        ) / det
        return torch.exp(-0.5 * q).unsqueeze(0)

    def __getitem__(self, i):
        s1, s2 = float(self.s1[i]), float(self.s2[i])
        v1, v2 = s1**2, s2**2
        y = torch.tensor([v1, v2], dtype=torch.float32)
        x = self.image(v1, v2, 0.0)
        if not self.extended:
            return x, y

        cov_xy = float(self.rho[i] * s1 * s2)  # |rho|<1 => positive definite
        b = torch.tensor([cov_xy], dtype=torch.float32)
        x_prime = self.image(v1, v2, cov_xy)
        return x, y, x_prime, b


class GaussianDatasetDirect(GaussianDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, extended=True)

    def __getitem__(self, i):
        x, y, x_prime, b = super().__getitem__(i)
        y = torch.cat([y, b])
        return x_prime, y


# -----------------------------------------------------------------------------
# Foundation model and adapters
# -----------------------------------------------------------------------------


def norm(channels):
    groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


def down(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 4, 2, 1), norm(out_ch), nn.SiLU()
    )


def up(in_ch, out_ch):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), norm(out_ch), nn.SiLU()
    )


class FeatureAdapter(nn.Module):
    """Low-rank, zero-initialized residual adapter for one feature map."""

    def __init__(self, channels, rank):
        super().__init__()
        self.down = nn.Conv2d(channels, rank, 1)
        self.cond = nn.Sequential(nn.Linear(3, rank), nn.SiLU(), nn.Linear(rank, rank))
        self.up = nn.Conv2d(rank, channels, 3, padding=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def raw(self, h, y_norm, b_norm):
        c = self.cond(torch.cat([y_norm, b_norm], dim=1))[:, :, None, None]
        return self.up(F.silu(self.down(h) + c))

    def forward(self, h, y_norm, b_norm):
        return h + self.raw(h, y_norm, b_norm) - self.raw(h, y_norm, torch.zeros_like(b_norm))


class FoundationAE(nn.Module):
    def __init__(
        self,
        image_size=128,
        sigma_min=10.0,
        sigma_max=20.0,
        rho_max=0.8,
        base=16,
        latent=32,
        y_embed=16,
        y_dim=2,
    ):
        super().__init__()
        self.config = dict(
            image_size=image_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho_max=rho_max,
            base=base,
            latent=latent,
            y_embed=y_embed,
        )
        self.encoder = nn.Sequential(
            # down(3, base),
            down(y_dim+1, base),
            down(base, 2 * base),
            down(2 * base, 4 * base),
            down(4 * base, 8 * base),
            nn.Conv2d(8 * base, latent, 3, padding=1),
        )
        self.y_proj = nn.Sequential(
            # nn.Linear(2, y_embed), nn.SiLU(), nn.Linear(y_embed, y_embed)
            nn.Linear(y_dim, y_embed), nn.SiLU(), nn.Linear(y_embed, y_embed)
        )
        self.stem = nn.Sequential(
            nn.Conv2d(latent + y_embed, 8 * base, 3, padding=1),
            norm(8 * base),
            nn.SiLU(),
        )
        self.up_blocks = nn.ModuleList(
            [
                up(8 * base, 4 * base),
                up(4 * base, 2 * base),
                up(2 * base, base),
                up(base, base),
            ]
        )
        self.to_image = nn.Conv2d(base, 1, 3, padding=1)
        self.adapter_channels = [8 * base, 4 * base, 2 * base, base, base]

    def normalize_y(self, y):
        lo, hi = self.config["sigma_min"] ** 2, self.config["sigma_max"] ** 2
        return 2 * (y - lo) / (hi - lo) - 1

    def normalize_b(self, b, y):
        rho = b / torch.sqrt(y[:, :1] * y[:, 1:2]).clamp_min(1e-8)
        return (rho / self.config["rho_max"]).clamp(-1, 1)

    def encode(self, x, y):
        yn = self.normalize_y(y)
        y_maps = yn[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        return self.encoder(torch.cat([x, y_maps], dim=1))

    def decode(self, z, y, adapters=None, b=None):
        yn = self.normalize_y(y)
        y_map = self.y_proj(yn)[:, :, None, None].expand(
            -1, -1, z.shape[-2], z.shape[-1]
        )
        h = self.stem(torch.cat([z, y_map], dim=1))
        bn = self.normalize_b(b, y) if adapters is not None else None

        if adapters is not None:
            h = adapters[0](h, yn, bn)
        for i, block in enumerate(self.up_blocks, start=1):
            h = block(h)
            if adapters is not None:
                h = adapters[i](h, yn, bn)
        return torch.sigmoid(self.to_image(h))

    def forward(self, x, y):
        return self.decode(self.encode(x, y), y)


class Editor(nn.Module):
    def __init__(self, foundation, adapter_rank):
        super().__init__()
        self.foundation = foundation
        self.foundation.requires_grad_(False).eval()
        self.adapters = nn.ModuleList(
            [FeatureAdapter(c, adapter_rank) for c in foundation.adapter_channels]
        )

    def forward(self, x, y, b):
        with torch.no_grad():
            z = self.foundation.encode(x, y)
            decoded_base = self.foundation.decode(z, y)
        # Frozen decoder operations remain in autograd so adapter gradients flow.
        decoded_edit = self.foundation.decode(z, y, self.adapters, b)
        delta = decoded_edit - decoded_base
        return x + delta, delta


# -----------------------------------------------------------------------------
# Logging figures
# -----------------------------------------------------------------------------


def fixed_pairs(dataset):
    batch = [dataset[i] for i in range(8)]
    return torch.stack([v[0] for v in batch]), torch.stack([v[1] for v in batch])


def fixed_quads(dataset):
    batch = [dataset[i] for i in range(8)]
    return tuple(torch.stack([v[j] for v in batch]) for j in range(4))


def finish_axes(axes, labels):
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])


@torch.no_grad()
def foundation_plot(model, examples, device):
    x, y = examples
    pred = model(x.to(device), y.to(device)).cpu().clamp(0, 1)
    diff = x - pred
    lim = 0.1
    fig, axes = plt.subplots(3, 8, figsize=(16, 6), squeeze=False)
    for i in range(8):
        axes[0, i].imshow(x[i, 0], cmap="turbo", vmin=0, vmax=1)
        axes[1, i].imshow(pred[i, 0], cmap="turbo", vmin=0, vmax=1)
        axes[2, i].imshow(diff[i, 0], cmap="coolwarm", vmin=-lim, vmax=lim)
        axes[0, i].set_title(f"v1={y[i,0]:.0f}\nv2={y[i,1]:.0f}", fontsize=8)
    finish_axes(axes, ["Input X", "Decoded", "Difference"])
    fig.tight_layout()
    return fig


@torch.no_grad()
def adapter_plot(editor, examples, device):
    x, y, xp, b = examples
    pred, _ = editor(x.to(device), y.to(device), b.to(device))
    pred = pred.cpu().clamp(0, 1)
    diff = xp - pred
    lim = 0.1
    fig, axes = plt.subplots(4, 8, figsize=(16, 8), squeeze=False)
    for i in range(8):
        axes[0, i].imshow(xp[i, 0], cmap="turbo", vmin=0, vmax=1)
        axes[1, i].imshow(x[i, 0], cmap="turbo", vmin=0, vmax=1)
        axes[2, i].imshow(pred[i, 0], cmap="turbo", vmin=0, vmax=1)
        axes[3, i].imshow(diff[i, 0], cmap="coolwarm", vmin=-lim, vmax=lim)
        rho = b[i, 0] / torch.sqrt(y[i, 0] * y[i, 1])
        axes[0, i].set_title(f"b={b[i,0]:.1f}\nrho={rho:.2f}", fontsize=8)
    finish_axes(axes, ["Target X'", "Prompt X", "Predicted Xp", "X' - Xp"])
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@torch.no_grad()
def validation_mse(model, loader, device, adapter=False):
    total, pixels = 0.0, 0
    for batch in loader:
        batch = [v.to(device) for v in batch]
        if adapter:
            x, y, xp, b = batch
            pred, _ = model(x, y, b)
            target = xp
        else:
            x, y = batch
            pred, target = model(x, y), x
        total += F.mse_loss(pred, target, reduction="sum").item()
        pixels += target.numel()
    return total / pixels


def train_foundation(model, train_loader, val_loader, examples, args, device, run, ckpt, num_epochs, tag='foundation'):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.foundation_lr, weight_decay=args.weight_decay
    )
    best, step = math.inf, 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x, y), x)
            loss.backward()
            opt.step()
            run.log({f"{tag}/train_loss_batch": loss.item(), f"{tag}/step": step})
            step += 1

        val = validation_mse(model.eval(), val_loader, device)
        fig = foundation_plot(model, examples, device)
        run.log(
            {
                f"{tag}/validation_loss": val,
                f"{tag}/epoch": epoch,
                f"{tag}/reconstructions": wandb.Image(fig),
            }
        )
        plt.close(fig)
        print(f"[{tag}] epoch {epoch:03d}: val_mse={val:.6e}", flush=True)
        if val < best:
            best = val
            torch.save({"state": model.state_dict(), "config": model.config}, ckpt)


def train_adapter(editor, train_loader, val_loader, examples, args, device, run, ckpt):
    opt = torch.optim.AdamW(
        editor.adapters.parameters(), lr=args.adapter_lr, weight_decay=args.weight_decay
    )
    best, step = math.inf, 0
    for epoch in range(1, args.adapter_epochs + 1):
        editor.foundation.eval()
        editor.adapters.train()
        for x, y, xp, b in train_loader:
            x, y, xp, b = [v.to(device) for v in (x, y, xp, b)]
            opt.zero_grad(set_to_none=True)
            pred, delta = editor(x, y, b)
            reconstruction = F.mse_loss(pred, xp)
            loss = reconstruction + args.adapter_reg * delta.square().mean()
            loss.backward()
            opt.step()
            run.log(
                {
                    "adapter/train_loss_batch": loss.item(),
                    "adapter/reconstruction_loss_batch": reconstruction.item(),
                    "adapter/step": step,
                }
            )
            step += 1

        val = validation_mse(editor, val_loader, device, adapter=True)
        fig = adapter_plot(editor, examples, device)
        run.log(
            {
                "adapter/validation_loss": val,
                "adapter/epoch": epoch,
                "adapter/validation_examples": wandb.Image(fig),
            }
        )
        plt.close(fig)
        print(f"[adapter] epoch {epoch:03d}: val_mse={val:.6e}", flush=True)
        if val < best:
            best = val
            torch.save(
                {"state": editor.adapters.state_dict(), "rank": args.adapter_rank}, ckpt
            )

def unadapted_plot(model, loader, device):

    for x, y, xp, b in loader:
        pred = model(x.to(device), y.to(device)).cpu().clamp(0, 1)
        diff = x - pred
        lim = 0.1
        fig, axes = plt.subplots(3, 8, figsize=(16, 6), squeeze=False)
        for i in range(8):
            axes[0, i].imshow(x[i, 0], cmap="turbo", vmin=0, vmax=1)
            axes[1, i].imshow(pred[i, 0], cmap="turbo", vmin=0, vmax=1)
            axes[2, i].imshow(diff[i, 0], cmap="coolwarm", vmin=-lim, vmax=lim)
            axes[0, i].set_title(f"v1={y[i,0]:.0f} v2={y[i,1]:.0f} b={b[i,0]:.1f}", fontsize=8)
        finish_axes(axes, ["Input X (extended)", "Decoded", "Difference"])
        fig.tight_layout()
        return fig

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def loader(dataset, batch_size, shuffle, args, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["all", "foundation", "adapter", "direct"], default="all")
    p.add_argument("--output-dir", type=Path, default=Path("gaussian_demo_runs"))
    p.add_argument("--foundation-checkpoint", type=Path)
    p.add_argument("--adapter-checkpoint", type=Path)
    p.add_argument("--direct-checkpoint", type=Path)

    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--sigma-min", type=float, default=10.0)
    p.add_argument("--sigma-max", type=float, default=20.0)
    p.add_argument("--rho-max", type=float, default=0.8)
    p.add_argument("--foundation-train-size", type=int, default=10_000)
    p.add_argument("--foundation-val-size", type=int, default=1_000)
    p.add_argument("--adapter-train-size", type=int, default=100)
    p.add_argument("--adapter-val-size", type=int, default=1_000)

    p.add_argument("--foundation-epochs", type=int, default=100)
    p.add_argument("--adapter-epochs", type=int, default=1000)
    p.add_argument("--direct-epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--adapter-batch-size", type=int, default=64)
    p.add_argument("--foundation-lr", type=float, default=1e-3)
    p.add_argument("--adapter-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--adapter-reg", type=float, default=1e-5)

    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--latent-channels", type=int, default=32)
    p.add_argument("--y-embed-channels", type=int, default=16)
    p.add_argument("--adapter-rank", type=int, default=8)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--wandb-project", default="gaussian-foundation-adapter-demo")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-run-name")
    p.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    return p.parse_args()


def make_dataset(args, n, seed, extended, model_config=None, direct=False):
    c = model_config or dict(
        image_size=args.image_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho_max=args.rho_max,
    )

    if not direct:
        dataset = GaussianDataset(
            n=n,
            image_size=c["image_size"],
            sigma_min=c["sigma_min"],
            sigma_max=c["sigma_max"],
            rho_max=c["rho_max"],
            seed=seed,
            extended=extended,
        )
    else:
        dataset = GaussianDatasetDirect(
            n=n,
            image_size=c["image_size"],
            sigma_min=c["sigma_min"],
            sigma_max=c["sigma_max"],
            rho_max=c["rho_max"],
            seed=seed,
        )
    return dataset


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    if device.type == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    foundation_ckpt = args.foundation_checkpoint or args.output_dir / "foundation.pt"
    adapter_ckpt = args.adapter_checkpoint or args.output_dir / "adapter.pt"
    direct_ckpt = args.direct_checkpoint or args.output_dir / "direct.pt"
    print(f"device={device}; foundation={foundation_ckpt}; adapter={adapter_ckpt}", flush=True)

    batch_dims_x = (args.batch_size, 1, args.image_size, args.image_size)
    batch_dims_y = (args.batch_size, 2)
    batch_dims_b = (args.batch_size, 1)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    ) as run:
        if args.mode in ("all", "foundation"):
            train_ds = make_dataset(args, args.foundation_train_size, args.seed + 1, False)
            val_ds = make_dataset(args, args.foundation_val_size, args.seed + 2, False)
            if len(val_ds) < 8:
                raise ValueError("foundation-val-size must be at least 8")
            model = FoundationAE(
                image_size=args.image_size,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho_max=args.rho_max,
                base=args.base_channels,
                latent=args.latent_channels,
                y_embed=args.y_embed_channels,
            ).to(device)
            print(torchinfo.summary(model, input_size=(batch_dims_x, batch_dims_y), device=device))

            run.log({"model/foundation_parameters": sum(p.numel() for p in model.parameters())})
            train_foundation(
                model,
                loader(train_ds, args.batch_size, True, args, device),
                loader(val_ds, args.batch_size, False, args, device),
                fixed_pairs(val_ds),
                args,
                device,
                run,
                foundation_ckpt,
                args.foundation_epochs,
            )

        if args.mode in ("all", "adapter"):
            checkpoint = torch.load(foundation_ckpt, map_location="cpu", weights_only=True)
            foundation = FoundationAE(**checkpoint["config"]).to(device)
            foundation.load_state_dict(checkpoint["state"])
            editor = Editor(foundation, args.adapter_rank).to(device)

            print(torchinfo.summary(editor, input_size=(batch_dims_x, batch_dims_y, batch_dims_b), device=device))

            train_ds = make_dataset(
                args, args.adapter_train_size, args.seed + 3, True, foundation.config
            )
            val_ds = make_dataset(
                args, args.adapter_val_size, args.seed + 4, True, foundation.config
            )
            if len(val_ds) < 8:
                raise ValueError("adapter-val-size must be at least 8")

            fig = unadapted_plot(foundation, 
                                loader(train_ds, args.batch_size, True, args, device), 
                                device)
            run.log({"model/unadapted_encoder": wandb.Image(fig)})

            n_adapter = sum(p.numel() for p in editor.adapters.parameters())
            n_foundation = sum(p.numel() for p in foundation.parameters())
            run.log(
                {
                    "model/adapter_trainable_parameters": n_adapter,
                    "model/foundation_parameters_frozen": n_foundation,
                    "model/adapter_fraction": n_adapter / n_foundation,
                }
            )
            train_adapter(
                editor,
                loader(train_ds, args.adapter_batch_size, True, args, device),
                loader(val_ds, args.adapter_batch_size, False, args, device),
                fixed_quads(train_ds),
                args,
                device,
                run,
                adapter_ckpt,
            )

        if args.mode in ("all", "direct"):
            train_ds = make_dataset(args, args.adapter_train_size, args.seed + 1, False, direct=True)
            val_ds = make_dataset(args, args.adapter_train_size, args.seed + 2, False, direct=True)
            if len(val_ds) < 8:
                raise ValueError("foundation-val-size must be at least 8")
            model = FoundationAE(
                image_size=args.image_size,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho_max=args.rho_max,
                base=args.base_channels,
                latent=args.latent_channels,
                y_embed=args.y_embed_channels,
                y_dim=3,
            ).to(device)
            run.log({"model/direct_parameters": sum(p.numel() for p in model.parameters())})
            train_foundation(
                model,
                loader(train_ds, args.batch_size, True, args, device),
                loader(val_ds, args.batch_size, False, args, device),
                fixed_pairs(val_ds),
                args,
                device,
                run,
                direct_ckpt,
                args.direct_epochs,
                tag='direct',
            )

            


if __name__ == "__main__":
    main()
