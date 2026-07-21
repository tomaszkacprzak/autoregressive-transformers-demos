#!/usr/bin/env python3
"""Frozen-ViT-tokenizer demo for arbitrary-resolution super-resolution.


The tokenizer is continuous: it has no vector quantizer or codebook. Its
position embedding is generated from normalized coordinates and cell sizes,
so it has no fixed absolute-position table.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchinfo

try:
    import wandb
except ImportError as exc:
    raise SystemExit("Install dependencies with: pip install torch matplotlib wandb") from exc


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

class GaussianDataset(Dataset):
    """128x128 centered Gaussians with covariance diag(sigma_x^2, sigma_y^2)."""

    def __init__(self, count=256, size=128, sigma_min=10.0, sigma_max=20.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        sx = sigma_min + (sigma_max - sigma_min) * torch.rand(count, 1, 1, generator=g)
        sy = sigma_min + (sigma_max - sigma_min) * torch.rand(count, 1, 1, generator=g)

        p = torch.arange(size, dtype=torch.float32)
        y, x = torch.meshgrid(p, p, indexing="ij")                    # [H, W], [H, W]
        c = (size - 1) / 2.0
        image = torch.exp(-0.5 * (
            (x[None] - c).square() / sx.square()
            + (y[None] - c).square() / sy.square()
        ))                                                             # [N, H, W]
        self.images = image[:, None]                                    # [N, 1, H, W]
        # sx and sy are intentionally not stored.

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index]                                       # [1, H, W]


def resize(image, size):
    """Bicubic resize: [B,C,H,W] -> [B,C,H_out,W_out]."""
    if image.shape[-2:] == tuple(size):
        return image
    return F.interpolate(
        image, size=size, mode="bicubic", align_corners=False, antialias=True
    ).clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Resolution-flexible ViT autoencoder
# -----------------------------------------------------------------------------

class CoordinatePositionEncoding(nn.Module):
    """Map [x,y,cell_x,cell_y] to one embedding per runtime patch location."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, h, w, device, dtype):
        x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * 2.0 / w - 1.0  # [w]
        y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * 2.0 / h - 1.0  # [h]
        yy, xx = torch.meshgrid(y, x, indexing="ij")                            # [h,w]
        cx, cy = torch.full_like(xx, 2.0 / w), torch.full_like(yy, 2.0 / h)
        q = torch.stack([xx, yy, cx, cy], dim=-1).reshape(1, h * w, 4)           # [1,N,4]
        return self.mlp(q)                                                       # [1,N,E]


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B,N,E]
        u = self.norm1(x)                                                   # [B,N,E]
        x = x + self.attn(u, u, u, need_weights=False)[0]                   # [B,N,E]
        return x + self.mlp(self.norm2(x))                                  # [B,N,E]


class ViTAutoencoder(nn.Module):
    """Patchwise ViT autoencoder with continuous latent tokens [B,N,Dz]."""

    def __init__(
        self, channels=1, patch_size=8, embed_dim=64, latent_dim=32,
        encoder_depth=2, decoder_depth=2, heads=4, mlp_ratio=2.0, dropout=0.0,
    ):
        super().__init__()
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        self.channels, self.patch_size = channels, patch_size
        self.embed_dim, self.latent_dim = embed_dim, latent_dim
        self.encoder_depth, self.decoder_depth = encoder_depth, decoder_depth
        self.heads, self.mlp_ratio, self.dropout = heads, mlp_ratio, dropout

        # [B,C,H,W] -> [B,E,H/P,W/P]
        self.patch_embed = nn.Conv2d(channels, embed_dim, patch_size, stride=patch_size)
        self.position = CoordinatePositionEncoding(embed_dim)
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.to_latent = nn.Linear(embed_dim, latent_dim)                  # E -> Dz

        self.from_latent = nn.Linear(latent_dim, embed_dim)                # Dz -> E
        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.patch_head = nn.Linear(embed_dim, channels * patch_size**2)   # E -> C*P^2

    def config(self):
        return dict(
            channels=self.channels, patch_size=self.patch_size,
            embed_dim=self.embed_dim, latent_dim=self.latent_dim,
            encoder_depth=self.encoder_depth, decoder_depth=self.decoder_depth,
            heads=self.heads, mlp_ratio=self.mlp_ratio, dropout=self.dropout,
        )

    def encode(self, image):
        # image: [B,C,H,W]
        B, _, H, W = image.shape
        P = self.patch_size
        if H % P or W % P:
            raise ValueError(f"{H}x{W} must be divisible by patch size {P}")

        feature = self.patch_embed(image)                                  # [B,E,h,w]
        _, _, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)                        # [B,N,E]
        tokens = tokens + self.position(h, w, tokens.device, tokens.dtype) # [B,N,E]
        for block in self.encoder:
            tokens = block(tokens)                                         # [B,N,E]
        latent = self.to_latent(self.encoder_norm(tokens))                 # [B,N,Dz]
        assert latent.shape == (B, h * w, self.latent_dim)
        return latent, (h, w)

    def decode(self, latent, grid_size):
        # latent: [B,N,Dz]
        B, N, Dz = latent.shape
        h, w = grid_size
        if Dz != self.latent_dim or N != h * w:
            raise ValueError("latent shape and grid_size do not agree")

        tokens = self.from_latent(latent)                                  # [B,N,E]
        tokens = tokens + self.position(h, w, tokens.device, tokens.dtype) # [B,N,E]
        for block in self.decoder:
            tokens = block(tokens)                                         # [B,N,E]
        patch_pixels = self.patch_head(self.decoder_norm(tokens))          # [B,N,C*P^2]

        P, C = self.patch_size, self.channels
        patches = patch_pixels.reshape(B, h, w, C, P, P)                   # [B,h,w,C,P,P]
        patches = patches.permute(0, 3, 1, 4, 2, 5)                        # [B,C,h,P,w,P]
        image = patches.reshape(B, C, h * P, w * P)                        # [B,C,H,W]
        return torch.sigmoid(image)                                        # [B,C,H,W]

    def forward(self, image):
        latent, grid_size = self.encode(image)                             # [B,N,Dz], (h,w)
        return self.decode(latent, grid_size)                              # [B,C,H,W]


def tokens_to_grid(tokens, grid_size):
    # [B,N,D] -> [B,D,h,w]
    B, N, D = tokens.shape
    h, w = grid_size
    if N != h * w:
        raise ValueError("token count does not match grid")
    return tokens.transpose(1, 2).reshape(B, D, h, w)


def grid_to_tokens(grid):
    # [B,D,h,w] -> [B,N,D]
    B, D, h, w = grid.shape
    return grid.reshape(B, D, h * w).transpose(1, 2)


# -----------------------------------------------------------------------------
# Frozen tokenizer and trainable real-space latent operator
# -----------------------------------------------------------------------------

class LatentOperator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim + 6, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, condition):
        # z: [B,Dz,h,w], condition: [B,6,h,w]
        residual = self.net(torch.cat([z, condition], dim=1))              # [B,Dz,h,w]
        return z + residual                                                # [B,Dz,h,w]


def get_padding(H, W, multiple):
    ph, pw = (-H) % multiple, (-W) % multiple
    top, left = ph // 2, pw // 2
    return left, pw - left, top, ph - top                                 # left,right,top,bottom


def pad_image(image, padding):
    # [B,C,H,W] -> [B,C,H_pad,W_pad]
    return image if padding == (0, 0, 0, 0) else F.pad(image, padding, mode="replicate")


def crop_image(image, padding, output_size):
    # [B,C,H_pad,W_pad] -> [B,C,H_out,W_out]
    left, _, top, _ = padding
    H, W = output_size
    return image[:, :, top:top + H, left:left + W]


def operator_condition(B, grid_size, patch_size, padding, low_size, output_size, device, dtype):
    """Return [x,y,patch_dx,patch_dy,low/out_x,low/out_y] at every latent token."""
    h, w = grid_size
    Hlow, Wlow = low_size
    Hout, Wout = output_size
    left, _, top, _ = padding

    x_pixel = (torch.arange(w, device=device, dtype=dtype) + 0.5) * patch_size - left  # [w]
    y_pixel = (torch.arange(h, device=device, dtype=dtype) + 0.5) * patch_size - top   # [h]
    x, y = 2.0 * x_pixel / Wout - 1.0, 2.0 * y_pixel / Hout - 1.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")                                    # [h,w]
    q = torch.stack([
        xx,
        yy,
        torch.full_like(xx, 2.0 * patch_size / Wout),
        torch.full_like(yy, 2.0 * patch_size / Hout),
        torch.full_like(xx, float(Wlow) / Wout),
        torch.full_like(yy, float(Hlow) / Hout),
    ])                                                                                # [6,h,w]
    return q[None].expand(B, -1, -1, -1)                                              # [B,6,h,w]


class FrozenTokenizerSR(nn.Module):
    def __init__(self, tokenizer, operator_hidden_dim=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.operator = LatentOperator(tokenizer.latent_dim, operator_hidden_dim)

        # Explicit freeze: these weights never change during SR training.
        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()  # keep dropout, etc. disabled in the frozen tokenizer
        return self

    def forward(self, low, output_size):
        # low: [B,1,Hlow,Wlow]
        B, _, Hlow, Wlow = low.shape
        Hout, Wout = output_size
        base = resize(low, output_size)                                     # [B,1,Hout,Wout]
        padding = get_padding(Hout, Wout, self.tokenizer.patch_size)
        base_pad = pad_image(base, padding)                                 # [B,1,Hpad,Wpad]

        with torch.no_grad():
            base_tokens, grid_size = self.tokenizer.encode(base_pad)        # [B,N,Dz], (h,w)
        base_grid = tokens_to_grid(base_tokens, grid_size)                  # [B,Dz,h,w]
        condition = operator_condition(
            B, grid_size, self.tokenizer.patch_size, padding,
            (Hlow, Wlow), output_size, base_grid.device, base_grid.dtype,
        )                                                                   # [B,6,h,w]
        predicted_grid = self.operator(base_grid, condition)                # [B,Dz,h,w]
        predicted_tokens = grid_to_tokens(predicted_grid)                   # [B,N,Dz]

        # Not under no_grad: gradients pass through the frozen decoder to the operator.
        decoded_pad = self.tokenizer.decode(predicted_tokens, grid_size)    # [B,1,Hpad,Wpad]
        prediction = crop_image(decoded_pad, padding, output_size)          # [B,1,Hout,Wout]
        return prediction, predicted_grid

    @torch.no_grad()
    def encode_target(self, target):
        # target: [B,1,Hout,Wout]
        H, W = target.shape[-2:]
        padding = get_padding(H, W, self.tokenizer.patch_size)
        tokens, grid_size = self.tokenizer.encode(pad_image(target, padding)) # [B,N,Dz]
        return tokens_to_grid(tokens, grid_size)                              # [B,Dz,h,w]


# -----------------------------------------------------------------------------
# Checkpoints, plots, and random sizes
# -----------------------------------------------------------------------------


def load_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def save_autoencoder(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config(), "state": model.state_dict()}, path)


def load_autoencoder(path, device):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; train the autoencoder first")
    checkpoint = load_dict(path, device)
    model = ViTAutoencoder(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state"])
    return model


def save_superres(model, path, autoencoder_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "hidden_dim": model.operator.hidden_dim,
        "operator_state": model.operator.state_dict(),
        "autoencoder_checkpoint": str(autoencoder_path),
    }, path)


def load_superres(autoencoder_path, superres_path, device):
    if not superres_path.exists():
        raise FileNotFoundError(f"Missing {superres_path}; train the superres first")
    tokenizer = load_autoencoder(autoencoder_path, device)
    checkpoint = load_dict(superres_path, device)
    model = FrozenTokenizerSR(tokenizer, checkpoint["hidden_dim"]).to(device)
    model.operator.load_state_dict(checkpoint["operator_state"])
    return model


def take_eight(dataset, device):
    return torch.stack([dataset[i] for i in range(8)]).to(device)           # [8,1,H,W]


def make_figure(rows, names, cmaps=None):
    """Each row is [8,1,H_row,W_row]."""

    if cmaps is None:
        cmaps = ["Spectral_r"] * len(rows)
    
    fig, axes = plt.subplots(len(rows), 8, figsize=(16, 2 * len(rows)), squeeze=False)
    for r, (row, name) in enumerate(zip(rows, names)):
        is_diff = "difference" in name.lower()
        vmax =  0.1 if r==(len(rows)-1) else  1
        vmin = -0.1 if r==(len(rows)-1) else  0
        for c in range(8):
            axes[r, c].pcolormesh(
                row[c, 0].detach().cpu().numpy(), cmap=cmaps[r], vmin=vmin, vmax=vmax,
            )
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"example {c + 1}")
        axes[r, 0].text(-0.10, 0.5, name, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right")
    fig.tight_layout()
    return fig


def random_multiple(low, high, multiple):
    lo, hi = math.ceil(low / multiple), math.floor(high / multiple)
    if lo > hi:
        raise ValueError(f"No multiple of {multiple} in [{low}, {high}]")
    return random.randint(lo, hi) * multiple


def random_sr_sizes(args):
    Hout, Wout = random.randint(args.target_min, args.target_max), random.randint(args.target_min, args.target_max)
    Hlow = random.randint(args.low_min, min(args.low_max, Hout - 1))
    Wlow = random.randint(args.low_min, min(args.low_max, Wout - 1))
    return (Hout, Wout), (Hlow, Wlow)


# -----------------------------------------------------------------------------
# Stage 1: autoencoder
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate_autoencoder(model, loader, device, size):
    model.eval()
    error, count = 0.0, 0
    for source in loader:                                                   # [B,1,128,128]
        target = resize(source.to(device), size)                            # [B,1,H,W]
        output = model(target)                                              # [B,1,H,W]
        error += F.l1_loss(output, target, reduction="sum").item()
        count += target.numel()
    return error / count


def train_autoencoder(model, train_loader, val_loader, val_set, device, run, args, path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.autoencoder_lr,
                                  weight_decay=args.weight_decay)
    step = 0
    for epoch in range(1, args.autoencoder_epochs + 1):
        model.train()
        for source in train_loader:                                        # [B,1,128,128]
            H = random_multiple(args.autoencoder_min_size, args.autoencoder_max_size, model.patch_size)
            W = random_multiple(args.autoencoder_min_size, args.autoencoder_max_size, model.patch_size)
            target = resize(source.to(device), (H, W))                     # [B,1,H,W]
            output = model(target)                                         # [B,1,H,W]
            loss = F.l1_loss(output, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            run.log({"autoencoder/train_loss": loss.item(), "autoencoder/batch": step,
                     "autoencoder/epoch": epoch, "autoencoder/height": H, "autoencoder/width": W})
            step += 1

        val_loss = validate_autoencoder(model, val_loader, device, (args.image_size, args.image_size))
        true = take_eight(val_set, device)                                  # [8,1,128,128]
        with torch.no_grad():
            decoded = model(true)                                           # [8,1,128,128]
        diff = decoded - true                                               # [8,1,128,128]
        fig = make_figure([true, decoded, diff],
                          ["true input", "decoded", "difference"],
                          ["Spectral_r", "Spectral_r", "coolwarm"])
        run.log({"autoencoder/validation_loss": val_loss,
                 "autoencoder/examples": wandb.Image(fig), "autoencoder/epoch": epoch})
        plt.close(fig)
        save_autoencoder(model, path)
        print(f"autoencoder epoch {epoch}: validation L1 = {val_loss:.6f}", flush=True)


# -----------------------------------------------------------------------------
# Stage 2: frozen-tokenizer super-resolution
# -----------------------------------------------------------------------------

@torch.no_grad()
def validate_superres(model, loader, device, low_size, output_size):
    model.eval()
    error, count = 0.0, 0
    for source in loader:                                                   # [B,1,128,128]
        target = resize(source.to(device), output_size)                     # [B,1,Hout,Wout]
        low = resize(target, low_size)                                      # [B,1,Hlow,Wlow]
        output, _ = model(low, output_size)                                 # [B,1,Hout,Wout]
        error += F.l1_loss(output, target, reduction="sum").item()
        count += target.numel()
    return error / count


def train_superres(model, train_loader, val_loader, train_set, device, run, args, path, ae_path):
    # Only operator parameters are optimized; the tokenizer is absent here.
    optimizer = torch.optim.AdamW(model.operator.parameters(), lr=args.superres_lr,
                                  weight_decay=args.weight_decay)
    assert all(not p.requires_grad for p in model.tokenizer.parameters())
    step = 0
    val_output = (args.val_output_height, args.val_output_width)
    val_low = (args.val_low_height, args.val_low_width)

    for epoch in range(1, args.superres_epochs + 1):
        model.train()
        for source in train_loader:                                        # [B,1,128,128]
            output_size, low_size = random_sr_sizes(args)
            target = resize(source.to(device), output_size)                # [B,1,Hout,Wout]
            low = resize(target, low_size)                                  # [B,1,Hlow,Wlow]
            prediction, z_prediction = model(low, output_size)             # [B,1,Hout,Wout], [B,Dz,h,w]
            image_loss = F.l1_loss(prediction, target)

            if args.latent_loss_weight > 0:
                z_target = model.encode_target(target)                      # [B,Dz,h,w]
                latent_loss = F.l1_loss(z_prediction, z_target)
            else:
                latent_loss = image_loss.new_zeros(())
            loss = image_loss + args.latent_loss_weight * latent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            assert all(p.grad is None for p in model.tokenizer.parameters())

            run.log({
                "superres/train_loss": loss.item(),
                "superres/train_image_loss": image_loss.item(),
                "superres/train_latent_loss": latent_loss.item(),
                "superres/batch": step, "superres/epoch": epoch,
                "superres/output_height": output_size[0], "superres/output_width": output_size[1],
                "superres/low_height": low_size[0], "superres/low_width": low_size[1],
            })
            step += 1

        val_loss = validate_superres(model, val_loader, device, val_low, val_output)
        source = take_eight(train_set, device)                              # [8,1,128,128]
        true = resize(source, val_output)                                   # [8,1,Hout,Wout]
        low = resize(true, val_low)                                         # [8,1,Hlow,Wlow]
        with torch.no_grad():
            prediction, _ = model(low, val_output)                          # [8,1,Hout,Wout]
        diff = (prediction - true).abs()                                    # [8,1,Hout,Wout]
        fig = make_figure(
            [true, low, prediction, diff],
            ["true high resolution", "low-resolution prompt", "super-resolved", "absolute difference"],
            ["turbo", "turbo", "turbo", "coolwarm"],
        )
        run.log({"superres/validation_loss": val_loss,
                 "superres/examples": wandb.Image(fig), "superres/epoch": epoch})
        plt.close(fig)
        save_superres(model, path, ae_path)
        print(f"superres epoch {epoch}: validation L1 = {val_loss:.6f}", flush=True)


# -----------------------------------------------------------------------------
# Inference and CLI
# -----------------------------------------------------------------------------

@torch.no_grad()
def infer(model, dataset, device, args):
    model.eval()
    source = dataset[args.inference_index][None].to(device)                 # [1,1,128,128]
    low = resize(source, (args.inference_low_height, args.inference_low_width))  # [1,1,Hlow,Wlow]
    output, _ = model(low, (args.output_height, args.output_width))         # [1,1,Hout,Wout]
    path = Path(args.output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, output[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    print(f"saved {args.output_height}x{args.output_width} image to {path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--validation-size", type=int, default=1000)
    p.add_argument("--sigma-min", type=float, default=20.0)
    p.add_argument("--sigma-max", type=float, default=40.0)

    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--encoder-depth", type=int, default=2)
    p.add_argument("--decoder-depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--operator-hidden-dim", type=int, default=64)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--autoencoder-epochs", type=int, default=50)
    p.add_argument("--superres-epochs", type=int, default=50)
    p.add_argument("--autoencoder-lr", type=float, default=3e-4)
    p.add_argument("--superres-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--latent-loss-weight", type=float, default=0.0)

    p.add_argument("--autoencoder-min-size", type=int, default=64)
    p.add_argument("--autoencoder-max-size", type=int, default=128)
    p.add_argument("--target-min", type=int, default=64)
    p.add_argument("--target-max", type=int, default=128)
    p.add_argument("--low-min", type=int, default=16)
    p.add_argument("--low-max", type=int, default=48)
    p.add_argument("--val-output-height", type=int, default=128)
    p.add_argument("--val-output-width", type=int, default=128)
    p.add_argument("--val-low-height", type=int, default=32)
    p.add_argument("--val-low-width", type=int, default=32)

    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--autoencoder-checkpoint", default=None)
    p.add_argument("--superres-checkpoint", default=None)
    p.add_argument("--output-height", type=int, default=197)
    p.add_argument("--output-width", type=int, default=311)
    p.add_argument("--inference-low-height", type=int, default=32)
    p.add_argument("--inference-low-width", type=int, default=32)
    p.add_argument("--inference-index", type=int, default=0)
    p.add_argument("--output-file", default="vit_superres.png")

    p.add_argument("--wandb-project", default="gaussian-vit-frozen-tokenizer-superres")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=4)
    return p.parse_args()


def main():

    args = parse_args()

    if args.train_size < 8 or args.validation_size < 8:
        raise ValueError("train-size and validation-size must be at least 8")
    if not 0 < args.sigma_min <= args.sigma_max:
        raise ValueError("require 0 < sigma-min <= sigma-max")
    if args.low_min >= args.target_min:
        raise ValueError("low-min must be smaller than target-min")
    if args.latent_loss_weight < 0:
        raise ValueError("latent-loss-weight must be nonnegative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    print("device:", device, flush=True)

    train_set = GaussianDataset(args.train_size, args.image_size, args.sigma_min, args.sigma_max, args.seed)
    val_set = GaussianDataset(args.validation_size, args.image_size, args.sigma_min, args.sigma_max, args.seed + 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    checkpoint_dir = Path(args.checkpoint_dir)
    ae_path = Path(args.autoencoder_checkpoint or checkpoint_dir / "vit_autoencoder.pt")
    sr_path = Path(args.superres_checkpoint or checkpoint_dir / "vit_superres_operator.pt")

    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name=args.wandb_run_name, mode=args.wandb_mode,
                    config=vars(args)) as run:


        model = ViTAutoencoder(
            channels=1, patch_size=args.patch_size, embed_dim=args.embed_dim,
            latent_dim=args.latent_dim, encoder_depth=args.encoder_depth,
            decoder_depth=args.decoder_depth, heads=args.heads,
            mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        ).to(device)
        print(torchinfo.summary(model, input_size=(1, 1, args.image_size, args.image_size), depth=3), flush=True)

        print("Training autoencoder...", flush=True)
        train_autoencoder(model, train_loader, val_loader, val_set,
                            device, run, args, ae_path)

        tokenizer = load_autoencoder(ae_path, device)
        model = FrozenTokenizerSR(tokenizer, args.operator_hidden_dim).to(device)
        print("frozen tokenizer parameters:", sum(p.numel() for p in model.tokenizer.parameters()), flush=True)
        print("trainable operator parameters:", sum(p.numel() for p in model.operator.parameters()), flush=True)
        print("Training superres...", flush=True)
        train_superres(model, train_loader, val_loader, train_set,
                        device, run, args, sr_path, ae_path)


        print("Inferring...", flush=True)
        infer(load_superres(ae_path, sr_path, device), val_set, device, args)


if __name__ == "__main__":
    main()
