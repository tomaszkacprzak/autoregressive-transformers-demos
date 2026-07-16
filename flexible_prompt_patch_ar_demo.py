#!/usr/bin/env python3
"""Flexible prompt/request autoregressive patch demo.

Dataset
-------
Each example is a 128x128 grayscale image containing one axis-aligned 2-D
Gaussian. sigma_row and sigma_col are sampled independently from [10, 20].
Images are split into 16x16 patches, giving N=64 patches and D=256 values per
patch.

Model
-----
1. Train a continuous patch autoencoder (no codebook, no quantization).
2. Freeze its encoder and use it to embed known patch values.
3. Train one transformer on a sequence

       [prompt patches, q1, c1, q2, c2, ..., qK]

   where q_t contains only the requested position and c_t contains the value of
   the previously generated/teacher-forced patch. A block-causal mask lets:

   * prompt tokens attend bidirectionally to prompt tokens only;
   * generation tokens attend to every prompt token;
   * generation tokens attend only to earlier generation tokens.

The output at q_t directly predicts the D raw values of requested patch t.

Shape symbols used in comments
------------------------------
B=batch size, H=W=128, P=16, G=8, N=64, D=256, E=d_model,
M=prompt count, R=request count, K=unknown requested count,
S=M+(2*K-1)=total transformer sequence length.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

NUM_PLOT_EXAMPLES = 8
PROMPT_ROLE, QUERY_ROLE, CONTENT_ROLE = 0, 1, 2



class GaussianDatasetNondeterministic(Dataset):
    """Nondeterministic collection of axis-aligned 2-D Gaussian images."""

    def __init__(
        self,
        size: int,
        image_size: int,
        sigma_min: float,
        sigma_max: float,
        mean_row: float,
        mean_col: float,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        
        self.sigma_row = torch.empty(size,4).uniform_(sigma_min, sigma_max, generator=generator)  # [dataset_size]
        self.sigma_col = torch.empty(size,4).uniform_(sigma_min, sigma_max, generator=generator)  # [dataset_size]
        self.mean_row = mean_row
        self.mean_col = mean_col

        coordinates = torch.arange(image_size, dtype=torch.float32)  # [H]
        self.rows, self.cols = torch.meshgrid(
            coordinates, coordinates, indexing="ij"
        )  # each [H,W]

    def __len__(self) -> int:
        return int(self.sigma_row.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:

        def gaussian(pos_id):

            sigma_row, sigma_col = self.sigma_row[index, pos_id], self.sigma_col[index, pos_id]
            exponent = (
                ((self.rows - self.mean_row) / sigma_row).square()
                + ((self.cols - self.mean_col) / sigma_col).square()
            )  # [H,W]
            image = torch.exp(-0.5 * exponent)  # [H,W]
            image = image[::2, ::2] # downsample by 2
            return image

        image00 = gaussian(0)
        image01 = gaussian(1)
        image10 = gaussian(2)
        image11 = gaussian(3)
        image0  = torch.cat([image00, image01], dim=1)
        image1  = torch.cat([image10, image11], dim=1)
        image   = torch.cat([image0, image1], dim=0)

        return image.unsqueeze(0)  # [1,H,W]


class GaussianDataset(Dataset):
    def __init__(
        self,
        size: int,
        image_size: int,
        sigma_min: float,
        sigma_max: float,
        mean_row: float,
        mean_col: float,
        seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.sigma_row = torch.empty(size).uniform_(
            sigma_min, sigma_max, generator=generator
        )  # [dataset_size]
        self.sigma_col = torch.empty(size).uniform_(
            sigma_min, sigma_max, generator=generator
        )  # [dataset_size]
        self.mean_row = mean_row
        self.mean_col = mean_col
        coordinates = torch.arange(image_size, dtype=torch.float32)
        self.rows, self.cols = torch.meshgrid(
            coordinates, coordinates, indexing="ij"
        )  # each [H,W]

    def __len__(self) -> int:
        return len(self.sigma_row)

    def __getitem__(self, index: int) -> torch.Tensor:
        exponent = (
            ((self.rows - self.mean_row) / self.sigma_row[index]).square()
            + ((self.cols - self.mean_col) / self.sigma_col[index]).square()
        )  # [H,W]
        return torch.exp(-0.5 * exponent).unsqueeze(0)  # [1,H,W]


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B,C,H,W] -> [B,N,C*P*P] in raster order."""
    batch, channels, height, width = images.shape
    rows, cols = height // patch_size, width // patch_size
    x = images.reshape(batch, channels, rows, patch_size, cols, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(batch, rows * cols, channels * patch_size**2)  # [B,N,D]


def unpatchify(
    patches: torch.Tensor,
    image_size: int,
    patch_size: int,
    channels: int = 1,
) -> torch.Tensor:
    """[B,N,C*P*P] -> [B,C,H,W]."""
    batch = patches.shape[0]
    grid = image_size // patch_size
    x = patches.reshape(batch, grid, grid, channels, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(batch, channels, image_size, image_size)  # [B,C,H,W]


def gather_patches(patches: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """patches [B,N,D], positions [B,L] -> selected [B,L,D]."""
    index = positions.unsqueeze(-1).expand(-1, -1, patches.shape[-1])  # [B,L,D]
    return patches.gather(1, index)  # [B,L,D]


@dataclass
class PatchLayout:
    prompt_positions: torch.Tensor  # [B,M]
    target_positions: torch.Tensor  # [B,K], random generation order
    prompt_mask: torch.Tensor  # [B,N]
    request_mask: torch.Tensor  # [B,N], may overlap prompt_mask
    prompt_count: int
    request_count: int
    target_count: int

    def to(self, device: torch.device) -> "PatchLayout":
        return PatchLayout(
            self.prompt_positions.to(device),
            self.target_positions.to(device),
            self.prompt_mask.to(device),
            self.request_mask.to(device),
            self.prompt_count,
            self.request_count,
            self.target_count,
        )


def rand_int(low: int, high: int, generator: torch.Generator) -> int:
    """Uniform integer in the inclusive interval [low, high]."""
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def sample_layout(
    batch_size: int,
    num_patches: int,
    generator: torch.Generator,
    min_prompt: int,
    max_prompt: int,
    min_request: int,
    max_request: int,
    prompt_count: int | None = None,
    request_count: int | None = None,
) -> PatchLayout:
    """Sample arbitrary prompt/request sets with shared counts in one batch.

    Using shared M and R avoids padding. Every image still receives an
    independent set of positions and an independent random target order.
    Requested positions are taken from non-prompt positions first; if R>N-M,
    the remaining requested positions overlap the prompt and are copied later.
    """
    m = prompt_count if prompt_count is not None else rand_int(min_prompt, max_prompt, generator)
    r = request_count if request_count is not None else rand_int(min_request, max_request, generator)
    k = min(r, num_patches - m)
    overlap = r - k

    permutations = torch.rand(batch_size, num_patches, generator=generator).argsort(1)  # [B,N]
    prompt_positions = permutations[:, :m]  # [B,M]
    target_positions = permutations[:, m : m + k]  # [B,K]

    prompt_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)  # [B,N]
    request_mask = torch.zeros_like(prompt_mask)  # [B,N]
    prompt_mask.scatter_(1, prompt_positions, True)
    request_mask.scatter_(1, target_positions, True)
    if overlap:
        request_mask.scatter_(1, prompt_positions[:, :overlap], True)

    return PatchLayout(
        prompt_positions,
        target_positions,
        prompt_mask,
        request_mask,
        m,
        r,
        k,
    )


class ContinuousTokenizer(nn.Module):
    """Small patch autoencoder: [...,D] -> [...,E] -> [...,D]."""

    def __init__(self, patch_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_dim),
            nn.Sigmoid(),
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        leading = patches.shape[:-1]
        z = self.encoder(patches.reshape(-1, self.patch_dim))  # [prod(leading),E]
        return z.reshape(*leading, self.latent_dim)  # [...,E]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        leading = latents.shape[:-1]
        x = self.decoder(latents.reshape(-1, self.latent_dim))  # [prod(leading),D]
        return x.reshape(*leading, self.patch_dim)  # [...,D]

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(patches))  # [...,D]


class FlexiblePatchAR(nn.Module):
    """Single-transformer prompt-conditioned continuous patch generator."""

    def __init__(
        self,
        tokenizer: ContinuousTokenizer,
        grid_size: int,
        patch_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if tokenizer.latent_dim != d_model:
            raise ValueError("tokenizer latent_dim must equal d_model")
        self.tokenizer = tokenizer.requires_grad_(False)
        self.grid_size = grid_size
        self.num_patches = grid_size**2
        self.patch_dim = patch_dim
        self.d_model = d_model

        self.row_embedding = nn.Embedding(grid_size, d_model)
        self.col_embedding = nn.Embedding(grid_size, d_model)
        self.role_embedding = nn.Embedding(3, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.patch_head = nn.Linear(d_model, patch_dim)

        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.col_embedding.weight, std=0.02)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.patch_head.weight)
        nn.init.zeros_(self.patch_head.bias)

    def spatial(self, positions: torch.Tensor) -> torch.Tensor:
        rows = positions // self.grid_size  # [B,L]
        cols = positions % self.grid_size  # [B,L]
        return self.row_embedding(rows) + self.col_embedding(cols)  # [B,L,E]

    def role(self, role_id: int, device: torch.device) -> torch.Tensor:
        index = torch.tensor(role_id, device=device, dtype=torch.long)
        return self.role_embedding(index).view(1, 1, self.d_model)  # [1,1,E]

    def build_sequence(
        self,
        prompt_patches: torch.Tensor,
        prompt_positions: torch.Tensor,
        target_positions: torch.Tensor,
        previous_patches: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Build [prompts, q1,c1,...,qK].

        prompt_patches [B,M,D], prompt_positions [B,M],
        target_positions [B,K], previous_patches [B,K-1,D].
        Returns sequence [B,S,E] and M.
        """
        batch, prompt_count = prompt_positions.shape
        target_count = target_positions.shape[1]
        if previous_patches.shape[:2] != (batch, target_count - 1):
            raise ValueError("previous_patches must have shape [B,K-1,D]")

        with torch.no_grad():
            prompt_latent = self.tokenizer.encode(prompt_patches)  # [B,M,E]
        prompt_tokens = (
            prompt_latent
            + self.spatial(prompt_positions)
            + self.role(PROMPT_ROLE, prompt_positions.device)
        )  # [B,M,E]

        query_tokens = (
            self.spatial(target_positions)
            + self.role(QUERY_ROLE, target_positions.device)
        )  # [B,K,E]

        if target_count == 1:
            suffix = query_tokens  # [B,1,E]
        else:
            with torch.no_grad():
                previous_latent = self.tokenizer.encode(previous_patches)  # [B,K-1,E]
            content_tokens = (
                previous_latent
                + self.spatial(target_positions[:, :-1])
                + self.role(CONTENT_ROLE, target_positions.device)
            )  # [B,K-1,E]
            pairs = torch.stack((query_tokens[:, :-1], content_tokens), dim=2)  # [B,K-1,2,E]
            prefix = pairs.reshape(batch, 2 * (target_count - 1), self.d_model)  # [B,2K-2,E]
            suffix = torch.cat((prefix, query_tokens[:, -1:]), dim=1)  # [B,2K-1,E]

        return torch.cat((prompt_tokens, suffix), dim=1), prompt_count  # [B,S,E], M

    @staticmethod
    def block_causal_mask(
        prompt_count: int,
        total_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return bool mask [S,S], where True means attention is forbidden."""
        mask = torch.ones(total_length, total_length, dtype=torch.bool, device=device)
        mask[:prompt_count, :prompt_count] = False  # prompt <-> prompt
        suffix_length = total_length - prompt_count
        causal_suffix = torch.triu(
            torch.ones(suffix_length, suffix_length, dtype=torch.bool, device=device),
            diagonal=1,
        )  # [2K-1,2K-1]
        mask[prompt_count:, :prompt_count] = False  # suffix -> all prompts
        mask[prompt_count:, prompt_count:] = causal_suffix  # suffix causal self-attention
        return mask

    def predict_queries(
        self,
        prompt_patches: torch.Tensor,
        prompt_positions: torch.Tensor,
        target_positions: torch.Tensor,
        previous_patches: torch.Tensor,
    ) -> torch.Tensor:
        """Return direct continuous predictions [B,K,D]."""
        sequence, prompt_count = self.build_sequence(
            prompt_patches,
            prompt_positions,
            target_positions,
            previous_patches,
        )  # [B,S,E], M
        mask = self.block_causal_mask(prompt_count, sequence.shape[1], sequence.device)  # [S,S]
        hidden = self.transformer(sequence, mask=mask)  # [B,S,E]
        query_hidden = hidden[:, prompt_count::2]  # [B,K,E]
        return torch.sigmoid(self.patch_head(query_hidden))  # [B,K,D]

    def forward(
        self,
        prompt_patches: torch.Tensor,
        prompt_positions: torch.Tensor,
        target_patches: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        # Only earlier target values are supplied: [B,K-1,D].
        return self.predict_queries(
            prompt_patches,
            prompt_positions,
            target_positions,
            target_patches[:, :-1],
        )  # [B,K,D]


class WBLogger:
    """W&B wrapper; disabled mode works even when wandb is not installed."""

    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        self.wandb = None
        self.run = None
        if args.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as error:
            raise SystemExit(
                "Install W&B with `pip install wandb`, or use --wandb-mode disabled"
            ) from error
        self.wandb = wandb
        self.run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=config,
        )

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        if self.run:
            self.run.define_metric(*args, **kwargs)

    def log(self, values: dict[str, Any]) -> None:
        if self.run:
            self.run.log(values)

    def log_figure(self, values: dict[str, Any], key: str, figure: plt.Figure) -> None:
        if self.run and self.wandb:
            values = dict(values)
            values[key] = self.wandb.Image(figure)
            self.run.log(values)

    def finish(self) -> None:
        if self.run:
            self.run.finish()


def tokenizer_figure(true_images: torch.Tensor, decoded_images: torch.Tensor, title: str) -> plt.Figure:
    true = true_images[:8, 0].detach().cpu().numpy()  # [8,H,W]
    decoded = decoded_images[:8, 0].detach().cpu().numpy()  # [8,H,W]
    rows = (true, decoded, true - decoded)
    labels = ("True image", "Tokenizer decoded", "Difference")
    fig, axes = plt.subplots(3, 8, figsize=(16, 6), squeeze=False)
    fig.suptitle(title)
    cmaps = ["turbo", "turbo", "bwr"]
    for row, images in enumerate(rows):
        vmin = 0 if row<2 else None
        vmax = 1 if row<2 else None
        for col in range(8):
            axes[row, col].imshow(images[col], cmap=cmaps[row], vmin=vmin, vmax=vmax)
            axes[row, col].set(xticks=[], yticks=[])
            if row == 0:
                axes[row, col].set_title(f"Example {col + 1}", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(labels[row])
    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    return fig


def generation_figure(
    true_requested: torch.Tensor,
    prompt_only: torch.Tensor,
    generated_requested: torch.Tensor,
    title: str,
) -> plt.Figure:
    true = true_requested[:8, 0].detach().cpu().numpy()  # [8,H,W]
    prompt = prompt_only[:8, 0].detach().cpu().numpy()  # [8,H,W]
    generated = generated_requested[:8, 0].detach().cpu().numpy()  # [8,H,W]
    rows = (true, prompt, generated, true - generated)
    labels = (
        "True requested",
        "Prompt only",
        "Generated requested",
        "Difference",
    )
    fig, axes = plt.subplots(4, 8, figsize=(16, 8), squeeze=False)
    fig.suptitle(title)
    cmaps = ["turbo", "turbo", "turbo", "bwr"]
    for row, images in enumerate(rows):
        vmin = 0 if row<3 else None
        vmax = 1 if row<3 else None
        for col in range(8):
            axes[row, col].imshow(images[col], cmap=cmaps[row], vmin=vmin, vmax=vmax)
            axes[row, col].set(xticks=[], yticks=[])
            if row == 0:
                axes[row, col].set_title(f"Example {col + 1}", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(labels[row])

    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    return fig


@torch.no_grad()
def reconstruct_images(
    tokenizer: ContinuousTokenizer,
    images: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    patches = patchify(images, patch_size)  # [B,N,D]
    decoded = tokenizer(patches)  # [B,N,D]
    return unpatchify(decoded, image_size, patch_size)  # [B,1,H,W]


@torch.no_grad()
def generate_targets(
    model: FlexiblePatchAR,
    all_patches: torch.Tensor,
    prompt_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Autoregressively generate [B,K,D] from prompt positions [B,M]."""
    prompt_patches = gather_patches(all_patches, prompt_positions)  # [B,M,D]
    generated: list[torch.Tensor] = []
    for step in range(target_positions.shape[1]):
        positions = target_positions[:, : step + 1]  # [B,step+1]
        previous = (
            torch.stack(generated, dim=1)
            if generated
            else all_patches.new_empty(all_patches.shape[0], 0, all_patches.shape[-1])
        )  # [B,step,D]
        predictions = model.predict_queries(
            prompt_patches, prompt_positions, positions, previous
        )  # [B,step+1,D]
        generated.append(predictions[:, -1])  # one [B,D]
    return torch.stack(generated, dim=1)  # [B,K,D]


def masked_images(
    patches: torch.Tensor,
    mask: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    selected = patches * mask.unsqueeze(-1).to(patches.dtype)  # [B,N,D]
    return unpatchify(selected, image_size, patch_size)  # [B,1,H,W]


@torch.no_grad()
def generation_figure_inputs(
    model: FlexiblePatchAR,
    images: torch.Tensor,
    layout: PatchLayout,
    image_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patches = patchify(images, patch_size)  # [8,N,D]
    generated_targets = generate_targets(
        model, patches, layout.prompt_positions, layout.target_positions
    )  # [8,K,D]

    true_requested = masked_images(patches, layout.request_mask, image_size, patch_size)
    prompt_only = masked_images(patches, layout.prompt_mask, image_size, patch_size)

    canvas = torch.zeros_like(patches)  # [8,N,D]
    overlap = layout.request_mask & layout.prompt_mask  # [8,N]
    canvas = torch.where(overlap.unsqueeze(-1), patches, canvas)
    scatter_index = layout.target_positions.unsqueeze(-1).expand(-1, -1, patches.shape[-1])  # [8,K,D]
    canvas.scatter_(1, scatter_index, generated_targets)
    generated_requested = unpatchify(canvas, image_size, patch_size)  # [8,1,H,W]
    return true_requested, prompt_only, generated_requested


@torch.no_grad()
def tokenizer_val_loss(
    model: ContinuousTokenizer,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
) -> float:
    model.eval()
    error, count = 0.0, 0
    for images in loader:
        patches = patchify(images.to(device), patch_size)  # [B,N,D]
        decoded = model(patches)  # [B,N,D]
        error += F.mse_loss(decoded, patches, reduction="sum").item()
        count += patches.numel()
    return error / count


@torch.no_grad()
def ar_val_loss(
    model: FlexiblePatchAR,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """Teacher-forced validation using the same layouts after every epoch."""
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 40_000)
    error, count = 0.0, 0
    for images in loader:
        images = images.to(device)  # [B,1,H,W]
        layout = sample_layout(
            images.shape[0],
            model.num_patches,
            generator,
            args.min_prompt_patches,
            args.max_prompt_patches,
            args.min_request_patches,
            args.max_request_patches,
        ).to(device)
        patches = patchify(images, args.patch_size)  # [B,N,D]
        prompt = gather_patches(patches, layout.prompt_positions)  # [B,M,D]
        targets = gather_patches(patches, layout.target_positions)  # [B,K,D]
        predictions = model(
            prompt, layout.prompt_positions, targets, layout.target_positions
        )  # [B,K,D]
        error += F.mse_loss(predictions, targets, reduction="sum").item()
        count += targets.numel()
    return error / count


def train_tokenizer(
    model: ContinuousTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    examples: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    logger: WBLogger,
    checkpoint: str | None,
) -> None:
   
    if checkpoint:
        if os.path.exists(checkpoint):
            model.load_state_dict(torch.load(checkpoint))
            print(f"Loaded checkpoint from {checkpoint}")
        else:
            print(f"Checkpoint {checkpoint} not found, starting from scratch")
            return
        model.load_state_dict(torch.load(checkpoint))
        print(f"Loaded checkpoint from {checkpoint}")
    else:
        print("No checkpoint found, starting from scratch")


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.tokenizer_lr, weight_decay=args.weight_decay
    )
    batch_step = 0
    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        losses = []
        for images in train_loader:
            patches = patchify(images.to(device), args.patch_size)  # [B,N,D]
            flat = patches.reshape(-1, model.patch_dim)  # [B*N,D]
            optimizer.zero_grad(set_to_none=True)
            decoded = model(flat)  # [B*N,D]
            loss = F.mse_loss(decoded, flat)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            logger.log(
                {
                    "tokenizer/batch_step": batch_step,
                    "tokenizer/train_loss": loss.item(),
                }
            )
            batch_step += 1

        validation = tokenizer_val_loss(model, val_loader, device, args.patch_size)
        fixed = examples.to(device)  # [8,1,H,W]
        decoded_images = reconstruct_images(
            model, fixed, args.image_size, args.patch_size
        )  # [8,1,H,W]
        figure = tokenizer_figure(fixed, decoded_images, f"Tokenizer epoch {epoch}")
        logger.log_figure(
            {
                "tokenizer/epoch": epoch,
                "tokenizer/epoch_train_loss": float(np.mean(losses)),
                "tokenizer/validation_loss": validation,
            },
            "tokenizer/comparison",
            figure,
        )
        plt.close(figure)
        print(
            f"[tokenizer] {epoch:03d}/{args.tokenizer_epochs:03d} "
            f"train={np.mean(losses):.6f} val={validation:.6f}"
        )


def train_ar(
    model: FlexiblePatchAR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    examples: torch.Tensor,
    plot_layout: PatchLayout,
    args: argparse.Namespace,
    device: torch.device,
    logger: WBLogger,
) -> None:
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.ar_lr, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed + 30_000)
    batch_step = 0

    for epoch in range(1, args.ar_epochs + 1):
        model.train()
        losses = []
        for images in train_loader:
            images = images.to(device)  # [B,1,H,W]
            layout = sample_layout(
                images.shape[0],
                model.num_patches,
                generator,
                args.min_prompt_patches,
                args.max_prompt_patches,
                args.min_request_patches,
                args.max_request_patches,
            ).to(device)
            patches = patchify(images, args.patch_size)  # [B,N,D]
            prompt = gather_patches(patches, layout.prompt_positions)  # [B,M,D]
            targets = gather_patches(patches, layout.target_positions)  # [B,K,D]

            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                prompt, layout.prompt_positions, targets, layout.target_positions
            )  # [B,K,D]
            loss = F.mse_loss(predictions, targets)
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()

            losses.append(loss.item())
            logger.log(
                {
                    "autoregressive/batch_step": batch_step,
                    "autoregressive/train_loss": loss.item(),
                    "autoregressive/prompt_count": layout.prompt_count,
                    "autoregressive/request_count": layout.request_count,
                    "autoregressive/target_count": layout.target_count,
                }
            )
            batch_step += 1

        validation = ar_val_loss(model, val_loader, device, args)
        fixed = examples.to(device)  # [8,1,H,W]
        layout = plot_layout.to(device)
        true_requested, prompt_only, generated_requested = generation_figure_inputs(
            model, fixed, layout, args.image_size, args.patch_size
        )  # each [8,1,H,W]
        title = (
            f"AR epoch {epoch}: prompt={layout.prompt_count}, "
            f"requested={layout.request_count}, generated={layout.target_count}"
        )
        figure = generation_figure(
            true_requested, prompt_only, generated_requested, title
        )
        logger.log_figure(
            {
                "autoregressive/epoch": epoch,
                "autoregressive/epoch_train_loss": float(np.mean(losses)),
                "autoregressive/validation_loss": validation,
            },
            "autoregressive/comparison",
            figure,
        )
        plt.close(figure)
        print(
            f"[autoregressive] {epoch:03d}/{args.ar_epochs:03d} "
            f"train={np.mean(losses):.6f} val={validation:.6f}"
        )


def fixed_examples(dataset: Dataset) -> torch.Tensor:
    return torch.stack([dataset[i] for i in range(NUM_PLOT_EXAMPLES)])  # [8,1,H,W]


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def define_metrics(logger: WBLogger) -> None:
    logger.define_metric("tokenizer/batch_step")
    logger.define_metric("tokenizer/train_loss", step_metric="tokenizer/batch_step")
    logger.define_metric("tokenizer/epoch")
    for key in (
        "tokenizer/epoch_train_loss",
        "tokenizer/validation_loss",
        "tokenizer/comparison",
    ):
        logger.define_metric(key, step_metric="tokenizer/epoch")

    logger.define_metric("autoregressive/batch_step")
    for key in (
        "autoregressive/train_loss",
        "autoregressive/prompt_count",
        "autoregressive/request_count",
        "autoregressive/target_count",
    ):
        logger.define_metric(key, step_metric="autoregressive/batch_step")
    logger.define_metric("autoregressive/epoch")
    for key in (
        "autoregressive/epoch_train_loss",
        "autoregressive/validation_loss",
        "autoregressive/comparison",
    ):
        logger.define_metric(key, step_metric="autoregressive/epoch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flexible prompt/request autoregressive Gaussian patch demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--sigma-min", type=float, default=20.0)
    parser.add_argument("--sigma-max", type=float, default=40.0)
    parser.add_argument("--mean-row", type=float, default=64.0)
    parser.add_argument("--mean-col", type=float, default=64.0)
    parser.add_argument("--dataset-type", type=str, default="deterministic", choices=["deterministic", "nondeterministic"])
    
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokenizer-epochs", type=int, default=20)
    parser.add_argument("--ar-epochs", type=int, default=40)
    parser.add_argument("--tokenizer-lr", type=float, default=3e-4)
    parser.add_argument("--ar-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--tokenizer-hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--min-prompt-patches", type=int, default=1)
    parser.add_argument("--max-prompt-patches", type=int, default=None)
    parser.add_argument("--min-request-patches", type=int, default=1)
    parser.add_argument("--max-request-patches", type=int, default=None)
    parser.add_argument("--plot-prompt-patches", type=int, default=8)
    parser.add_argument("--plot-request-patches", type=int, default=24)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("flexible_autoreg_outputs/")
    )
    parser.add_argument("--checkpoint", type=Path, default=None)

    parser.add_argument("--wandb-project", default="flexible-patch-ar-demo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size != 128 or args.patch_size != 16:
        raise ValueError("This demo is defined for 128x128 images and 16x16 patches")
    if args.train_size < 8 or args.val_size < 8:
        raise ValueError("train-size and val-size must be at least 8")
    if not 0 < args.sigma_min < args.sigma_max:
        raise ValueError("Require 0 < sigma-min < sigma-max")
    if args.d_model % args.num_heads:
        raise ValueError("d-model must be divisible by num-heads")
    if min(args.batch_size, args.tokenizer_epochs, args.ar_epochs, args.cpu_threads) < 1:
        raise ValueError("batch size, epochs, and cpu threads must be positive")

    num_patches = (args.image_size // args.patch_size) ** 2
    args.max_prompt_patches = args.max_prompt_patches or num_patches - 1
    args.max_request_patches = args.max_request_patches or num_patches
    if not 1 <= args.min_prompt_patches <= args.max_prompt_patches <= num_patches - 1:
        raise ValueError("Prompt range must satisfy 1 <= min <= max <= N-1")
    if not 1 <= args.min_request_patches <= args.max_request_patches <= num_patches:
        raise ValueError("Request range must satisfy 1 <= min <= max <= N")
    if not 1 <= args.plot_prompt_patches <= num_patches - 1:
        raise ValueError("plot-prompt-patches must be in [1,N-1]")
    if not 1 <= args.plot_request_patches <= num_patches:
        raise ValueError("plot-request-patches must be in [1,N]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_kwargs = dict(
        image_size=args.image_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        mean_row=args.mean_row,
        mean_col=args.mean_col,
    )

    DatasetClass = GaussianDataset if args.dataset_type == "deterministic" else GaussianDatasetNondeterministic

    train_set = DatasetClass(args.train_size, seed=args.seed, **data_kwargs)
    val_set = DatasetClass(args.val_size, seed=args.seed + 1, **data_kwargs)
    train_loader = make_loader(
        train_set, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, args.num_workers, device
    )

    grid_size = args.image_size // args.patch_size
    num_patches = grid_size**2
    patch_dim = args.patch_size**2
    examples = fixed_examples(val_set)  # [8,1,H,W]
    plot_layout = sample_layout(
        8,
        num_patches,
        torch.Generator().manual_seed(args.seed + 50_000),
        args.min_prompt_patches,
        args.max_prompt_patches,
        args.min_request_patches,
        args.max_request_patches,
        prompt_count=args.plot_prompt_patches,
        request_count=args.plot_request_patches,
    )

    tokenizer = ContinuousTokenizer(
        patch_dim, args.d_model, args.tokenizer_hidden_dim
    ).to(device)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(grid_size=grid_size, num_patches=num_patches, patch_dim=patch_dim)

    logger = WBLogger(args, config)
    try:
        define_metrics(logger)
        train_tokenizer(
            tokenizer, train_loader, val_loader, examples, args, device, logger, args.checkpoint
        )
        torch.save(
            {"state_dict": tokenizer.state_dict(), "config": config},
            args.output_dir / "continuous_tokenizer.pt",
        )

        model = FlexiblePatchAR(
            tokenizer,
            grid_size,
            patch_dim,
            args.d_model,
            args.num_heads,
            args.num_layers,
            args.ffn_dim,
            args.dropout,
        ).to(device)
        train_ar(
            model,
            train_loader,
            val_loader,
            examples,
            plot_layout,
            args,
            device,
            logger,
        )
        torch.save(
            {"state_dict": model.state_dict(), "config": config},
            args.output_dir / "flexible_patch_autoregressor.pt",
        )
    finally:
        logger.finish()

    print(f"Saved checkpoints in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
