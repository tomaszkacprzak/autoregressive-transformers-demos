#!/usr/bin/env python3
"""Flexible-prompt autoregressive patch prediction in continuous latent space.

Dataset
-------
Each example is a 128x128 grayscale image containing one axis-aligned 2-D
Gaussian. The row and column standard deviations are sampled independently
from [10, 20] pixels. Images are split into 16x16 patches, so each image has
an 8x8 patch grid.

Training stages
---------------
1. Train a continuous patch autoencoder (no codebook and no quantization):

       patch x_p -> latent z_p -> reconstructed patch x_hat_p.

2. Freeze the tokenizer and train a flexible-prompt autoregressive transformer
   directly in tokenizer latent space. For an ordered set of requested,
   non-prompt positions (u_1, ..., u_K), the transformer sequence is

       [prompt tokens, q_1, c_1, q_2, c_2, ..., q_K],

   where q_t contains the requested position but no patch value, and c_t
   contains the true previous latent during teacher forcing or the generated
   previous latent during inference.

3. During inference, generated latents are fed directly into later content
   tokens. No generated patch is decoded and re-encoded between AR steps. All
   requested latents are decoded together only after generation finishes.

Autoregressive loss choices
---------------------------
--ar-loss latent
    L = L_latent

--ar-loss decoder-aware
    L = lambda_z * L_latent + lambda_x * L_pixel

where

    L_latent = MSE(z_hat, z)
    L_pixel  = MSE(Decoder(z_hat), x).

The tokenizer is frozen during AR training. In decoder-aware mode, gradients
flow through the frozen decoder to z_hat and then to the transformer, but the
frozen decoder parameters are not updated.

Tensor symbols used in comments
-------------------------------
B : batch size
H,W : image height and width, both 128
P : patch side length, 16
G : patch-grid side length, 8
N : number of patches, G*G = 64
D : flattened patch dimension, P*P = 256
Z : tokenizer latent dimension
E : transformer model dimension
M : number of prompt patches in a batch
R : number of requested patches in a batch, including prompt/request overlap
K : number of requested patches that are not already prompts and must be made
S : transformer sequence length, M + (2*K - 1)
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
PROMPT_ROLE = 0
QUERY_ROLE = 1
CONTENT_ROLE = 2


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
    """Deterministic collection of axis-aligned 2-D Gaussian images."""

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

        coordinates = torch.arange(image_size, dtype=torch.float32)  # [H]
        self.rows, self.cols = torch.meshgrid(
            coordinates, coordinates, indexing="ij"
        )  # each [H,W]

    def __len__(self) -> int:
        return int(self.sigma_row.numel())

    def __getitem__(self, index: int) -> torch.Tensor:
        exponent = (
            ((self.rows - self.mean_row) / self.sigma_row[index]).square()
            + ((self.cols - self.mean_col) / self.sigma_col[index]).square()
        )  # [H,W]
        image = torch.exp(-0.5 * exponent)  # [H,W]
        return image.unsqueeze(0)  # [1,H,W]


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images in raster order: [B,1,H,W] -> [B,N,D]."""
    batch, channels, height, width = images.shape
    grid_rows = height // patch_size
    grid_cols = width // patch_size
    x = images.reshape(
        batch, channels, grid_rows, patch_size, grid_cols, patch_size
    )  # [B,1,G,P,G,P]
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B,G,G,1,P,P]
    return x.reshape(
        batch, grid_rows * grid_cols, channels * patch_size * patch_size
    )  # [B,N,D]


def unpatchify(
    patches: torch.Tensor,
    image_size: int,
    patch_size: int,
    channels: int = 1,
) -> torch.Tensor:
    """Convert raster-ordered patches: [B,N,D] -> [B,1,H,W]."""
    batch = patches.shape[0]
    grid = image_size // patch_size
    x = patches.reshape(
        batch, grid, grid, channels, patch_size, patch_size
    )  # [B,G,G,1,P,P]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B,1,G,P,G,P]
    return x.reshape(batch, channels, image_size, image_size)  # [B,1,H,W]


def gather_items(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather sequence values: [B,N,F], [B,L] -> [B,L,F]."""
    feature_dim = values.shape[-1]
    indices = positions.unsqueeze(-1).expand(-1, -1, feature_dim)  # [B,L,F]
    return values.gather(1, indices)  # [B,L,F]


@dataclass
class PatchLayout:
    """Prompt/request layout shared in cardinality but not positions."""

    prompt_positions: torch.Tensor  # [B,M]
    target_positions: torch.Tensor  # [B,K], random AR generation order
    prompt_mask: torch.Tensor  # [B,N]
    request_mask: torch.Tensor  # [B,N], may overlap prompt_mask
    prompt_count: int
    request_count: int
    target_count: int

    def to(self, device: torch.device) -> "PatchLayout":
        return PatchLayout(
            prompt_positions=self.prompt_positions.to(device),
            target_positions=self.target_positions.to(device),
            prompt_mask=self.prompt_mask.to(device),
            request_mask=self.request_mask.to(device),
            prompt_count=self.prompt_count,
            request_count=self.request_count,
            target_count=self.target_count,
        )


def random_integer(low: int, high: int, generator: torch.Generator) -> int:
    """Uniform integer from the inclusive interval [low, high]."""
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
    """Sample arbitrary prompt and request sets.

    Every sample receives independent positions and an independent random
    target order. M and R are shared within a batch so no padding is needed.

    Requested positions are selected from non-prompt positions first. If
    R > N-M, the remaining requested positions overlap the prompt and will be
    copied exactly rather than predicted.
    """
    m = (
        prompt_count
        if prompt_count is not None
        else random_integer(min_prompt, max_prompt, generator)
    )
    r = (
        request_count
        if request_count is not None
        else random_integer(min_request, max_request, generator)
    )
    k = min(r, num_patches - m)
    overlap = r - k

    permutations = torch.rand(
        batch_size, num_patches, generator=generator
    ).argsort(dim=1)  # [B,N]
    prompt_positions = permutations[:, :m]  # [B,M]
    target_positions = permutations[:, m : m + k]  # [B,K]

    prompt_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)  # [B,N]
    request_mask = torch.zeros_like(prompt_mask)  # [B,N]
    prompt_mask.scatter_(1, prompt_positions, True)
    request_mask.scatter_(1, target_positions, True)
    if overlap > 0:
        request_mask.scatter_(1, prompt_positions[:, :overlap], True)

    return PatchLayout(
        prompt_positions=prompt_positions,
        target_positions=target_positions,
        prompt_mask=prompt_mask,
        request_mask=request_mask,
        prompt_count=m,
        request_count=r,
        target_count=k,
    )


class ContinuousTokenizer(nn.Module):
    """Patch autoencoder: [...,D] -> [...,Z] -> [...,D]."""

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
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        leading_shape = patches.shape[:-1]
        flat = patches.reshape(-1, self.patch_dim)  # [prod(leading),D]
        latents = self.encoder(flat)  # [prod(leading),Z]
        return latents.reshape(*leading_shape, self.latent_dim)  # [...,Z]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        leading_shape = latents.shape[:-1]
        flat = latents.reshape(-1, self.latent_dim)  # [prod(leading),Z]
        patches = self.decoder(flat)  # [prod(leading),D]
        return patches.reshape(*leading_shape, self.patch_dim)  # [...,D]

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        latents = self.encode(patches)  # [...,Z]
        return self.decode(latents)  # [...,D]


class FlexibleLatentPatchAR(nn.Module):
    """Flexible-prompt AR transformer that predicts tokenizer latents."""

    def __init__(
        self,
        grid_size: int,
        latent_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.latent_to_model = nn.Linear(latent_dim, d_model)  # Z -> E
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
            encoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.latent_head = nn.Linear(d_model, latent_dim)  # E -> Z
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def spatial_embedding(self, positions: torch.Tensor) -> torch.Tensor:
        rows = positions // self.grid_size  # [B,L]
        cols = positions % self.grid_size  # [B,L]
        row_values = self.row_embedding(rows)  # [B,L,E]
        col_values = self.col_embedding(cols)  # [B,L,E]
        return row_values + col_values  # [B,L,E]

    def role_embedding_value(self, role_id: int, device: torch.device) -> torch.Tensor:
        role_index = torch.tensor(role_id, dtype=torch.long, device=device)  # []
        role = self.role_embedding(role_index)  # [E]
        return role.reshape(1, 1, self.d_model)  # [1,1,E]

    def build_sequence(
        self,
        prompt_latents: torch.Tensor,
        prompt_positions: torch.Tensor,
        target_positions: torch.Tensor,
        previous_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Build [prompts, q1, c1, ..., qK].

        prompt_latents   [B,M,Z]
        prompt_positions [B,M]
        target_positions [B,K]
        previous_latents [B,K-1,Z]
        return sequence  [B,S,E], prompt_count M
        """
        batch, prompt_count, latent_dim = prompt_latents.shape
        target_count = target_positions.shape[1]
        if latent_dim != self.latent_dim:
            raise ValueError("prompt_latents has the wrong latent dimension")
        if previous_latents.shape != (batch, target_count - 1, self.latent_dim):
            raise ValueError("previous_latents must have shape [B,K-1,Z]")

        prompt_content = self.latent_to_model(prompt_latents)  # [B,M,E]
        prompt_tokens = (
            prompt_content
            + self.spatial_embedding(prompt_positions)  # [B,M,E]
            + self.role_embedding_value(PROMPT_ROLE, prompt_positions.device)  # [1,1,E]
        )  # [B,M,E]

        query_tokens = (
            self.spatial_embedding(target_positions)  # [B,K,E]
            + self.role_embedding_value(QUERY_ROLE, target_positions.device)  # [1,1,E]
        )  # [B,K,E]

        if target_count == 1:
            suffix = query_tokens  # [B,1,E]
        else:
            previous_content = self.latent_to_model(previous_latents)  # [B,K-1,E]
            content_tokens = (
                previous_content
                + self.spatial_embedding(target_positions[:, :-1])  # [B,K-1,E]
                + self.role_embedding_value(CONTENT_ROLE, target_positions.device)  # [1,1,E]
            )  # [B,K-1,E]

            pairs = torch.stack(
                (query_tokens[:, :-1], content_tokens), dim=2
            )  # [B,K-1,2,E]
            paired_prefix = pairs.reshape(
                batch, 2 * (target_count - 1), self.d_model
            )  # [B,2K-2,E]
            suffix = torch.cat(
                (paired_prefix, query_tokens[:, -1:]), dim=1
            )  # [B,2K-1,E]

        sequence = torch.cat((prompt_tokens, suffix), dim=1)  # [B,S,E]
        return sequence, prompt_count

    @staticmethod
    def block_causal_mask(
        prompt_count: int,
        total_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create bool attention mask [S,S]; True means forbidden.

        Prompt rows can read prompt columns only. Suffix rows can read every
        prompt column and earlier-or-current suffix columns.
        """
        mask = torch.ones(
            total_length, total_length, dtype=torch.bool, device=device
        )  # [S,S]
        mask[:prompt_count, :prompt_count] = False

        suffix_length = total_length - prompt_count
        suffix_mask = torch.triu(
            torch.ones(
                suffix_length, suffix_length, dtype=torch.bool, device=device
            ),
            diagonal=1,
        )  # [2K-1,2K-1]
        mask[prompt_count:, :prompt_count] = False
        mask[prompt_count:, prompt_count:] = suffix_mask
        return mask  # [S,S]

    def predict_queries(
        self,
        prompt_latents: torch.Tensor,
        prompt_positions: torch.Tensor,
        target_positions: torch.Tensor,
        previous_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Predict all query latents: [B,K,Z]."""
        sequence, prompt_count = self.build_sequence(
            prompt_latents=prompt_latents,
            prompt_positions=prompt_positions,
            target_positions=target_positions,
            previous_latents=previous_latents,
        )  # sequence [B,S,E]
        mask = self.block_causal_mask(
            prompt_count, sequence.shape[1], sequence.device
        )  # [S,S]
        hidden = self.transformer(sequence, mask=mask)  # [B,S,E]
        query_hidden = hidden[:, prompt_count::2]  # [B,K,E]
        return self.latent_head(query_hidden)  # [B,K,Z]

    def forward(
        self,
        prompt_latents: torch.Tensor,
        prompt_positions: torch.Tensor,
        target_latents: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced latent prediction: [B,K,Z]."""
        previous_latents = target_latents[:, :-1]  # [B,K-1,Z]
        return self.predict_queries(
            prompt_latents=prompt_latents,
            prompt_positions=prompt_positions,
            target_positions=target_positions,
            previous_latents=previous_latents,
        )  # [B,K,Z]


class WBLogger:
    """Small W&B wrapper; disabled mode does not require wandb to be installed."""

    def __init__(self, args: argparse.Namespace, config: dict[str, Any]) -> None:
        self.wandb = None
        self.run = None
        if args.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as error:
            raise SystemExit(
                "Install W&B with `pip install wandb`, or pass --wandb-mode disabled"
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
        if self.run is not None:
            self.run.define_metric(*args, **kwargs)

    def log(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.log(values)

    def log_figure(
        self,
        values: dict[str, Any],
        key: str,
        figure: plt.Figure,
    ) -> None:
        if self.run is not None and self.wandb is not None:
            payload = dict(values)
            payload[key] = self.wandb.Image(figure)
            self.run.log(payload)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def tokenizer_comparison_figure(
    true_images: torch.Tensor,
    decoded_images: torch.Tensor,
    title: str,
) -> plt.Figure:
    true = true_images[:NUM_PLOT_EXAMPLES, 0].detach().cpu().numpy()  # [8,H,W]
    decoded = decoded_images[:NUM_PLOT_EXAMPLES, 0].detach().cpu().numpy()  # [8,H,W]
    difference = np.abs(true - decoded)  # [8,H,W]
    rows = (true, decoded, difference)
    labels = ("True image", "Tokenizer decoded", "Absolute difference")

    fig, axes = plt.subplots(
        3,
        NUM_PLOT_EXAMPLES,
        figsize=(16, 6),
        squeeze=False,
    )
    fig.suptitle(title)
    cmaps = ["turbo", "turbo", "coolwarm"]
    for row_index, row_images in enumerate(rows):
        for col_index in range(NUM_PLOT_EXAMPLES):
            vmin = 0.0 if row_index < 2 else None
            vmax = 1.0 if row_index < 2 else None
            axes[row_index, col_index].imshow(
                row_images[col_index], cmap=cmaps[row_index], vmin=vmin, vmax=vmax
            )
            axes[row_index, col_index].set(xticks=[], yticks=[])
            if row_index == 0:
                axes[row_index, col_index].set_title(
                    f"Example {col_index + 1}", fontsize=9
                )
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(labels[row_index])

    fig.subplots_adjust(wspace=0.001, hspace=0.001)
    return fig


def generation_comparison_figure(
    true_requested: torch.Tensor,
    prompt_only: torch.Tensor,
    generated_requested: torch.Tensor,
    title: str,
) -> plt.Figure:
    true = true_requested[:NUM_PLOT_EXAMPLES, 0].detach().cpu().numpy()  # [8,H,W]
    prompt = prompt_only[:NUM_PLOT_EXAMPLES, 0].detach().cpu().numpy()  # [8,H,W]
    generated = generated_requested[:NUM_PLOT_EXAMPLES, 0].detach().cpu().numpy()  # [8,H,W]
    difference = np.abs(true - generated)  # [8,H,W]
    rows = (true, prompt, generated, difference)
    labels = (
        "True requested",
        "Prompt only",
        "Generated requested",
        "Absolute difference",
    )

    fig, axes = plt.subplots(
        4,
        NUM_PLOT_EXAMPLES,
        figsize=(16, 8),
        squeeze=False,
    )
    fig.suptitle(title)
    cmaps = ["turbo", "turbo", "turbo", "coolwarm"]
    for row_index, row_images in enumerate(rows):
        for col_index in range(NUM_PLOT_EXAMPLES):
            vmin = 0.0 if row_index <3 else None
            vmax = 1.0 if row_index <3 else None
            axes[row_index, col_index].imshow(
                row_images[col_index], cmap=cmaps[row_index], vmin=vmin, vmax=vmax
            )
            axes[row_index, col_index].set(xticks=[], yticks=[])
            if row_index == 0:
                axes[row_index, col_index].set_title(
                    f"Example {col_index + 1}", fontsize=9
                )
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(labels[row_index])
    fig.subplots_adjust(wspace=0.001, hspace=0.001)
    return fig


@torch.no_grad()
def reconstruct_images(
    tokenizer: ContinuousTokenizer,
    images: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    patches = patchify(images, patch_size)  # [B,N,D]
    decoded_patches = tokenizer(patches)  # [B,N,D]
    return unpatchify(decoded_patches, image_size, patch_size)  # [B,1,H,W]


@torch.no_grad()
def generate_target_latents(
    model: FlexibleLatentPatchAR,
    all_latents: torch.Tensor,
    prompt_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Autoregressively generate requested non-prompt latents.

    all_latents      [B,N,Z]
    prompt_positions [B,M]
    target_positions [B,K]
    return           [B,K,Z]

    The decoder is never called inside this loop.
    """
    prompt_latents = gather_items(all_latents, prompt_positions)  # [B,M,Z]
    generated: list[torch.Tensor] = []

    for step in range(target_positions.shape[1]):
        positions_so_far = target_positions[:, : step + 1]  # [B,step+1]
        if generated:
            previous_latents = torch.stack(generated, dim=1)  # [B,step,Z]
        else:
            previous_latents = all_latents.new_empty(
                all_latents.shape[0], 0, all_latents.shape[-1]
            )  # [B,0,Z]

        predictions = model.predict_queries(
            prompt_latents=prompt_latents,
            prompt_positions=prompt_positions,
            target_positions=positions_so_far,
            previous_latents=previous_latents,
        )  # [B,step+1,Z]
        generated.append(predictions[:, -1])  # [B,Z]

    return torch.stack(generated, dim=1)  # [B,K,Z]


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
    model: FlexibleLatentPatchAR,
    tokenizer: ContinuousTokenizer,
    images: torch.Tensor,
    layout: PatchLayout,
    image_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the three image tensors used by the four-row AR figure."""
    patches = patchify(images, patch_size)  # [8,N,D]
    all_latents = tokenizer.encode(patches)  # [8,N,Z]

    generated_latents = generate_target_latents(
        model=model,
        all_latents=all_latents,
        prompt_positions=layout.prompt_positions,
        target_positions=layout.target_positions,
    )  # [8,K,Z]

    # Decode all generated latents in one batched operation after AR generation.
    generated_patches = tokenizer.decode(generated_latents)  # [8,K,D]

    true_requested = masked_images(
        patches, layout.request_mask, image_size, patch_size
    )  # [8,1,H,W]
    prompt_only = masked_images(
        patches, layout.prompt_mask, image_size, patch_size
    )  # [8,1,H,W]

    canvas = torch.zeros_like(patches)  # [8,N,D]
    overlap = layout.request_mask & layout.prompt_mask  # [8,N]
    canvas = torch.where(overlap.unsqueeze(-1), patches, canvas)  # [8,N,D]

    scatter_indices = layout.target_positions.unsqueeze(-1).expand(
        -1, -1, patches.shape[-1]
    )  # [8,K,D]
    canvas.scatter_(1, scatter_indices, generated_patches)  # [8,N,D]
    generated_requested = unpatchify(
        canvas, image_size, patch_size
    )  # [8,1,H,W]
    return true_requested, prompt_only, generated_requested


@torch.no_grad()
def tokenizer_validation_loss(
    tokenizer: ContinuousTokenizer,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
) -> float:
    tokenizer.eval()
    squared_error = 0.0
    element_count = 0
    for images in loader:
        images = images.to(device)  # [B,1,H,W]
        patches = patchify(images, patch_size)  # [B,N,D]
        decoded = tokenizer(patches)  # [B,N,D]
        squared_error += F.mse_loss(decoded, patches, reduction="sum").item()
        element_count += patches.numel()
    return squared_error / element_count


@dataclass
class ARMetrics:
    total: float
    latent: float
    pixel: float


def combine_ar_losses(
    latent_loss: torch.Tensor,
    pixel_loss: torch.Tensor | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return the selected scalar AR training objective."""
    if args.ar_loss == "latent":
        return latent_loss
    if pixel_loss is None:
        raise ValueError("decoder-aware loss requires pixel_loss")
    return args.lambda_z * latent_loss + args.lambda_x * pixel_loss


@torch.no_grad()
def autoregressive_validation_metrics(
    model: FlexibleLatentPatchAR,
    tokenizer: ContinuousTokenizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> ARMetrics:
    """Teacher-forced validation with deterministic random layouts."""
    model.eval()
    tokenizer.eval()
    generator = torch.Generator().manual_seed(args.seed + 40_000)

    latent_sse = 0.0
    latent_count = 0
    pixel_sse = 0.0
    pixel_count = 0

    for images in loader:
        images = images.to(device)  # [B,1,H,W]
        layout = sample_layout(
            batch_size=images.shape[0],
            num_patches=model.num_patches,
            generator=generator,
            min_prompt=args.min_prompt_patches,
            max_prompt=args.max_prompt_patches,
            min_request=args.min_request_patches,
            max_request=args.max_request_patches,
        ).to(device)

        patches = patchify(images, args.patch_size)  # [B,N,D]
        all_latents = tokenizer.encode(patches)  # [B,N,Z]
        prompt_latents = gather_items(
            all_latents, layout.prompt_positions
        )  # [B,M,Z]
        target_latents = gather_items(
            all_latents, layout.target_positions
        )  # [B,K,Z]
        target_patches = gather_items(
            patches, layout.target_positions
        )  # [B,K,D]

        predicted_latents = model(
            prompt_latents,
            layout.prompt_positions,
            target_latents,
            layout.target_positions,
        )  # [B,K,Z]
        predicted_patches = tokenizer.decode(predicted_latents)  # [B,K,D]

        latent_sse += F.mse_loss(
            predicted_latents, target_latents, reduction="sum"
        ).item()
        latent_count += target_latents.numel()
        pixel_sse += F.mse_loss(
            predicted_patches, target_patches, reduction="sum"
        ).item()
        pixel_count += target_patches.numel()

    latent_mse = latent_sse / latent_count
    pixel_mse = pixel_sse / pixel_count
    total = (
        latent_mse
        if args.ar_loss == "latent"
        else args.lambda_z * latent_mse + args.lambda_x * pixel_mse
    )
    return ARMetrics(total=total, latent=latent_mse, pixel=pixel_mse)


def train_tokenizer(
    tokenizer: ContinuousTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    plot_examples: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    logger: WBLogger,
) -> None:
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=args.tokenizer_lr,
        weight_decay=args.weight_decay,
    )
    batch_step = 0

    for epoch in range(1, args.tokenizer_epochs + 1):
        tokenizer.train()
        epoch_sse = 0.0
        epoch_count = 0

        for images in train_loader:
            images = images.to(device)  # [B,1,H,W]
            patches = patchify(images, args.patch_size)  # [B,N,D]

            optimizer.zero_grad(set_to_none=True)
            decoded = tokenizer(patches)  # [B,N,D]
            loss = F.mse_loss(decoded, patches)  # scalar
            loss.backward()
            optimizer.step()

            epoch_sse += F.mse_loss(
                decoded.detach(), patches, reduction="sum"
            ).item()
            epoch_count += patches.numel()
            logger.log(
                {
                    "tokenizer/batch_step": batch_step,
                    "tokenizer/train_loss": loss.item(),
                }
            )
            batch_step += 1

        train_loss = epoch_sse / epoch_count
        validation_loss = tokenizer_validation_loss(
            tokenizer, val_loader, device, args.patch_size
        )

        fixed = plot_examples.to(device)  # [8,1,H,W]
        decoded_images = reconstruct_images(
            tokenizer, fixed, args.image_size, args.patch_size
        )  # [8,1,H,W]
        figure = tokenizer_comparison_figure(
            fixed,
            decoded_images,
            title=f"Tokenizer epoch {epoch}",
        )
        logger.log_figure(
            {
                "tokenizer/epoch": epoch,
                "tokenizer/epoch_train_loss": train_loss,
                "tokenizer/validation_loss": validation_loss,
            },
            key="tokenizer/comparison",
            figure=figure,
        )
        plt.close(figure)

        print(
            f"[tokenizer] {epoch:03d}/{args.tokenizer_epochs:03d} "
            f"train={train_loss:.6f} val={validation_loss:.6f}"
        )


def train_autoregressor(
    model: FlexibleLatentPatchAR,
    tokenizer: ContinuousTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    plot_examples: torch.Tensor,
    plot_layout: PatchLayout,
    args: argparse.Namespace,
    device: torch.device,
    logger: WBLogger,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.ar_lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed + 30_000)
    batch_step = 0

    tokenizer.eval()
    tokenizer.requires_grad_(False)

    for epoch in range(1, args.ar_epochs + 1):
        model.train()
        latent_sse = 0.0
        latent_count = 0
        pixel_sse = 0.0
        pixel_count = 0

        for images in train_loader:
            images = images.to(device)  # [B,1,H,W]
            layout = sample_layout(
                batch_size=images.shape[0],
                num_patches=model.num_patches,
                generator=generator,
                min_prompt=args.min_prompt_patches,
                max_prompt=args.max_prompt_patches,
                min_request=args.min_request_patches,
                max_request=args.max_request_patches,
            ).to(device)

            patches = patchify(images, args.patch_size)  # [B,N,D]
            with torch.no_grad():
                all_latents = tokenizer.encode(patches)  # [B,N,Z]

            prompt_latents = gather_items(
                all_latents, layout.prompt_positions
            )  # [B,M,Z]
            target_latents = gather_items(
                all_latents, layout.target_positions
            )  # [B,K,Z]
            target_patches = gather_items(
                patches, layout.target_positions
            )  # [B,K,D]

            optimizer.zero_grad(set_to_none=True)
            predicted_latents = model(
                prompt_latents,
                layout.prompt_positions,
                target_latents,
                layout.target_positions,
            )  # [B,K,Z]
            latent_loss = F.mse_loss(predicted_latents, target_latents)  # scalar

            pixel_loss: torch.Tensor | None = None
            if args.ar_loss == "decoder-aware":
                # Frozen decoder: [B,K,Z] -> [B,K,D]. Gradients still flow
                # from this loss through predicted_latents to the transformer.
                predicted_patches = tokenizer.decode(predicted_latents)  # [B,K,D]
                pixel_loss = F.mse_loss(predicted_patches, target_patches)  # scalar

            total_loss = combine_ar_losses(latent_loss, pixel_loss, args)  # scalar
            total_loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            latent_sse += F.mse_loss(
                predicted_latents.detach(), target_latents, reduction="sum"
            ).item()
            latent_count += target_latents.numel()
            if pixel_loss is not None:
                pixel_sse += F.mse_loss(
                    predicted_patches.detach(), target_patches, reduction="sum"
                ).item()
                pixel_count += target_patches.numel()

            log_values: dict[str, Any] = {
                "autoregressive/batch_step": batch_step,
                "autoregressive/train_loss": total_loss.item(),
                "autoregressive/train_latent_loss": latent_loss.item(),
                "autoregressive/prompt_count": layout.prompt_count,
                "autoregressive/request_count": layout.request_count,
                "autoregressive/target_count": layout.target_count,
            }
            if pixel_loss is not None:
                log_values["autoregressive/train_pixel_loss"] = pixel_loss.item()
            logger.log(log_values)
            batch_step += 1

        train_latent = latent_sse / latent_count
        if args.ar_loss == "decoder-aware":
            train_pixel = pixel_sse / pixel_count
            train_total = args.lambda_z * train_latent + args.lambda_x * train_pixel
        else:
            train_pixel = float("nan")
            train_total = train_latent

        validation = autoregressive_validation_metrics(
            model, tokenizer, val_loader, device, args
        )

        fixed = plot_examples.to(device)  # [8,1,H,W]
        layout = plot_layout.to(device)
        true_requested, prompt_only, generated_requested = generation_figure_inputs(
            model=model,
            tokenizer=tokenizer,
            images=fixed,
            layout=layout,
            image_size=args.image_size,
            patch_size=args.patch_size,
        )  # each [8,1,H,W]

        title = (
            f"Latent AR epoch {epoch}; loss={args.ar_loss}; "
            f"prompt={layout.prompt_count}, requested={layout.request_count}, "
            f"generated={layout.target_count}"
        )
        figure = generation_comparison_figure(
            true_requested=true_requested,
            prompt_only=prompt_only,
            generated_requested=generated_requested,
            title=title,
        )

        epoch_values: dict[str, Any] = {
            "autoregressive/epoch": epoch,
            "autoregressive/epoch_train_loss": train_total,
            "autoregressive/epoch_train_latent_loss": train_latent,
            "autoregressive/validation_loss": validation.total,
            "autoregressive/validation_latent_loss": validation.latent,
            "autoregressive/validation_pixel_loss": validation.pixel,
        }
        if args.ar_loss == "decoder-aware":
            epoch_values["autoregressive/epoch_train_pixel_loss"] = train_pixel

        logger.log_figure(
            epoch_values,
            key="autoregressive/comparison",
            figure=figure,
        )
        plt.close(figure)

        pixel_text = (
            f" train_pixel={train_pixel:.6f}"
            if args.ar_loss == "decoder-aware"
            else ""
        )
        print(
            f"[autoregressive] {epoch:03d}/{args.ar_epochs:03d} "
            f"train={train_total:.6f} train_latent={train_latent:.6f}"
            f"{pixel_text} val={validation.total:.6f} "
            f"val_latent={validation.latent:.6f} "
            f"val_pixel={validation.pixel:.6f}"
        )


def fixed_examples(dataset: Dataset) -> torch.Tensor:
    examples = [dataset[index] for index in range(NUM_PLOT_EXAMPLES)]
    return torch.stack(examples, dim=0)  # [8,1,H,W]


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
    logger.define_metric(
        "tokenizer/train_loss", step_metric="tokenizer/batch_step"
    )
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
        "autoregressive/train_latent_loss",
        "autoregressive/train_pixel_loss",
        "autoregressive/prompt_count",
        "autoregressive/request_count",
        "autoregressive/target_count",
    ):
        logger.define_metric(key, step_metric="autoregressive/batch_step")

    logger.define_metric("autoregressive/epoch")
    for key in (
        "autoregressive/epoch_train_loss",
        "autoregressive/epoch_train_latent_loss",
        "autoregressive/epoch_train_pixel_loss",
        "autoregressive/validation_loss",
        "autoregressive/validation_latent_loss",
        "autoregressive/validation_pixel_loss",
        "autoregressive/comparison",
    ):
        logger.define_metric(key, step_metric="autoregressive/epoch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flexible-prompt latent-space autoregressive Gaussian patch demo"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=20.0)
    parser.add_argument("--sigma-max", type=float, default=40.0)
    parser.add_argument("--mean-row", type=float, default=64.0)
    parser.add_argument("--mean-col", type=float, default=64.0)
    parser.add_argument("--dataset-type", type=str, default="deterministic", choices=["deterministic", "nondeterministic"])

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tokenizer-epochs", type=int, default=20)
    parser.add_argument("--ar-epochs", type=int, default=40)
    parser.add_argument("--tokenizer-lr", type=float, default=3e-4)
    parser.add_argument("--ar-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--tokenizer-hidden-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--ar-loss",
        choices=("latent", "decoder-aware"),
        default="latent",
        help=(
            "latent: L_latent; decoder-aware: "
            "lambda_z*L_latent + lambda_x*L_pixel"
        ),
    )
    parser.add_argument("--lambda-z", type=float, default=1.0)
    parser.add_argument("--lambda-x", type=float, default=1.0)

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
        "--output-dir",
        type=Path,
        default=Path("outputs/flexible_gaussian_latent_ar"),
    )

    parser.add_argument("--wandb-project", default="flexible-latent-patch-ar-demo")
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
    if args.train_size < NUM_PLOT_EXAMPLES or args.val_size < NUM_PLOT_EXAMPLES:
        raise ValueError("train-size and val-size must both be at least 8")
    if not 0.0 < args.sigma_min < args.sigma_max:
        raise ValueError("Require 0 < sigma-min < sigma-max")
    if args.d_model % args.num_heads != 0:
        raise ValueError("d-model must be divisible by num-heads")
    if args.latent_dim < 1 or args.tokenizer_hidden_dim < 1 or args.d_model < 1:
        raise ValueError("latent and model dimensions must be positive")
    if args.num_heads < 1 or args.num_layers < 1 or args.ffn_dim < 1:
        raise ValueError("transformer dimensions and layer counts must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    if args.batch_size < 1 or args.tokenizer_epochs < 1 or args.ar_epochs < 1:
        raise ValueError("batch size and epoch counts must be positive")
    if args.cpu_threads < 1 or args.num_workers < 0:
        raise ValueError("cpu-threads must be positive and num-workers nonnegative")
    if args.gradient_clip < 0.0:
        raise ValueError("gradient-clip must be nonnegative")
    if args.lambda_z < 0.0 or args.lambda_x < 0.0:
        raise ValueError("lambda-z and lambda-x must be nonnegative")
    if args.ar_loss == "decoder-aware" and args.lambda_z + args.lambda_x <= 0.0:
        raise ValueError("decoder-aware loss requires lambda-z + lambda-x > 0")

    num_patches = (args.image_size // args.patch_size) ** 2
    args.max_prompt_patches = args.max_prompt_patches or num_patches - 1
    args.max_request_patches = args.max_request_patches or num_patches

    if not (
        1
        <= args.min_prompt_patches
        <= args.max_prompt_patches
        <= num_patches - 1
    ):
        raise ValueError("Prompt range must satisfy 1 <= min <= max <= N-1")
    if not (
        1
        <= args.min_request_patches
        <= args.max_request_patches
        <= num_patches
    ):
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
    print(
        f"Autoregressive objective: {args.ar_loss}"
        + (
            f" (lambda_z={args.lambda_z}, lambda_x={args.lambda_x})"
            if args.ar_loss == "decoder-aware"
            else ""
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = dict(
        image_size=args.image_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        mean_row=args.mean_row,
        mean_col=args.mean_col,
    )

    
    DatasetClass = GaussianDataset if if args.dataset_type == "deterministic" else GaussianDatasetNondeterministic   

    train_set = DatasetClass(
        size=args.train_size, seed=args.seed, **dataset_kwargs
    )
    val_set = DatasetClass(
        size=args.val_size, seed=args.seed + 1, **dataset_kwargs
    )
    train_loader = make_loader(
        train_set, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, args.num_workers, device
    )

    grid_size = args.image_size // args.patch_size
    num_patches = grid_size * grid_size
    patch_dim = args.patch_size * args.patch_size

    plot_examples = fixed_examples(val_set)  # [8,1,H,W]
    plot_layout = sample_layout(
        batch_size=NUM_PLOT_EXAMPLES,
        num_patches=num_patches,
        generator=torch.Generator().manual_seed(args.seed + 50_000),
        min_prompt=args.min_prompt_patches,
        max_prompt=args.max_prompt_patches,
        min_request=args.min_request_patches,
        max_request=args.max_request_patches,
        prompt_count=args.plot_prompt_patches,
        request_count=args.plot_request_patches,
    )

    tokenizer = ContinuousTokenizer(
        patch_dim=patch_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.tokenizer_hidden_dim,
    ).to(device)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        grid_size=grid_size,
        num_patches=num_patches,
        patch_dim=patch_dim,
    )

    logger = WBLogger(args, config)
    try:
        define_metrics(logger)

        train_tokenizer(
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            plot_examples=plot_examples,
            args=args,
            device=device,
            logger=logger,
        )
        tokenizer.requires_grad_(False)
        tokenizer.eval()
        torch.save(
            {"state_dict": tokenizer.state_dict(), "config": config},
            args.output_dir / "continuous_tokenizer.pt",
        )

        autoregressor = FlexibleLatentPatchAR(
            grid_size=grid_size,
            latent_dim=args.latent_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        ).to(device)

        train_autoregressor(
            model=autoregressor,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            plot_examples=plot_examples,
            plot_layout=plot_layout,
            args=args,
            device=device,
            logger=logger,
        )
        torch.save(
            {"state_dict": autoregressor.state_dict(), "config": config},
            args.output_dir / "flexible_latent_autoregressor.pt",
        )
    finally:
        logger.finish()

    print(f"Saved checkpoints in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
