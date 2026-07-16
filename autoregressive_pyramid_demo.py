#!/usr/bin/env python3
"""Minimal resolution-aware autoregressive image-pyramid demo.

The script trains two stages on synthetic 128x128 grayscale Gaussian images:

1. A scale-conditioned Transformer patch autoencoder. Each 16x16 image patch is
   split into small, non-overlapping pixel tiles; a linear projection turns the
   tiles into tokens. There are no convolutional layers.
2. A block-causal Transformer that predicts patch latents autoregressively over
   a resolution pyramid. Missing ancestors of requested patches are generated
   first as latent-only support nodes.

The image pyramid is constructed recursively with exact average pooling. Thus,
for a 2x refinement, every parent pixel is exactly the arithmetic mean of the
corresponding four child pixels.

Dependencies:
    pip install torch matplotlib wandb

Example:
    python demo_autoregressive_pyramid.py --wandb-mode offline

The default grid schedule is 2,4,8 with 16x16 patches, corresponding to image
resolutions 32x32, 64x64, and 128x128.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except ImportError:  # A clear error is raised in main().
    wandb = None


# -----------------------------------------------------------------------------
# Configuration and reproducibility
# -----------------------------------------------------------------------------


def parse_grid_sizes(text: str) -> tuple[int, ...]:
    try:
        grids = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid sizes must be comma-separated integers") from exc

    if not grids:
        raise argparse.ArgumentTypeError("at least one grid size is required")
    if any(grid <= 0 for grid in grids):
        raise argparse.ArgumentTypeError("grid sizes must be positive")
    if any(coarse >= fine for coarse, fine in zip(grids[:-1], grids[1:])):
        raise argparse.ArgumentTypeError("grid sizes must be strictly increasing")
    if any(fine % coarse != 0 for coarse, fine in zip(grids[:-1], grids[1:])):
        raise argparse.ArgumentTypeError(
            "every finer grid must be an integer multiple of the previous grid"
        )
    return grids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transformer-tokenizer autoregressive mean-pyramid demo"
    )

    # Data and pyramid.
    parser.add_argument("--grid-sizes", type=parse_grid_sizes, default=parse_grid_sizes("2,4,8"))
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--sigma-min", type=float, default=10.0)
    parser.add_argument("--sigma-max", type=float, default=20.0)
    parser.add_argument("--mean-row", type=float, default=64.0)
    parser.add_argument("--mean-col", type=float, default=64.0)

    # Optimization.
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tokenizer-epochs", type=int, default=10)
    parser.add_argument("--ar-epochs", type=int, default=20)
    parser.add_argument("--tokenizer-lr", type=float, default=3e-4)
    parser.add_argument("--ar-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # Transformer patch tokenizer. token-size=4 gives 4x4=16 tokens per 16x16 patch.
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--tokenizer-token-size", type=int, default=4)
    parser.add_argument("--tokenizer-d-model", type=int, default=64)
    parser.add_argument("--tokenizer-heads", type=int, default=4)
    parser.add_argument("--tokenizer-encoder-layers", type=int, default=2)
    parser.add_argument("--tokenizer-decoder-layers", type=int, default=2)
    parser.add_argument("--tokenizer-ffn-dim", type=int, default=128)
    parser.add_argument("--tokenizer-dropout", type=float, default=0.0)
    parser.add_argument("--tokenizer-mean-weight", type=float, default=0.1)

    # Autoregressive Transformer.
    parser.add_argument("--ar-d-model", type=int, default=128)
    parser.add_argument("--ar-heads", type=int, default=4)
    parser.add_argument("--ar-layers", type=int, default=4)
    parser.add_argument("--ar-ffn-dim", type=int, default=512)
    parser.add_argument("--ar-dropout", type=float, default=0.1)
    parser.add_argument("--fourier-frequencies", type=int, default=6)
    parser.add_argument("--ar-loss", choices=("latent", "decoder-aware"), default="decoder-aware")
    parser.add_argument("--lambda-z", type=float, default=1.0)
    parser.add_argument("--lambda-x", type=float, default=1.0)
    parser.add_argument("--ar-mean-weight", type=float, default=0.1)

    # Random training layouts.
    parser.add_argument("--min-prompt-patches", type=int, default=1)
    parser.add_argument("--max-prompt-patches", type=int, default=8)
    parser.add_argument("--min-request-patches", type=int, default=1)
    parser.add_argument("--max-request-patches", type=int, default=12)
    parser.add_argument(
        "--full-request-probability",
        type=float,
        default=0.05,
        help="Probability of requesting an entire sampled resolution during AR training",
    )

    # Fixed validation figures.
    parser.add_argument("--plot-request-order", type=int, default=3)
    parser.add_argument("--plot-prompt-patches", type=int, default=8)
    parser.add_argument("--plot-request-patches", type=int, default=24)

    # Runtime and logging.
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--torch-threads", type=int, default=4,
        help="CPU intra-op thread count; small Transformer batches often run faster with a modest value",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("pyramid_demo_outputs"))
    parser.add_argument("--wandb-project", type=str, default="autoregressive-pyramid-demo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Use 'online' after authenticating with W&B; 'offline' works without network sync",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if args.patch_size % args.tokenizer_token_size != 0:
        raise ValueError("--tokenizer-token-size must divide --patch-size")
    if args.tokenizer_d_model % args.tokenizer_heads != 0:
        raise ValueError("--tokenizer-d-model must be divisible by --tokenizer-heads")
    if args.ar_d_model % args.ar_heads != 0:
        raise ValueError("--ar-d-model must be divisible by --ar-heads")
    if args.val_size < 8:
        raise ValueError("--val-size must be at least 8 for the requested figures")
    if args.sigma_min <= 0 or args.sigma_max < args.sigma_min:
        raise ValueError("sigma range must satisfy 0 < sigma-min <= sigma-max")
    if args.min_prompt_patches < 1 or args.max_prompt_patches < args.min_prompt_patches:
        raise ValueError("invalid prompt-patch count range")
    if args.min_request_patches < 1 or args.max_request_patches < args.min_request_patches:
        raise ValueError("invalid request-patch count range")
    if not 0.0 <= args.full_request_probability <= 1.0:
        raise ValueError("--full-request-probability must be in [0, 1]")
    if not 1 <= args.plot_request_order <= len(args.grid_sizes):
        raise ValueError("--plot-request-order must index one of --grid-sizes, starting at 1")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive")
    if args.tokenizer_epochs < 1 or args.ar_epochs < 1:
        raise ValueError("both training stages need at least one epoch")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


# -----------------------------------------------------------------------------
# Synthetic Gaussian-image dataset
# -----------------------------------------------------------------------------


class GaussianImageDataset(Dataset[torch.Tensor]):
    """Deterministic images of one axis-aligned 2D Gaussian.

    sigma_row and sigma_col are independently sampled once when the dataset is
    created. They are used only to render the image and are never returned to a
    model.
    """

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
        super().__init__()
        self.size = size
        self.image_size = image_size
        self.mean_row = mean_row
        self.mean_col = mean_col

        generator = torch.Generator().manual_seed(seed)
        self.sigma_row = sigma_min + (sigma_max - sigma_min) * torch.rand(
            size, generator=generator
        )
        self.sigma_col = sigma_min + (sigma_max - sigma_min) * torch.rand(
            size, generator=generator
        )

        coordinates = torch.arange(image_size, dtype=torch.float32)
        self.rows = coordinates[:, None]
        self.cols = coordinates[None, :]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        sigma_row = self.sigma_row[index]
        sigma_col = self.sigma_col[index]
        exponent = -0.5 * (
            ((self.rows - self.mean_row) / sigma_row).square()
            + ((self.cols - self.mean_col) / sigma_col).square()
        )
        return torch.exp(exponent).unsqueeze(0)


# -----------------------------------------------------------------------------
# Mean pyramid and patch operations
# -----------------------------------------------------------------------------


def build_mean_pyramid(
    finest_images: torch.Tensor,
    grid_sizes: Sequence[int],
    patch_size: int,
) -> dict[int, torch.Tensor]:
    """Build a finest-to-coarsest pyramid using exact arithmetic means.

    If adjacent grid sizes differ by factor k, each coarse pixel is the mean of
    the corresponding k x k fine pixels. For the default 2,4,8 pyramid, k=2 and
    each parent pixel is the mean of four child pixels.
    """

    expected_size = grid_sizes[-1] * patch_size
    if finest_images.shape[-2:] != (expected_size, expected_size):
        raise ValueError(
            f"finest image must be {expected_size}x{expected_size}, got "
            f"{tuple(finest_images.shape[-2:])}"
        )

    pyramid: dict[int, torch.Tensor] = {grid_sizes[-1]: finest_images}
    for coarse_grid, fine_grid in reversed(list(zip(grid_sizes[:-1], grid_sizes[1:]))):
        factor = fine_grid // coarse_grid
        pyramid[coarse_grid] = F.avg_pool2d(
            pyramid[fine_grid], kernel_size=factor, stride=factor
        )
    return pyramid


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B,C,H,W] -> [B,N,C,P,P] in raster order."""

    batch, channels, height, width = images.shape
    if height != width or height % patch_size != 0:
        raise ValueError("images must be square and divisible by patch_size")
    grid = height // patch_size
    return (
        images.reshape(batch, channels, grid, patch_size, grid, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid * grid, channels, patch_size, patch_size)
    )


def unpatchify(patches: torch.Tensor, grid: int, patch_size: int) -> torch.Tensor:
    """[B,N,C,P,P] -> [B,C,H,W] in raster order."""

    batch, count, channels, patch_height, patch_width = patches.shape
    if count != grid * grid or patch_height != patch_size or patch_width != patch_size:
        raise ValueError("patch tensor shape does not match grid and patch size")
    return (
        patches.reshape(batch, grid, grid, channels, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, grid * patch_size, grid * patch_size)
    )


# -----------------------------------------------------------------------------
# Resolution-aware pyramid nodes and layouts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PyramidNode:
    node_id: int
    level: int
    grid: int
    row: int
    col: int
    local_index: int


@dataclass
class Layout:
    prompt_ids: list[int]
    request_ids: list[int]
    target_ids: list[int]
    prompt_level: int
    request_level: int


class PyramidSpec:
    """Runtime description of all patch nodes in a square grid pyramid."""

    def __init__(self, grid_sizes: Sequence[int]) -> None:
        self.grid_sizes = tuple(grid_sizes)
        self.nodes: list[PyramidNode] = []
        self.ids_by_level: list[list[int]] = []
        self.key_to_id: dict[tuple[int, int, int], int] = {}
        features: list[list[float]] = []

        next_id = 0
        for level, grid in enumerate(self.grid_sizes):
            level_ids: list[int] = []
            for row in range(grid):
                for col in range(grid):
                    local_index = row * grid + col
                    node = PyramidNode(next_id, level, grid, row, col, local_index)
                    self.nodes.append(node)
                    self.key_to_id[(level, row, col)] = next_id
                    level_ids.append(next_id)

                    # Normalized global footprint. Width and height encode scale.
                    features.append(
                        [
                            (col + 0.5) / grid,
                            (row + 0.5) / grid,
                            1.0 / grid,
                            1.0 / grid,
                        ]
                    )
                    next_id += 1
            self.ids_by_level.append(level_ids)

        self.features = torch.tensor(features, dtype=torch.float32)

    def node_id(self, level: int, row: int, col: int) -> int:
        return self.key_to_id[(level, row, col)]

    def parent(self, node_id: int) -> int | None:
        node = self.nodes[node_id]
        if node.level == 0:
            return None
        coarse_grid = self.grid_sizes[node.level - 1]
        factor = node.grid // coarse_grid
        return self.node_id(node.level - 1, node.row // factor, node.col // factor)

    def ancestors(self, node_id: int) -> list[int]:
        result: list[int] = []
        current = self.parent(node_id)
        while current is not None:
            result.append(current)
            current = self.parent(current)
        return result

    def children(self, node_id: int) -> list[int]:
        node = self.nodes[node_id]
        if node.level + 1 >= len(self.grid_sizes):
            return []
        fine_grid = self.grid_sizes[node.level + 1]
        factor = fine_grid // node.grid
        return [
            self.node_id(node.level + 1, node.row * factor + row_offset, node.col * factor + col_offset)
            for row_offset in range(factor)
            for col_offset in range(factor)
        ]

    def generation_order(
        self,
        request_ids: Sequence[int],
        prompt_ids: Sequence[int],
        rng: random.Random | None,
    ) -> list[int]:
        """Return missing ancestors and requests in coarse-to-fine order."""

        prompt_set = set(prompt_ids)
        unknown_requests = set(request_ids) - prompt_set
        closure: set[int] = set()
        for node_id in unknown_requests:
            closure.add(node_id)
            closure.update(self.ancestors(node_id))
        closure.difference_update(prompt_set)

        ordered: list[int] = []
        for level in range(len(self.grid_sizes)):
            level_nodes = [node_id for node_id in closure if self.nodes[node_id].level == level]
            if rng is None:
                level_nodes.sort(key=lambda item: self.nodes[item].local_index)
            else:
                rng.shuffle(level_nodes)
            ordered.extend(level_nodes)
        return ordered


def _sample_count(rng: random.Random, minimum: int, maximum: int, available: int) -> int:
    upper = min(maximum, available)
    lower = min(minimum, upper)
    return rng.randint(lower, upper)


def sample_layout(
    spec: PyramidSpec,
    rng: random.Random,
    min_prompt: int,
    max_prompt: int,
    min_request: int,
    max_request: int,
    full_request_probability: float,
    prompt_level: int | None = None,
    request_level: int | None = None,
) -> Layout:
    """Sample one layout shared by all examples in a minibatch."""

    if prompt_level is None:
        prompt_level = rng.randrange(len(spec.grid_sizes))
    if request_level is None:
        request_level = rng.randrange(len(spec.grid_sizes))

    prompt_candidates = list(spec.ids_by_level[prompt_level])
    request_candidates = list(spec.ids_by_level[request_level])

    # Keep at least one possible unknown request if prompt and request share a level.
    prompt_available = len(prompt_candidates)
    if prompt_level == request_level:
        prompt_available = max(1, prompt_available - 1)
    prompt_count = _sample_count(rng, min_prompt, max_prompt, prompt_available)
    prompt_ids = rng.sample(prompt_candidates, prompt_count)

    if rng.random() < full_request_probability:
        request_ids = request_candidates
    else:
        request_count = _sample_count(rng, min_request, max_request, len(request_candidates))
        request_ids = rng.sample(request_candidates, request_count)

    if set(request_ids).issubset(set(prompt_ids)):
        non_prompt = list(set(request_candidates) - set(prompt_ids))
        if not non_prompt:
            raise RuntimeError("could not sample an unknown request patch")
        request_ids[0] = rng.choice(non_prompt)

    target_ids = spec.generation_order(request_ids, prompt_ids, rng=rng)
    if not target_ids:
        raise RuntimeError("sampled layout contains no autoregressive targets")

    return Layout(prompt_ids, request_ids, target_ids, prompt_level, request_level)


def make_plot_layout(
    spec: PyramidSpec,
    request_level: int,
    prompt_count: int,
    request_count: int,
    seed: int,
) -> Layout:
    """Fixed same-resolution, disjoint prompt/request layout for figures."""

    rng = random.Random(seed)
    candidates = list(spec.ids_by_level[request_level])
    rng.shuffle(candidates)

    prompt_count = min(prompt_count, max(1, len(candidates) - 1))
    remaining = len(candidates) - prompt_count
    request_count = min(request_count, remaining)
    if request_count < 1:
        raise ValueError("plot layout needs at least one request patch")

    prompt_ids = candidates[:prompt_count]
    request_ids = candidates[prompt_count : prompt_count + request_count]
    target_ids = spec.generation_order(request_ids, prompt_ids, rng=None)
    return Layout(prompt_ids, request_ids, target_ids, request_level, request_level)


# -----------------------------------------------------------------------------
# Transformer patch autoencoder: no convolutions
# -----------------------------------------------------------------------------


class ScaleEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        grids = grids.to(dtype=torch.float32)
        features = torch.stack((torch.log2(grids), grids.reciprocal()), dim=-1)
        return self.network(features)


class TransformerPatchAutoencoder(nn.Module):
    """Scale-conditioned Transformer autoencoder for one fixed-size image patch."""

    def __init__(
        self,
        patch_size: int,
        token_size: int,
        channels: int,
        latent_dim: int,
        d_model: int,
        num_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_size % token_size != 0:
            raise ValueError("token_size must divide patch_size")

        self.patch_size = patch_size
        self.token_size = token_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.tokens_per_side = patch_size // token_size
        self.num_tokens = self.tokens_per_side**2
        self.token_values = channels * token_size * token_size

        self.input_projection = nn.Linear(self.token_values, d_model)
        self.local_positions = nn.Parameter(torch.randn(self.num_tokens, d_model) * 0.02)
        self.encoder_summary_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.scale_embedding = ScaleEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_layers, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

        self.latent_to_memory = nn.Linear(latent_dim, d_model)
        self.decoder_queries = nn.Parameter(torch.randn(self.num_tokens, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, self.token_values)

    def _patch_to_tokens(self, patches: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = patches.shape
        if channels != self.channels or height != self.patch_size or width != self.patch_size:
            raise ValueError("unexpected patch shape")
        side = self.tokens_per_side
        token = self.token_size
        return (
            patches.reshape(batch, channels, side, token, side, token)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, self.num_tokens, self.token_values)
        )

    def _tokens_to_patch(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        side = self.tokens_per_side
        token = self.token_size
        return (
            tokens.reshape(batch, side, side, self.channels, token, token)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, self.channels, self.patch_size, self.patch_size)
        )

    @staticmethod
    def _grid_tensor(
        grids: int | float | torch.Tensor,
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(grids, torch.Tensor):
            result = grids.to(device=device, dtype=torch.float32).reshape(-1)
            if result.numel() == 1:
                result = result.expand(count)
            if result.numel() != count:
                raise ValueError("grid tensor must contain one value per patch")
            return result
        return torch.full((count,), float(grids), device=device)

    def encode(
        self,
        patches: torch.Tensor,
        grids: int | float | torch.Tensor,
    ) -> torch.Tensor:
        count = patches.shape[0]
        grid_tensor = self._grid_tensor(grids, count, patches.device)
        scale = self.scale_embedding(grid_tensor)

        pixel_tokens = self.input_projection(self._patch_to_tokens(patches))
        pixel_tokens = pixel_tokens + self.local_positions.unsqueeze(0) + scale.unsqueeze(1)
        summary = self.encoder_summary_token.expand(count, -1, -1) + scale.unsqueeze(1)
        encoded = self.encoder(torch.cat((summary, pixel_tokens), dim=1))
        latent = self.to_latent(self.encoder_norm(encoded[:, 0]))
        return self.latent_norm(latent)

    def decode(
        self,
        latents: torch.Tensor,
        grids: int | float | torch.Tensor,
    ) -> torch.Tensor:
        count = latents.shape[0]
        grid_tensor = self._grid_tensor(grids, count, latents.device)
        scale = self.scale_embedding(grid_tensor)

        memory = self.latent_to_memory(latents).unsqueeze(1) + scale.unsqueeze(1)
        queries = (
            self.decoder_queries.unsqueeze(0)
            + self.local_positions.unsqueeze(0)
            + scale.unsqueeze(1)
        ).expand(count, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        values = torch.sigmoid(self.output_projection(self.decoder_norm(decoded)))
        return self._tokens_to_patch(values)

    def forward(
        self,
        patches: torch.Tensor,
        grids: int | float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(patches, grids)
        return latents, self.decode(latents, grids)


# -----------------------------------------------------------------------------
# Resolution-aware block-causal autoregressor
# -----------------------------------------------------------------------------


class FourierPositionEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_frequencies: int) -> None:
        super().__init__()
        frequencies = math.pi * (2.0 ** torch.arange(num_frequencies, dtype=torch.float32))
        self.register_buffer("frequencies", frequencies, persistent=False)
        encoded_dim = input_dim * (1 + 2 * num_frequencies)
        self.network = nn.Sequential(
            nn.Linear(encoded_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        angles = features.unsqueeze(-1) * self.frequencies
        encoded = torch.cat(
            (
                features,
                torch.sin(angles).flatten(start_dim=-2),
                torch.cos(angles).flatten(start_dim=-2),
            ),
            dim=-1,
        )
        return self.network(encoded)


class PyramidAutoregressor(nn.Module):
    PROMPT_ROLE = 0
    QUERY_ROLE = 1
    CONTENT_ROLE = 2

    def __init__(
        self,
        spec: PyramidSpec,
        latent_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float,
        fourier_frequencies: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.register_buffer("node_features", spec.features.clone(), persistent=False)

        self.position_embedding = FourierPositionEmbedding(
            input_dim=spec.features.shape[1],
            d_model=d_model,
            num_frequencies=fourier_frequencies,
        )
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.role_embeddings = nn.Parameter(torch.randn(3, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.latent_head = nn.Linear(d_model, latent_dim)

    def _positions(self, node_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        indices = torch.as_tensor(node_ids, dtype=torch.long, device=device)
        return self.position_embedding(self.node_features.index_select(0, indices))

    @staticmethod
    def _block_causal_mask(prompt_count: int, target_count: int, device: torch.device) -> torch.Tensor:
        sequence_length = prompt_count + 2 * target_count - 1
        mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device)
        mask[:prompt_count, :prompt_count] = False
        for row in range(prompt_count, sequence_length):
            mask[row, :prompt_count] = False
            mask[row, prompt_count : row + 1] = False
        return mask

    def forward_teacher(
        self,
        prompt_latents: torch.Tensor,
        prompt_ids: Sequence[int],
        target_latents: torch.Tensor,
        target_ids: Sequence[int],
    ) -> torch.Tensor:
        """Teacher-forced prediction of all target latents in one masked pass."""

        batch, prompt_count, _ = prompt_latents.shape
        target_count = target_latents.shape[1]
        if prompt_count != len(prompt_ids) or target_count != len(target_ids):
            raise ValueError("latent tensors and node-id lists disagree")
        if prompt_count < 1 or target_count < 1:
            raise ValueError("prompt and target sets must both be nonempty")

        device = prompt_latents.device
        prompt_positions = self._positions(prompt_ids, device)
        target_positions = self._positions(target_ids, device)

        prompt_tokens = (
            self.latent_projection(prompt_latents)
            + prompt_positions.unsqueeze(0)
            + self.role_embeddings[self.PROMPT_ROLE]
        )

        pieces: list[torch.Tensor] = [prompt_tokens]
        for index in range(target_count):
            query = (
                target_positions[index]
                + self.role_embeddings[self.QUERY_ROLE]
            ).view(1, 1, -1).expand(batch, -1, -1)
            pieces.append(query)
            if index < target_count - 1:
                content = (
                    self.latent_projection(target_latents[:, index])
                    + target_positions[index]
                    + self.role_embeddings[self.CONTENT_ROLE]
                ).unsqueeze(1)
                pieces.append(content)

        sequence = torch.cat(pieces, dim=1)
        mask = self._block_causal_mask(prompt_count, target_count, device)
        hidden = self.transformer(sequence, mask=mask, is_causal=False)
        hidden = self.final_norm(hidden)

        query_indices = prompt_count + 2 * torch.arange(target_count, device=device)
        return self.latent_head(hidden.index_select(1, query_indices))

    @torch.no_grad()
    def generate(
        self,
        prompt_latents: torch.Tensor,
        prompt_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> torch.Tensor:
        """Autoregressively generate target latents; no decoding occurs in the loop."""

        batch, prompt_count, _ = prompt_latents.shape
        if prompt_count != len(prompt_ids) or prompt_count < 1:
            raise ValueError("prompt tensor and prompt node list disagree")
        if not target_ids:
            return prompt_latents.new_empty((batch, 0, self.latent_dim))

        device = prompt_latents.device
        prompt_positions = self._positions(prompt_ids, device)
        target_positions = self._positions(target_ids, device)
        prompt_tokens = (
            self.latent_projection(prompt_latents)
            + prompt_positions.unsqueeze(0)
            + self.role_embeddings[self.PROMPT_ROLE]
        )

        generated: list[torch.Tensor] = []
        for current in range(len(target_ids)):
            pieces: list[torch.Tensor] = [prompt_tokens]
            for previous in range(current):
                query = (
                    target_positions[previous]
                    + self.role_embeddings[self.QUERY_ROLE]
                ).view(1, 1, -1).expand(batch, -1, -1)
                content = (
                    self.latent_projection(generated[previous])
                    + target_positions[previous]
                    + self.role_embeddings[self.CONTENT_ROLE]
                ).unsqueeze(1)
                pieces.extend((query, content))

            current_query = (
                target_positions[current]
                + self.role_embeddings[self.QUERY_ROLE]
            ).view(1, 1, -1).expand(batch, -1, -1)
            pieces.append(current_query)

            sequence = torch.cat(pieces, dim=1)
            mask = self._block_causal_mask(prompt_count, current + 1, device)
            hidden = self.transformer(sequence, mask=mask, is_causal=False)
            prediction = self.latent_head(self.final_norm(hidden[:, -1]))
            generated.append(prediction)

        return torch.stack(generated, dim=1)


# -----------------------------------------------------------------------------
# Shared tensor helpers and losses
# -----------------------------------------------------------------------------


def pyramid_patches(
    pyramid: dict[int, torch.Tensor],
    spec: PyramidSpec,
    patch_size: int,
) -> dict[int, torch.Tensor]:
    return {
        level: patchify(pyramid[grid], patch_size)
        for level, grid in enumerate(spec.grid_sizes)
    }


@torch.no_grad()
def encode_pyramid(
    tokenizer: TransformerPatchAutoencoder,
    patches_by_level: dict[int, torch.Tensor],
    spec: PyramidSpec,
) -> dict[int, torch.Tensor]:
    latents: dict[int, torch.Tensor] = {}
    for level, grid in enumerate(spec.grid_sizes):
        patches = patches_by_level[level]
        batch, count, channels, height, width = patches.shape
        flat = patches.reshape(batch * count, channels, height, width)
        encoded = tokenizer.encode(flat, grid)
        latents[level] = encoded.reshape(batch, count, -1)
    return latents


def gather_nodes(
    tensors_by_level: dict[int, torch.Tensor],
    node_ids: Sequence[int],
    spec: PyramidSpec,
) -> torch.Tensor:
    if not node_ids:
        raise ValueError("cannot gather an empty node list")
    return torch.stack(
        [
            tensors_by_level[spec.nodes[node_id].level][:, spec.nodes[node_id].local_index]
            for node_id in node_ids
        ],
        dim=1,
    )


def decode_node_latents(
    tokenizer: TransformerPatchAutoencoder,
    latents: torch.Tensor,
    node_ids: Sequence[int],
    spec: PyramidSpec,
) -> torch.Tensor:
    batch, count, latent_dim = latents.shape
    if count != len(node_ids):
        raise ValueError("latent count and node-id count disagree")
    grids = torch.tensor(
        [spec.nodes[node_id].grid for node_id in node_ids],
        device=latents.device,
        dtype=torch.float32,
    )
    flat_grids = grids.unsqueeze(0).expand(batch, -1).reshape(-1)
    decoded = tokenizer.decode(latents.reshape(batch * count, latent_dim), flat_grids)
    return decoded.reshape(
        batch,
        count,
        tokenizer.channels,
        tokenizer.patch_size,
        tokenizer.patch_size,
    )


def predicted_mean_consistency_loss(
    decoded_patches: torch.Tensor,
    target_ids: Sequence[int],
    spec: PyramidSpec,
) -> tuple[torch.Tensor, int]:
    """Compare a predicted parent to the mean of a complete predicted child group."""

    index_by_id = {node_id: index for index, node_id in enumerate(target_ids)}
    losses: list[torch.Tensor] = []

    for parent_id in target_ids:
        children = spec.children(parent_id)
        if not children or not all(child in index_by_id for child in children):
            continue

        parent = spec.nodes[parent_id]
        child = spec.nodes[children[0]]
        factor = child.grid // parent.grid
        child_rows: list[torch.Tensor] = []
        for row_offset in range(factor):
            row_patches = [
                decoded_patches[:, index_by_id[children[row_offset * factor + col_offset]]]
                for col_offset in range(factor)
            ]
            child_rows.append(torch.cat(row_patches, dim=-1))
        stitched = torch.cat(child_rows, dim=-2)
        child_mean = F.avg_pool2d(stitched, kernel_size=factor, stride=factor)
        parent_patch = decoded_patches[:, index_by_id[parent_id]]
        losses.append(F.mse_loss(parent_patch, child_mean))

    if not losses:
        return decoded_patches.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


def tokenizer_batch_loss(
    tokenizer: TransformerPatchAutoencoder,
    images: torch.Tensor,
    spec: PyramidSpec,
    patch_size: int,
    mean_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pyramid = build_mean_pyramid(images, spec.grid_sizes, patch_size)
    reconstructions: dict[int, torch.Tensor] = {}
    reconstruction_losses: list[torch.Tensor] = []

    for level, grid in enumerate(spec.grid_sizes):
        true_image = pyramid[grid]
        true_patches = patchify(true_image, patch_size)
        batch, count, channels, height, width = true_patches.shape
        flat = true_patches.reshape(batch * count, channels, height, width)
        _, decoded = tokenizer(flat, grid)
        decoded_patches = decoded.reshape(batch, count, channels, height, width)
        reconstructed = unpatchify(decoded_patches, grid, patch_size)
        reconstructions[level] = reconstructed
        reconstruction_losses.append(F.mse_loss(reconstructed, true_image))

    reconstruction_loss = torch.stack(reconstruction_losses).mean()

    consistency_losses: list[torch.Tensor] = []
    for coarse_level in range(len(spec.grid_sizes) - 1):
        coarse_grid = spec.grid_sizes[coarse_level]
        fine_grid = spec.grid_sizes[coarse_level + 1]
        factor = fine_grid // coarse_grid
        mean_of_children = F.avg_pool2d(
            reconstructions[coarse_level + 1], kernel_size=factor, stride=factor
        )
        consistency_losses.append(
            F.mse_loss(reconstructions[coarse_level], mean_of_children)
        )

    if consistency_losses:
        consistency_loss = torch.stack(consistency_losses).mean()
    else:
        consistency_loss = reconstruction_loss.new_zeros(())

    total_loss = reconstruction_loss + mean_weight * consistency_loss
    return total_loss, reconstruction_loss, consistency_loss


def autoregressive_batch_loss(
    model: PyramidAutoregressor,
    tokenizer: TransformerPatchAutoencoder,
    prompt_latents: torch.Tensor,
    prompt_ids: Sequence[int],
    target_latents: torch.Tensor,
    target_patches: torch.Tensor,
    target_ids: Sequence[int],
    spec: PyramidSpec,
    loss_mode: str,
    lambda_z: float,
    lambda_x: float,
    mean_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    predictions = model.forward_teacher(prompt_latents, prompt_ids, target_latents, target_ids)
    latent_loss = F.mse_loss(predictions, target_latents)
    decoded = decode_node_latents(tokenizer, predictions, target_ids, spec)
    pixel_loss = F.mse_loss(decoded, target_patches)
    mean_loss, mean_groups = predicted_mean_consistency_loss(decoded, target_ids, spec)

    if loss_mode == "latent":
        total_loss = latent_loss + mean_weight * mean_loss
    else:
        total_loss = lambda_z * latent_loss + lambda_x * pixel_loss + mean_weight * mean_loss
    return total_loss, latent_loss, pixel_loss, mean_loss, mean_groups


# -----------------------------------------------------------------------------
# Validation and plotting
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_tokenizer(
    tokenizer: TransformerPatchAutoencoder,
    loader: DataLoader[torch.Tensor],
    spec: PyramidSpec,
    patch_size: int,
    mean_weight: float,
    device: torch.device,
) -> dict[str, float]:
    tokenizer.eval()
    totals = {"loss": 0.0, "reconstruction": 0.0, "mean": 0.0, "examples": 0}
    for images in loader:
        images = images.to(device, non_blocking=True)
        total, reconstruction, consistency = tokenizer_batch_loss(
            tokenizer, images, spec, patch_size, mean_weight
        )
        batch = images.shape[0]
        totals["loss"] += total.item() * batch
        totals["reconstruction"] += reconstruction.item() * batch
        totals["mean"] += consistency.item() * batch
        totals["examples"] += batch
    count = max(1, totals.pop("examples"))
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def make_tokenizer_figure(
    tokenizer: TransformerPatchAutoencoder,
    fixed_images: torch.Tensor,
    spec: PyramidSpec,
    patch_size: int,
    epoch: int,
    device: torch.device,
) -> plt.Figure:
    tokenizer.eval()
    true_images = fixed_images.to(device)
    final_grid = spec.grid_sizes[-1]
    patches = patchify(true_images, patch_size)
    batch, count, channels, height, width = patches.shape
    _, decoded = tokenizer(
        patches.reshape(batch * count, channels, height, width), final_grid
    )
    decoded_images = unpatchify(
        decoded.reshape(batch, count, channels, height, width), final_grid, patch_size
    )
    differences = true_images - decoded_images

    rows = [true_images, decoded_images, differences]
    row_labels = ["true", "decoded", "difference"]
    figure, axes = plt.subplots(3, 8, figsize=(16, 6), squeeze=False)
    cmaps = ["Spectral_r", "Spectral_r", "coolwarm"]
    for row_index, row_images in enumerate(rows):
        for column in range(8):
            vmin = 0 if row_index<2 else None
            vmax = 1 if row_index<2 else None
            axes[row_index, column].imshow(
                row_images[column, 0].detach().cpu().numpy(),
                cmap=cmaps[row_index],
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 0].set_ylabel(row_labels[row_index])

    resolution = final_grid * patch_size
    figure.suptitle(
        f"Tokenizer epoch {epoch}: final resolution {resolution}x{resolution}", fontsize=12
    )
    figure.subplots_adjust(wspace=0, hspace=0)
    return figure


@torch.no_grad()
def evaluate_autoregressor(
    model: PyramidAutoregressor,
    tokenizer: TransformerPatchAutoencoder,
    loader: DataLoader[torch.Tensor],
    spec: PyramidSpec,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    tokenizer.eval()
    totals = {"loss": 0.0, "latent": 0.0, "pixel": 0.0, "mean": 0.0, "batches": 0}

    for batch_index, images in enumerate(loader):
        images = images.to(device, non_blocking=True)
        pyramid = build_mean_pyramid(images, spec.grid_sizes, args.patch_size)
        patches = pyramid_patches(pyramid, spec, args.patch_size)
        latents = encode_pyramid(tokenizer, patches, spec)

        # Deterministic layouts, cycling through prompt/request resolution pairs.
        request_level = batch_index % len(spec.grid_sizes)
        prompt_level = (batch_index // len(spec.grid_sizes)) % len(spec.grid_sizes)
        layout_rng = random.Random(args.seed + 100_000 + batch_index)
        layout = sample_layout(
            spec,
            layout_rng,
            args.min_prompt_patches,
            args.max_prompt_patches,
            args.min_request_patches,
            args.max_request_patches,
            full_request_probability=0.0,
            prompt_level=prompt_level,
            request_level=request_level,
        )

        prompt_latents = gather_nodes(latents, layout.prompt_ids, spec)
        target_latents = gather_nodes(latents, layout.target_ids, spec)
        target_patches = gather_nodes(patches, layout.target_ids, spec)
        loss, latent_loss, pixel_loss, mean_loss, _ = autoregressive_batch_loss(
            model,
            tokenizer,
            prompt_latents,
            layout.prompt_ids,
            target_latents,
            target_patches,
            layout.target_ids,
            spec,
            args.ar_loss,
            args.lambda_z,
            args.lambda_x,
            args.ar_mean_weight,
        )

        totals["loss"] += loss.item()
        totals["latent"] += latent_loss.item()
        totals["pixel"] += pixel_loss.item()
        totals["mean"] += mean_loss.item()
        totals["batches"] += 1

    count = max(1, totals.pop("batches"))
    return {key: value / count for key, value in totals.items()}


def _canvas_from_selected_patches(
    source_patches: torch.Tensor,
    selected_local_indices: Iterable[int],
    grid: int,
    patch_size: int,
) -> torch.Tensor:
    canvas_patches = torch.zeros_like(source_patches)
    for local_index in selected_local_indices:
        canvas_patches[:, local_index] = source_patches[:, local_index]
    return unpatchify(canvas_patches, grid, patch_size)


@torch.no_grad()
def make_autoregressive_figure(
    model: PyramidAutoregressor,
    tokenizer: TransformerPatchAutoencoder,
    fixed_images: torch.Tensor,
    layout: Layout,
    spec: PyramidSpec,
    patch_size: int,
    epoch: int,
    device: torch.device,
) -> plt.Figure:
    model.eval()
    tokenizer.eval()
    images = fixed_images.to(device)
    pyramid = build_mean_pyramid(images, spec.grid_sizes, patch_size)
    patches = pyramid_patches(pyramid, spec, patch_size)
    latents = encode_pyramid(tokenizer, patches, spec)

    prompt_latents = gather_nodes(latents, layout.prompt_ids, spec)
    generated_latents = model.generate(prompt_latents, layout.prompt_ids, layout.target_ids)
    generated_index = {node_id: index for index, node_id in enumerate(layout.target_ids)}

    request_grid = spec.grid_sizes[layout.request_level]
    request_level_patches = patches[layout.request_level]
    generated_request_patches = torch.zeros_like(request_level_patches)

    prompt_set = set(layout.prompt_ids)
    unknown_request_ids = [node_id for node_id in layout.request_ids if node_id not in prompt_set]
    if unknown_request_ids:
        unknown_latents = torch.stack(
            [generated_latents[:, generated_index[node_id]] for node_id in unknown_request_ids],
            dim=1,
        )
        decoded_unknown = decode_node_latents(
            tokenizer, unknown_latents, unknown_request_ids, spec
        )
        for output_index, node_id in enumerate(unknown_request_ids):
            generated_request_patches[:, spec.nodes[node_id].local_index] = decoded_unknown[
                :, output_index
            ]

    # Requested prompt patches are copied exactly.
    for node_id in layout.request_ids:
        if node_id in prompt_set:
            local_index = spec.nodes[node_id].local_index
            generated_request_patches[:, local_index] = request_level_patches[:, local_index]

    request_local_indices = [spec.nodes[node_id].local_index for node_id in layout.request_ids]
    prompt_local_indices = [spec.nodes[node_id].local_index for node_id in layout.prompt_ids]
    true_request = _canvas_from_selected_patches(
        request_level_patches, request_local_indices, request_grid, patch_size
    )
    prompt_only = _canvas_from_selected_patches(
        request_level_patches, prompt_local_indices, request_grid, patch_size
    )
    generated_request = unpatchify(generated_request_patches, request_grid, patch_size)
    difference = true_request - generated_request

    rows = [true_request, prompt_only, generated_request, difference]
    row_labels = [
        "true requested",
        "prompt only",
        "generated requested",
        "difference",
    ]
    figure, axes = plt.subplots(4, 8, figsize=(16, 8), squeeze=False)
    cmaps = ["Spectral_r", "Spectral_r", "Spectral_r", "coolwarm"]
    for row_index, row_images in enumerate(rows):
        for column in range(8):
            vmin = 0 if row_index<3 else None
            vmax = 1 if row_index<3 else None
            axes[row_index, column].imshow(
                row_images[column, 0].detach().cpu().numpy(),
                cmap=cmaps[row_index],
                vmin=vmin,
                vmax=vmax,
            )
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
        axes[row_index, 0].set_ylabel(row_labels[row_index])

    resolution = request_grid * patch_size
    figure.suptitle(
        f"AR epoch {epoch} | request order={layout.request_level + 1} | "
        f"grid={request_grid}x{request_grid} patches | resolution={resolution}x{resolution}",
        fontsize=12,
    )
    figure.subplots_adjust(wspace=0, hspace=0)
    return figure


# -----------------------------------------------------------------------------
# Training loops and W&B logging
# -----------------------------------------------------------------------------


def configure_wandb_metrics(run: Any) -> None:
    run.define_metric("tokenizer/batch_step")
    run.define_metric("tokenizer/train_batch_*", step_metric="tokenizer/batch_step")
    run.define_metric("tokenizer/epoch")
    run.define_metric("tokenizer/train_epoch_*", step_metric="tokenizer/epoch")
    run.define_metric("tokenizer/val_*", step_metric="tokenizer/epoch", summary="min")

    run.define_metric("ar/batch_step")
    run.define_metric("ar/train_batch_*", step_metric="ar/batch_step")
    run.define_metric("ar/epoch")
    run.define_metric("ar/train_epoch_*", step_metric="ar/epoch")
    run.define_metric("ar/val_*", step_metric="ar/epoch", summary="min")


def save_checkpoint(path: Path, module: nn.Module, args: argparse.Namespace, stage: str) -> None:
    torch.save(
        {
            "stage": stage,
            "model_state_dict": module.state_dict(),
            "grid_sizes": list(args.grid_sizes),
            "patch_size": args.patch_size,
            "config": vars(args),
        },
        path,
    )


def train_tokenizer(
    tokenizer: TransformerPatchAutoencoder,
    train_loader: DataLoader[torch.Tensor],
    val_loader: DataLoader[torch.Tensor],
    fixed_images: torch.Tensor,
    spec: PyramidSpec,
    args: argparse.Namespace,
    device: torch.device,
    run: Any,
) -> None:
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=args.tokenizer_lr,
        weight_decay=args.weight_decay,
    )
    batch_step = 0

    for epoch in range(1, args.tokenizer_epochs + 1):
        tokenizer.train()
        epoch_totals = {"loss": 0.0, "reconstruction": 0.0, "mean": 0.0, "batches": 0}

        for images in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total, reconstruction, consistency = tokenizer_batch_loss(
                tokenizer,
                images,
                spec,
                args.patch_size,
                args.tokenizer_mean_weight,
            )
            total.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(tokenizer.parameters(), args.gradient_clip)
            optimizer.step()

            batch_step += 1
            epoch_totals["loss"] += total.item()
            epoch_totals["reconstruction"] += reconstruction.item()
            epoch_totals["mean"] += consistency.item()
            epoch_totals["batches"] += 1
            run.log(
                {
                    "tokenizer/batch_step": batch_step,
                    "tokenizer/train_batch_loss": total.item(),
                    "tokenizer/train_batch_reconstruction": reconstruction.item(),
                    "tokenizer/train_batch_mean_consistency": consistency.item(),
                }
            )

        validation = evaluate_tokenizer(
            tokenizer,
            val_loader,
            spec,
            args.patch_size,
            args.tokenizer_mean_weight,
            device,
        )
        figure = make_tokenizer_figure(
            tokenizer, fixed_images, spec, args.patch_size, epoch, device
        )
        figure_path = args.output_dir / f"tokenizer_epoch_{epoch:03d}.png"
        figure.savefig(figure_path, dpi=140)

        count = max(1, epoch_totals.pop("batches"))
        epoch_average = {key: value / count for key, value in epoch_totals.items()}
        run.log(
            {
                "tokenizer/epoch": epoch,
                "tokenizer/train_epoch_loss": epoch_average["loss"],
                "tokenizer/train_epoch_reconstruction": epoch_average["reconstruction"],
                "tokenizer/train_epoch_mean_consistency": epoch_average["mean"],
                "tokenizer/val_loss": validation["loss"],
                "tokenizer/val_reconstruction": validation["reconstruction"],
                "tokenizer/val_mean_consistency": validation["mean"],
                "tokenizer/reconstruction_figure": wandb.Image(str(figure_path)),
            }
        )
        plt.close(figure)
        save_checkpoint(args.output_dir / "tokenizer_latest.pt", tokenizer, args, "tokenizer")

        print(
            f"[tokenizer {epoch:03d}/{args.tokenizer_epochs:03d}] "
            f"train={epoch_average['loss']:.6f} val={validation['loss']:.6f}"
        )


def train_autoregressor(
    model: PyramidAutoregressor,
    tokenizer: TransformerPatchAutoencoder,
    train_loader: DataLoader[torch.Tensor],
    val_loader: DataLoader[torch.Tensor],
    fixed_images: torch.Tensor,
    plot_layout: Layout,
    spec: PyramidSpec,
    args: argparse.Namespace,
    device: torch.device,
    run: Any,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.ar_lr,
        weight_decay=args.weight_decay,
    )
    layout_rng = random.Random(args.seed + 50_000)
    batch_step = 0

    tokenizer.eval()
    tokenizer.requires_grad_(False)

    for epoch in range(1, args.ar_epochs + 1):
        model.train()
        epoch_totals = {
            "loss": 0.0,
            "latent": 0.0,
            "pixel": 0.0,
            "mean": 0.0,
            "batches": 0,
        }

        for images in train_loader:
            images = images.to(device, non_blocking=True)
            pyramid = build_mean_pyramid(images, spec.grid_sizes, args.patch_size)
            patches = pyramid_patches(pyramid, spec, args.patch_size)
            latents = encode_pyramid(tokenizer, patches, spec)
            layout = sample_layout(
                spec,
                layout_rng,
                args.min_prompt_patches,
                args.max_prompt_patches,
                args.min_request_patches,
                args.max_request_patches,
                args.full_request_probability,
            )

            prompt_latents = gather_nodes(latents, layout.prompt_ids, spec)
            target_latents = gather_nodes(latents, layout.target_ids, spec)
            target_patches = gather_nodes(patches, layout.target_ids, spec)

            optimizer.zero_grad(set_to_none=True)
            loss, latent_loss, pixel_loss, mean_loss, mean_groups = autoregressive_batch_loss(
                model,
                tokenizer,
                prompt_latents,
                layout.prompt_ids,
                target_latents,
                target_patches,
                layout.target_ids,
                spec,
                args.ar_loss,
                args.lambda_z,
                args.lambda_x,
                args.ar_mean_weight,
            )
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            batch_step += 1
            epoch_totals["loss"] += loss.item()
            epoch_totals["latent"] += latent_loss.item()
            epoch_totals["pixel"] += pixel_loss.item()
            epoch_totals["mean"] += mean_loss.item()
            epoch_totals["batches"] += 1
            run.log(
                {
                    "ar/batch_step": batch_step,
                    "ar/train_batch_loss": loss.item(),
                    "ar/train_batch_latent": latent_loss.item(),
                    "ar/train_batch_pixel": pixel_loss.item(),
                    "ar/train_batch_mean_consistency": mean_loss.item(),
                    "ar/train_batch_mean_groups": mean_groups,
                    "ar/train_batch_prompt_count": len(layout.prompt_ids),
                    "ar/train_batch_request_count": len(layout.request_ids),
                    "ar/train_batch_generated_count": len(layout.target_ids),
                    "ar/train_batch_prompt_order": layout.prompt_level + 1,
                    "ar/train_batch_request_order": layout.request_level + 1,
                }
            )

        validation = evaluate_autoregressor(
            model, tokenizer, val_loader, spec, args, device
        )
        figure = make_autoregressive_figure(
            model,
            tokenizer,
            fixed_images,
            plot_layout,
            spec,
            args.patch_size,
            epoch,
            device,
        )

        count = max(1, epoch_totals.pop("batches"))
        epoch_average = {key: value / count for key, value in epoch_totals.items()}
        run.log(
            {
                "ar/epoch": epoch,
                "ar/train_epoch_loss": epoch_average["loss"],
                "ar/train_epoch_latent": epoch_average["latent"],
                "ar/train_epoch_pixel": epoch_average["pixel"],
                "ar/train_epoch_mean_consistency": epoch_average["mean"],
                "ar/val_loss": validation["loss"],
                "ar/val_latent": validation["latent"],
                "ar/val_pixel": validation["pixel"],
                "ar/val_mean_consistency": validation["mean"],
                "ar/generation_figure": wandb.Image(figure),
            }
        )
        plt.close(figure)
        save_checkpoint(args.output_dir / "autoregressor_latest.pt", model, args, "autoregressor")

        print(
            f"[autoregressor {epoch:03d}/{args.ar_epochs:03d}] "
            f"train={epoch_average['loss']:.6f} val={validation['loss']:.6f}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    if wandb is None:
        raise SystemExit("Weights & Biases is required. Install it with: pip install wandb")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.torch_threads, 4)))
    except RuntimeError:
        # Inter-op threads can only be set once in a process.
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_size = args.grid_sizes[-1] * args.patch_size
    train_dataset = GaussianImageDataset(
        args.train_size,
        image_size,
        args.sigma_min,
        args.sigma_max,
        args.mean_row,
        args.mean_col,
        seed=args.seed,
    )
    val_dataset = GaussianImageDataset(
        args.val_size,
        image_size,
        args.sigma_min,
        args.sigma_max,
        args.mean_row,
        args.mean_col,
        seed=args.seed + 1,
    )

    loader_generator = torch.Generator().manual_seed(args.seed + 2)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    fixed_images = torch.stack([val_dataset[index] for index in range(8)])

    spec = PyramidSpec(args.grid_sizes)
    tokenizer = TransformerPatchAutoencoder(
        patch_size=args.patch_size,
        token_size=args.tokenizer_token_size,
        channels=1,
        latent_dim=args.latent_dim,
        d_model=args.tokenizer_d_model,
        num_heads=args.tokenizer_heads,
        encoder_layers=args.tokenizer_encoder_layers,
        decoder_layers=args.tokenizer_decoder_layers,
        ffn_dim=args.tokenizer_ffn_dim,
        dropout=args.tokenizer_dropout,
    ).to(device)
    autoregressor = PyramidAutoregressor(
        spec=spec,
        latent_dim=args.latent_dim,
        d_model=args.ar_d_model,
        num_heads=args.ar_heads,
        num_layers=args.ar_layers,
        ffn_dim=args.ar_ffn_dim,
        dropout=args.ar_dropout,
        fourier_frequencies=args.fourier_frequencies,
    ).to(device)

    plot_layout = make_plot_layout(
        spec,
        request_level=args.plot_request_order - 1,
        prompt_count=args.plot_prompt_patches,
        request_count=args.plot_request_patches,
        seed=args.seed + 3,
    )

    serializable_config = dict(vars(args))
    serializable_config["output_dir"] = str(args.output_dir)
    serializable_config["image_size"] = image_size
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable_config, handle, indent=2)

    print(f"device: {device}")
    print(f"pyramid grids: {args.grid_sizes}")
    print(f"finest image: {image_size}x{image_size}; patch: {args.patch_size}x{args.patch_size}")
    print(f"tokenizer parameters: {parameter_count(tokenizer):,}")
    print(f"autoregressor parameters: {parameter_count(autoregressor):,}")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        mode=args.wandb_mode,
        config=serializable_config,
    )
    try:
        configure_wandb_metrics(run)
        train_tokenizer(
            tokenizer,
            train_loader,
            val_loader,
            fixed_images,
            spec,
            args,
            device,
            run,
        )
        train_autoregressor(
            autoregressor,
            tokenizer,
            train_loader,
            val_loader,
            fixed_images,
            plot_layout,
            spec,
            args,
            device,
            run,
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
