#!/usr/bin/env python3
"""Continuous-patch autoregressive transformer demo.

Stage 1 trains a small patch autoencoder as a continuous tokenizer. There is no
codebook or quantization. Stage 2 freezes the tokenizer encoder and trains a
causal transformer to directly predict the 256 raw grayscale values of the next
16x16 patch with mean-squared error.
"""

from __future__ import annotations

import argparse
import os
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

try:
    import wandb
except ImportError:
    wandb = None


NUM_PLOT_EXAMPLES = 8

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
    """128x128 grayscale images of one axis-aligned 2-D Gaussian."""

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
        )
        self.sigma_col = torch.empty(size).uniform_(
            sigma_min, sigma_max, generator=generator
        )
        self.mean_row = mean_row
        self.mean_col = mean_col

        coordinates = torch.arange(image_size, dtype=torch.float32)
        self.row_grid, self.col_grid = torch.meshgrid(
            coordinates, coordinates, indexing="ij"
        )

    def __len__(self) -> int:
        return len(self.sigma_row)

    def __getitem__(self, index: int) -> torch.Tensor:
        distance = (
            ((self.row_grid - self.mean_row) / self.sigma_row[index]).square()
            + ((self.col_grid - self.mean_col) / self.sigma_col[index]).square()
        )
        return torch.exp(-0.5 * distance).unsqueeze(0)  # [1, H, W]


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B,C,H,W] -> [B,N,C*P*P], using raster order."""
    batch, channels, height, width = images.shape
    rows, cols = height // patch_size, width // patch_size
    patches = images.reshape(
        batch, channels, rows, patch_size, cols, patch_size
    )
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    return patches.reshape(batch, rows * cols, channels * patch_size**2)


def unpatchify(
    patches: torch.Tensor,
    image_size: int,
    patch_size: int,
    channels: int = 1,
) -> torch.Tensor:
    """[B,N,C*P*P] -> [B,C,H,W]."""
    batch = patches.shape[0]
    grid = image_size // patch_size
    images = patches.reshape(
        batch, grid, grid, channels, patch_size, patch_size
    )
    images = images.permute(0, 3, 1, 4, 2, 5).contiguous()
    return images.reshape(batch, channels, image_size, image_size)


class ContinuousTokenizer(nn.Module):
    """Patch autoencoder with a continuous latent vector and no codebook."""

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
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        shape = patches.shape[:-1]
        latent = self.encoder(patches.reshape(-1, self.patch_dim))
        return latent.reshape(*shape, self.latent_dim)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        shape = latent.shape[:-1]
        patches = self.decoder(latent.reshape(-1, self.latent_dim))
        return patches.reshape(*shape, self.patch_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(patches))


class PatchTransformer(nn.Module):
    """Decoder-only causal transformer with direct continuous patch output."""

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
            raise ValueError("tokenizer latent_dim must equal d_model in this demo")

        self.tokenizer = tokenizer.requires_grad_(False)
        self.grid_size = grid_size
        self.num_patches = grid_size**2
        self.patch_dim = patch_dim

        self.row_embedding = nn.Embedding(grid_size, d_model)
        self.col_embedding = nn.Embedding(grid_size, d_model)

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
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.row_embedding.weight, std=0.02)
        nn.init.normal_(self.col_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.patch_head.weight)
        nn.init.zeros_(self.patch_head.bias)
        for parameter in self.transformer.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Output at sequence position i predicts the patch at position i+1."""
        sequence_length = patches.shape[1]
        positions = torch.arange(sequence_length, device=patches.device)
        rows = positions // self.grid_size
        cols = positions % self.grid_size

        with torch.no_grad():
            tokens = self.tokenizer.encode(patches)
        tokens = tokens + (
            self.row_embedding(rows) + self.col_embedding(cols)
        ).unsqueeze(0)

        # Boolean True entries are forbidden attention locations.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=patches.device,
            ),
            diagonal=1,
        )
        hidden = self.transformer(tokens, mask=causal_mask)
        return torch.sigmoid(self.patch_head(hidden))


@torch.no_grad()
def reconstruct_images(
    tokenizer: ContinuousTokenizer,
    images: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    patches = patchify(images, patch_size)
    decoded = tokenizer(patches)
    return unpatchify(decoded, image_size, patch_size)


@torch.no_grad()
def generate_images(
    model: PatchTransformer,
    true_images: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    """Use only the true first patch; generate all remaining patches recursively."""
    true_patches = patchify(true_images, patch_size)
    generated = true_patches[:, :1].clone()

    for _ in range(1, model.num_patches):
        next_patch = model(generated)[:, -1]
        generated = torch.cat([generated, next_patch[:, None]], dim=1)

    return unpatchify(generated, image_size, patch_size)


def comparison_figure(
    true_images: torch.Tensor,
    predicted_images: torch.Tensor,
    title: str,
) -> plt.Figure:
    true = true_images[:NUM_PLOT_EXAMPLES, 0].detach().cpu().numpy()
    predicted = predicted_images[:NUM_PLOT_EXAMPLES, 0].detach().cpu().numpy()
    difference = true - predicted

    fig, axes = plt.subplots(
        3,
        NUM_PLOT_EXAMPLES,
        figsize=(16, 6),
        squeeze=False,
    )
    fig.suptitle(title)
    row_names = ("True", "Decoded / generated", "Difference")

    cmaps = ["Spectral_r", "Spectral_r", "bwr"]
    for row, images in enumerate((true, predicted, difference)):
        for col in range(NUM_PLOT_EXAMPLES):
            vmin = 0 if row < 2 else None
            vmax = 1 if row < 2 else None
            axes[row, col].imshow(images[col], cmap=cmaps[row], vmin=vmin, vmax=vmax)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(f"Example {col + 1}", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel(row_names[row])

    fig.subplots_adjust(hspace=0, wspace=0)
    return fig


@torch.no_grad()
def tokenizer_validation_loss(
    model: ContinuousTokenizer,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
) -> float:
    model.eval()
    squared_error, count = 0.0, 0
    for images in loader:
        patches = patchify(images.to(device, non_blocking=True), patch_size)
        decoded = model(patches)
        squared_error += F.mse_loss(decoded, patches, reduction="sum").item()
        count += patches.numel()
    return squared_error / count


@torch.no_grad()
def transformer_validation_loss(
    model: PatchTransformer,
    loader: DataLoader,
    device: torch.device,
    patch_size: int,
) -> float:
    model.eval()
    squared_error, count = 0.0, 0
    for images in loader:
        patches = patchify(images.to(device, non_blocking=True), patch_size)
        target = patches[:, 1:]
        prediction = model(patches[:, :-1])
        squared_error += F.mse_loss(prediction, target, reduction="sum").item()
        count += target.numel()
    return squared_error / count


def train_tokenizer(
    model: ContinuousTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    plot_examples: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint: str | None,
    run,
) -> None:

    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Tokenizer checkpoint file {checkpoint} not found")
        model.load_state_dict(torch.load(checkpoint)["state_dict"])
        print(f"Loaded tokenizer checkpoint from {checkpoint}", flush=True)
        return 

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.tokenizer_lr, weight_decay=args.weight_decay
    )
    batch_step = 0

    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        epoch_losses = []

        for images in train_loader:
            patches = patchify(
                images.to(device, non_blocking=True), args.patch_size
            ).reshape(-1, model.patch_dim)

            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(patches), patches)
            loss.backward()
            optimizer.step()

            value = loss.item()
            epoch_losses.append(value)
            run.log(
                {
                    "tokenizer/batch_step": batch_step,
                    "tokenizer/train_loss": value,
                }
            )
            batch_step += 1

        val_loss = tokenizer_validation_loss(
            model, val_loader, device, args.patch_size
        )
        fixed = plot_examples.to(device)
        reconstruction = reconstruct_images(
            model, fixed, args.image_size, args.patch_size
        )
        figure = comparison_figure(
            fixed, reconstruction, f"Tokenizer reconstruction — epoch {epoch}"
        )
        run.log(
            {
                "tokenizer/epoch": epoch,
                "tokenizer/epoch_train_loss": float(np.mean(epoch_losses)),
                "tokenizer/validation_loss": val_loss,
                "tokenizer/reconstructions": wandb.Image(figure),
            }
        )
        plt.close(figure)
        print(
            f"[tokenizer] epoch {epoch:03d}/{args.tokenizer_epochs:03d} "
            f"train={np.mean(epoch_losses):.6f} val={val_loss:.6f}", 
            flush=True,
        )


def train_transformer(
    model: PatchTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    generation_examples: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    run,
) -> None:
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.ar_lr, weight_decay=args.weight_decay
    )
    batch_step = 0

    for epoch in range(1, args.ar_epochs + 1):
        model.train()
        epoch_losses = []

        for images in train_loader:
            patches = patchify(images.to(device, non_blocking=True), args.patch_size)
            inputs, targets = patches[:, :-1], patches[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inputs), targets)
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()

            value = loss.item()
            epoch_losses.append(value)
            run.log(
                {
                    "autoregressive/batch_step": batch_step,
                    "autoregressive/train_loss": value,
                }
            )
            batch_step += 1

        val_loss = transformer_validation_loss(
            model, val_loader, device, args.patch_size
        )
        fixed = generation_examples.to(device)
        generated = generate_images(
            model, fixed, args.image_size, args.patch_size
        )
        figure = comparison_figure(
            fixed, generated, f"Autoregressive generation — epoch {epoch}"
        )
        run.log(
            {
                "autoregressive/epoch": epoch,
                "autoregressive/epoch_train_loss": float(np.mean(epoch_losses)),
                "autoregressive/validation_loss": val_loss,
                "autoregressive/generations": wandb.Image(figure),
            }
        )
        plt.close(figure)
        print(
            f"[autoregressive] epoch {epoch:03d}/{args.ar_epochs:03d} "
            f"train={np.mean(epoch_losses):.6f} val={val_loss:.6f}", 
            flush=True,
        )


def fixed_examples(dataset: Dataset) -> torch.Tensor:
    return torch.stack([dataset[i] for i in range(NUM_PLOT_EXAMPLES)])


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
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def define_metrics(run) -> None:
    run.define_metric("tokenizer/batch_step")
    run.define_metric("tokenizer/train_loss", step_metric="tokenizer/batch_step")
    run.define_metric("tokenizer/epoch")
    for name in (
        "tokenizer/epoch_train_loss",
        "tokenizer/validation_loss",
        "tokenizer/reconstructions",
    ):
        run.define_metric(name, step_metric="tokenizer/epoch")

    run.define_metric("autoregressive/batch_step")
    run.define_metric(
        "autoregressive/train_loss", step_metric="autoregressive/batch_step"
    )
    run.define_metric("autoregressive/epoch")
    for name in (
        "autoregressive/epoch_train_loss",
        "autoregressive/validation_loss",
        "autoregressive/generations",
    ):
        run.define_metric(name, step_metric="autoregressive/epoch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toy continuous-patch autoregressive transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--sigma-min", type=float, default=20.0)
    parser.add_argument("--sigma-max", type=float, default=40.0)
    parser.add_argument("--mean-row", type=float, default=64.0)
    parser.add_argument("--mean-col", type=float, default=64.0)
    parser.add_argument("--dataset-type", type=str, default="deterministic", choices=["deterministic", "nondeterministic"])

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tokenizer-epochs", type=int, default=4)
    parser.add_argument("--ar-epochs", type=int, default=4)
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

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("autoreg_outputs"))
    parser.add_argument("--checkpoint-tokenizer", default=None)

    parser.add_argument("--wandb-project", default="continuous-patch-ar-demo")
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
        raise ValueError("This requested demo uses 128x128 images and 16x16 patches")
    if args.train_size < NUM_PLOT_EXAMPLES or args.val_size < NUM_PLOT_EXAMPLES:
        raise ValueError("train-size and val-size must both be at least 8")
    if not 0 < args.sigma_min < args.sigma_max:
        raise ValueError("Require 0 < sigma-min < sigma-max")
    if args.d_model % args.num_heads:
        raise ValueError("d-model must be divisible by num-heads")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be at least 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if wandb is None:
        raise SystemExit("Install W&B first: pip install wandb")

    set_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cpu":
        # Small transformer workloads can be slower with excessive CPU threads.
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    print(f"Using device: {device}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_data_args = dict(
        image_size=args.image_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        mean_row=args.mean_row,
        mean_col=args.mean_col,
    )

    DatasetClass = GaussianDataset if args.dataset_type == "deterministic" else GaussianDatasetNondeterministic

    train_set = DatasetClass(
        size=args.train_size, seed=args.seed, **common_data_args
    )
    val_set = DatasetClass(
        size=args.val_size, seed=args.seed + 1, **common_data_args
    )
    train_loader = make_loader(
        train_set, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, args.num_workers, device
    )

    tokenizer_examples = fixed_examples(val_set)
    generation_examples = fixed_examples(train_set)
    patch_dim = args.patch_size**2
    grid_size = args.image_size // args.patch_size

    print(f"Tokenizer examples: patch_dim={patch_dim}, latent_dim={args.d_model}, hidden_dim={args.tokenizer_hidden_dim}", flush=True)
    tokenizer = ContinuousTokenizer(
        patch_dim=patch_dim,
        latent_dim=args.d_model,
        hidden_dim=args.tokenizer_hidden_dim,
    ).to(device)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
    ) as run:
        define_metrics(run)

        print(f"Training tokenizer...", flush=True)
        train_tokenizer(
            tokenizer,
            train_loader,
            val_loader,
            tokenizer_examples,
            args,
            device,
            args.checkpoint_tokenizer,
            run,
        )
        torch.save(
            {"state_dict": tokenizer.state_dict(), "config": config},
            args.output_dir / "continuous_tokenizer.pt",
        )

        transformer = PatchTransformer(
            tokenizer=tokenizer,
            grid_size=grid_size,
            patch_dim=patch_dim,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        ).to(device)

        print(f"Training transformer...", flush=True)
        train_transformer(
            transformer,
            train_loader,
            val_loader,
            generation_examples,
            args,
            device,
            run,
        )
        torch.save(
            {"state_dict": transformer.state_dict(), "config": config},
            args.output_dir / "autoregressive_transformer.pt",
        )

    print(f"Saved checkpoints in {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
