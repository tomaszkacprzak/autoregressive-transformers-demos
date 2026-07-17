#!/usr/bin/env python3
"""Toy arbitrary-resolution super-resolution with a real-space neural operator.

The "tokenizer" is a continuous convolutional encoder/decoder. It uses no
vector quantization and no codebook.

Operator:
    z = encoder(low_image)
    z(y) = bilinear interpolation of z at target coordinate y
    output(y) = decoder(z(y), y, target_pixel_size)
"""

import argparse
import random
from pathlib import Path

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
# Dataset
# -----------------------------------------------------------------------------


class GaussianDataset(Dataset):
    """Centered 128x128 Gaussians with covariance diag(sigma_x^2, sigma_y^2)."""

    def __init__(self, count, size=128, sigma_min=10.0, sigma_max=20.0, seed=0):
        generator = torch.Generator().manual_seed(seed)
        sigma_x = sigma_min + (sigma_max - sigma_min) * torch.rand(
            count, 1, 1, generator=generator
        )
        sigma_y = sigma_min + (sigma_max - sigma_min) * torch.rand(
            count, 1, 1, generator=generator
        )

        p = torch.arange(size, dtype=torch.float32)
        y, x = torch.meshgrid(p, p, indexing="ij")
        center = (size - 1) / 2

        image = torch.exp(
            -0.5
            * (
                (x[None] - center).square() / sigma_x.square()
                + (y[None] - center).square() / sigma_y.square()
            )
        )
        self.images = image[:, None]  # [N, 1, H, W]
        # sigma_x and sigma_y are intentionally not stored or returned.

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


# -----------------------------------------------------------------------------
# Continuous real-space operator
# -----------------------------------------------------------------------------


class ContinuousOperator(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, latent_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1),
            nn.GELU(),
        )
        # latent + x + y + target cell width + target cell height
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim + 4, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    @staticmethod
    def query_grid(batch, height, width, device, dtype):
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2 / width - 1
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2 / height - 1
        y, x = torch.meshgrid(y, x, indexing="ij")
        cell_x = torch.full_like(x, 2 / width)
        cell_y = torch.full_like(y, 2 / height)
        q = torch.stack([x, y, cell_x, cell_y])[None]
        return q.expand(batch, -1, -1, -1)

    def forward(self, image, output_size=None):
        if output_size is None:
            output_size = image.shape[-2:]
        height, width = output_size

        latent = self.encoder(image)

        # Real-space local operator evaluation:
        # latent(y) = sum_i bilinear_weight_i(y) * latent_i
        latent_at_queries = F.interpolate(
            latent, size=output_size, mode="bilinear", align_corners=False
        )
        queries = self.query_grid(
            image.shape[0], height, width, image.device, image.dtype
        )
        return torch.sigmoid(self.decoder(torch.cat([latent_at_queries, queries], 1)))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def resize(image, size):
    if image.shape[-2:] == tuple(size):
        return image
    return F.interpolate(
        image,
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def take_eight(dataset, device):
    return torch.stack([dataset[i] for i in range(8)]).to(device)


def make_figure(rows, row_names, cmaps=None):
    """rows is a list of tensors, each with shape [8, 1, H, W]."""
    figure, axes = plt.subplots(len(rows), 8, figsize=(16, 2 * len(rows)), squeeze=False)

    if cmaps is None:
        cmaps = ["turbo"] * len(rows)

    for row_index, row in enumerate(rows):
        for column in range(8):
            image = row[column, 0].detach().cpu().numpy()
            kwargs = {"cmap": cmaps[row_index]}
            if row_index != len(rows) - 1:
                kwargs.update(vmin=0, vmax=1)
            else:
                kwargs.update(vmin=-0.1, vmax=0.1)
            axes[row_index, column].pcolormesh(image, **kwargs)
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(f"example {column + 1}")

        axes[row_index, 0].text(
            -0.10,
            0.5,
            row_names[row_index],
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
        )

    figure.subplots_adjust(wspace=0, hspace=0)
    return figure


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    if not path.exists():
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device))


def tokenizer_validation(model, loader, device):
    model.eval()
    loss_sum = 0.0
    value_count = 0
    with torch.no_grad():
        for true_image in loader:
            true_image = true_image.to(device)
            decoded = model(true_image)
            loss_sum += F.l1_loss(decoded, true_image, reduction="sum").item()
            value_count += true_image.numel()
    return loss_sum / value_count


def superres_validation(model, loader, low_size, output_size, device):
    model.eval()
    loss_sum = 0.0
    value_count = 0
    with torch.no_grad():
        for true_image in loader:
            true_image = true_image.to(device)
            target = resize(true_image, output_size)
            low = resize(target, low_size)
            prediction = model(low, output_size)
            loss_sum += F.l1_loss(prediction, target, reduction="sum").item()
            value_count += target.numel()
    return loss_sum / value_count


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_tokenizer(model, train_loader, val_loader, val_set, device, run, args, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_step = 0

    for epoch in range(1, args.tokenizer_epochs + 1):
        model.train()
        for true_image in train_loader:
            true_image = true_image.to(device)
            decoded = model(true_image)
            loss = F.l1_loss(decoded, true_image)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            run.log(
                {
                    "tokenizer/train_loss": loss.item(),
                    "tokenizer/batch": batch_step,
                    "tokenizer/epoch": epoch,
                }
            )
            batch_step += 1

        val_loss = tokenizer_validation(model, val_loader, device)
        true_image = take_eight(val_set, device)
        with torch.no_grad():
            decoded = model(true_image)
        figure = make_figure(
            [true_image, decoded, decoded - true_image],
            ["true input", "decoded", "absolute difference"],
            ["turbo", "turbo", "coolwarm"],
        )
        run.log(
            {
                "tokenizer/validation_loss": val_loss,
                "tokenizer/examples": wandb.Image(figure),
                "tokenizer/epoch": epoch,
            }
        )
        plt.close(figure)
        save_model(model, path)
        print(f"tokenizer epoch {epoch}: validation L1 = {val_loss:.6f}", flush=True)


def random_sizes(args):
    target_h = random.randint(args.target_min, args.target_max)
    target_w = random.randint(args.target_min, args.target_max)
    low_h = random.randint(args.low_min, min(args.low_max, target_h - 1))
    low_w = random.randint(args.low_min, min(args.low_max, target_w - 1))
    return (target_h, target_w), (low_h, low_w)


def train_superres(model, train_loader, val_loader, train_set, device, run, args, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_step = 0
    validation_output = (args.image_size, args.image_size)
    validation_low = (args.val_low_height, args.val_low_width)

    for epoch in range(1, args.superres_epochs + 1):
        model.train()
        for true_image in train_loader:
            true_image = true_image.to(device)
            target_size, low_size = random_sizes(args)
            target = resize(true_image, target_size)
            low = resize(target, low_size)
            prediction = model(low, target_size)
            loss = F.l1_loss(prediction, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            run.log(
                {
                    "superres/train_loss": loss.item(),
                    "superres/batch": batch_step,
                    "superres/epoch": epoch,
                    "superres/target_height": target_size[0],
                    "superres/target_width": target_size[1],
                    "superres/low_height": low_size[0],
                    "superres/low_width": low_size[1],
                }
            )
            batch_step += 1

        val_loss = superres_validation(
            model, val_loader, validation_low, validation_output, device
        )

        true_hr = take_eight(train_set, device)  # requested: examples from training set
        low = resize(true_hr, validation_low)
        with torch.no_grad():
            prediction = model(low, validation_output)
        figure = make_figure(
            [true_hr, low, prediction, prediction - true_hr],
            [
                "true high resolution",
                "low-resolution prompt",
                "super-resolved",
                "absolute difference",
            ],
            ["turbo", "turbo", "turbo", "coolwarm"],
        )
        run.log(
            {
                "superres/validation_loss": val_loss,
                "superres/examples": wandb.Image(figure),
                "superres/epoch": epoch,
            }
        )
        plt.close(figure)
        save_model(model, path)
        print(f"superres epoch {epoch}: validation L1 = {val_loss:.6f}", flush=True)


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def infer(model, dataset, device, args):
    model.eval()
    source = dataset[args.inference_index][None].to(device)
    low_size = (args.inference_low_height, args.inference_low_width)
    output_size = (args.output_height, args.output_width)

    low = resize(source, low_size)
    with torch.no_grad():
        prediction = model(low, output_size)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(
        output_path,
        prediction[0, 0].cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    print(f"saved {output_size[0]}x{output_size[1]} image to {output_path}", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tokenizer", "superres", "all", "infer"], default="all")

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--sigma-min", type=float, default=20.0)
    parser.add_argument("--sigma-max", type=float, default=40.0)

    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tokenizer-epochs", type=int, default=50)
    parser.add_argument("--superres-epochs", type=int, default=100)

    # Random target and low-resolution sizes used during super-resolution training.
    parser.add_argument("--target-min", type=int, default=64)
    parser.add_argument("--target-max", type=int, default=128)
    parser.add_argument("--low-min", type=int, default=16)
    parser.add_argument("--low-max", type=int, default=48)
    parser.add_argument("--val-low-height", type=int, default=32)
    parser.add_argument("--val-low-width", type=int, default=32)

    parser.add_argument("--checkpoint-dir", default="gaussian_operator_superres_output")
    parser.add_argument("--tokenizer-checkpoint", default=None)
    parser.add_argument("--superres-checkpoint", default=None)

    parser.add_argument("--output-height", type=int, default=192)
    parser.add_argument("--output-width", type=int, default=256)
    parser.add_argument("--inference-low-height", type=int, default=32)
    parser.add_argument("--inference-low-width", type=int, default=32)
    parser.add_argument("--inference-index", type=int, default=0)
    parser.add_argument("--output-file", default="superres.png")

    parser.add_argument("--wandb-project", default="gaussian-operator-superres")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.train_size < 8 or args.validation_size < 8:
        raise ValueError("train-size and validation-size must be at least 8")
    if args.low_min >= args.target_min:
        raise ValueError("low-min must be smaller than target-min")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    print("device:", device, flush=True)

    train_set = GaussianDataset(
        args.train_size, args.image_size, args.sigma_min, args.sigma_max, args.seed
    )
    val_set = GaussianDataset(
        args.validation_size,
        args.image_size,
        args.sigma_min,
        args.sigma_max,
        args.seed + 1,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ContinuousOperator(args.latent_dim, args.hidden_dim).to(device)

    print(torchinfo.summary(model, input_size=(1, 1, 128, 128), depth=4, device=device), flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    tokenizer_path = Path(
        args.tokenizer_checkpoint or checkpoint_dir / "tokenizer.pt"
    )
    superres_path = Path(
        args.superres_checkpoint or checkpoint_dir / "superres.pt"
    )

    if args.stage == "infer":
        load_model(model, superres_path, device)
        infer(model, val_set, device, args)
        return

    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    ) as run:
        if args.stage in {"tokenizer", "all"}:
            train_tokenizer(
                model, train_loader, val_loader, val_set, device, run, args, tokenizer_path
            )

        if args.stage == "superres" and tokenizer_path.exists():
            load_model(model, tokenizer_path, device)

        if args.stage in {"superres", "all"}:
            train_superres(
                model, train_loader, val_loader, train_set, device, run, args, superres_path
            )


if __name__ == "__main__":
    main()
