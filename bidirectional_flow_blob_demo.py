#!/usr/bin/env python3
"""
Toy bidirectional conditional flow matching demo.

Implements:
  1. Synthetic 128x128 grayscale Gaussian-blob images paired with labels
     y = [sigma_1, sigma_2], with sigma_i in [20, 40] pixels.
  2. A small Vision Transformer autoencoder.
  3. A shared transformer vector field for:
       - image-latent flow conditioned on labels: p(z_image | y)
       - label flow conditioned on image latents: p(y | z_image)
  4. Straight-path conditional flow-matching training.
  5. Sampling labels given an image.
  6. Sampling images given labels.

The defaults are intentionally small enough for a toy demo. A CUDA GPU is recommended.

Example:
    python toy_bidirectional_flow_matching.py --device cuda --ae-epochs 15 --flow-epochs 30
"""



from __future__ import annotations
import os

import argparse
import importlib
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torch.utils.data import DataLoader, Dataset, random_split
import wandb

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 1. Toy dataset
# -----------------------------------------------------------------------------


class GaussianBlobDataset(Dataset):
    """
    Generates a single anisotropic 2D Gaussian per image.

    Labels:
        y[0] = sigma_1 in pixels
        y[1] = sigma_2 in pixels

    Optional random center jitter, amplitude variation, and observation noise make
    p(image | label) non-degenerate while preserving the label-controlled widths.
    """

    def __init__(
        self,
        n_samples: int,
        image_size: int = 128,
        sigma_min: float = 20.0,
        sigma_max: float = 40.0,
        center_jitter: float = 3.0,
        amplitude_range: tuple[float, float] = (0.85, 1.0),
        noise_std: float = 0.01,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.image_size = image_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        generator = torch.Generator().manual_seed(seed)

        self.sigmas = sigma_min + (sigma_max - sigma_min) * torch.rand(
            n_samples, 2, generator=generator
        )

        center = (image_size - 1) / 2.0
        self.centers = center + center_jitter * torch.randn(n_samples, 2, generator=generator)
        amp_min, amp_max = amplitude_range
        self.amplitudes = amp_min + (amp_max - amp_min) * torch.rand(n_samples, generator=generator)
        self.noise = noise_std * torch.randn(
            n_samples, 1, image_size, image_size, generator=generator
        )

        coords = torch.arange(image_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.xx = xx
        self.yy = yy

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sigma_x, sigma_y = self.sigmas[index]
        center_x, center_y = self.centers[index]
        amplitude = self.amplitudes[index]

        exponent = -0.5 * (
            ((self.xx - center_x) / sigma_x) ** 2 + ((self.yy - center_y) / sigma_y) ** 2
        )
        image = amplitude * torch.exp(exponent)
        image = image.unsqueeze(0) + self.noise[index]
        image = image.clamp(0.0, 1.0)

        label = torch.tensor([sigma_x, sigma_y], dtype=torch.float32)
        return image, label


# -----------------------------------------------------------------------------
# Label transform
# -----------------------------------------------------------------------------


class LabelStandardizer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clamp_min(1e-6).clone())

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return (labels - self.mean) / self.std

    def inverse(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.std + self.mean


# -----------------------------------------------------------------------------
# 2. Simplest useful Vision Transformer autoencoder
# -----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(images)
        return tokens.flatten(2).transpose(1, 2)


class TinyViTAutoencoder(nn.Module):
    """
    Encoder:
        patchify -> positional embeddings -> TransformerEncoder

    Decoder:
        each latent token -> reconstructed pixel patch -> unpatchify

    This is deliberately minimal: no CLS token and no convolutional decoder.
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        self.position = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
        )

        patch_dim = in_channels * patch_size * patch_size
        self.patch_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, patch_dim),
        )

        nn.init.trunc_normal_(self.position, std=0.02)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        return self.encoder(tokens + self.position)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size = patches.shape[0]
        p = self.patch_size
        g = self.grid_size
        c = self.in_channels

        patches = patches.view(batch_size, g, g, c, p, p)
        images = torch.einsum("bhwcpq->bchpwq", patches)
        return images.reshape(batch_size, c, g * p, g * p)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        patches = self.patch_decoder(latents)
        return torch.sigmoid(self.unpatchify(patches))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(images)
        reconstructions = self.decode(latents)
        return reconstructions, latents


class LatentStandardizer(nn.Module):
    """Per-token-channel affine normalization of ViT latent tokens."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clamp_min(1e-6).clone())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.mean) / self.std

    def inverse(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.std + self.mean


# -----------------------------------------------------------------------------
# 3. Shared bidirectional flow transformer
# -----------------------------------------------------------------------------


class FourierTimeEmbedding(nn.Module):
    def __init__(self, model_dim: int, n_frequencies: int = 16) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_frequencies, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * time[:, None] * self.frequencies[None, :]
        features = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.mlp(features)


class BidirectionalFlowTransformer(nn.Module):
    """
    One shared transformer implements two conditional vector fields.

    Task 0: image latent flow conditioned on clean labels
        image_tokens are the time-dependent state
        label_state is the clean condition

    Task 1: label flow conditioned on clean image latents
        label_state is the time-dependent state
        image_tokens are the clean condition
    """

    IMAGE_GIVEN_LABEL = 0
    LABEL_GIVEN_IMAGE = 1

    def __init__(
        self,
        num_image_tokens: int,
        image_token_dim: int,
        num_labels: int = 2,
        model_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_image_tokens = num_image_tokens
        self.image_token_dim = image_token_dim
        self.num_labels = num_labels
        self.model_dim = model_dim

        self.image_input = nn.Linear(image_token_dim, model_dim)
        self.label_scalar_input = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        self.image_position = nn.Parameter(torch.zeros(1, num_image_tokens, model_dim))
        self.label_position = nn.Parameter(torch.zeros(1, num_labels, model_dim))
        self.task_embedding = nn.Embedding(2, model_dim)
        self.modality_embedding = nn.Embedding(2, model_dim)

        self.time_embedding = FourierTimeEmbedding(model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
        )

        self.image_velocity_head = nn.Linear(model_dim, image_token_dim)
        self.label_velocity_head = nn.Linear(model_dim, 1)

        nn.init.trunc_normal_(self.image_position, std=0.02)
        nn.init.trunc_normal_(self.label_position, std=0.02)
        nn.init.zeros_(self.image_velocity_head.weight)
        nn.init.zeros_(self.image_velocity_head.bias)
        nn.init.zeros_(self.label_velocity_head.weight)
        nn.init.zeros_(self.label_velocity_head.bias)

    def forward(
        self,
        image_tokens: torch.Tensor,
        label_state: torch.Tensor,
        time: torch.Tensor,
        task: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = image_tokens.shape[0]
        task_ids = torch.full(
            (batch_size,),
            task,
            device=image_tokens.device,
            dtype=torch.long,
        )

        time_emb = self.time_embedding(time)
        task_emb = self.task_embedding(task_ids)

        image_emb = (
            self.image_input(image_tokens)
            + self.image_position
            + self.modality_embedding.weight[0][None, None, :]
        )
        label_emb = (
            self.label_scalar_input(label_state.unsqueeze(-1))
            + self.label_position
            + self.modality_embedding.weight[1][None, None, :]
        )

        # Add task and time information to every token.
        global_emb = (time_emb + task_emb)[:, None, :]
        image_emb = image_emb + global_emb
        label_emb = label_emb + global_emb

        task_token = (time_emb + task_emb)[:, None, :]
        sequence = torch.cat([task_token, label_emb, image_emb], dim=1)
        output = self.backbone(sequence)

        label_features = output[:, 1 : 1 + self.num_labels]
        image_features = output[:, 1 + self.num_labels :]

        image_velocity = self.image_velocity_head(image_features)
        label_velocity = self.label_velocity_head(label_features).squeeze(-1)
        return image_velocity, label_velocity


# -----------------------------------------------------------------------------
# Straight-path conditional flow matching
# -----------------------------------------------------------------------------


@dataclass
class FlowLosses:
    total: torch.Tensor
    image: torch.Tensor
    label: torch.Tensor


def flow_matching_loss(
    model: BidirectionalFlowTransformer,
    clean_image_latents: torch.Tensor,
    clean_labels: torch.Tensor,
    image_weight: float = 1.0,
    label_weight: float = 1.0,
) -> FlowLosses:
    batch_size = clean_image_latents.shape[0]
    device = clean_image_latents.device

    # Image latent flow: N(0, I) -> encoded image latent, conditioned on label.
    image_noise = torch.randn_like(clean_image_latents)
    time_x = torch.rand(batch_size, device=device)
    time_x_view = time_x[:, None, None]
    image_state_t = (1.0 - time_x_view) * image_noise + time_x_view * clean_image_latents
    image_target_velocity = clean_image_latents - image_noise

    predicted_image_velocity, _ = model(
        image_tokens=image_state_t,
        label_state=clean_labels,
        time=time_x,
        task=BidirectionalFlowTransformer.IMAGE_GIVEN_LABEL,
    )
    image_loss = F.mse_loss(
        predicted_image_velocity,
        image_target_velocity,
    )

    # Label flow: N(0, I) -> standardized label, conditioned on clean image latent.
    label_noise = torch.randn_like(clean_labels)
    time_y = torch.rand(batch_size, device=device)
    time_y_view = time_y[:, None]
    label_state_t = (1.0 - time_y_view) * label_noise + time_y_view * clean_labels
    label_target_velocity = clean_labels - label_noise

    _, predicted_label_velocity = model(
        image_tokens=clean_image_latents,
        label_state=label_state_t,
        time=time_y,
        task=BidirectionalFlowTransformer.LABEL_GIVEN_IMAGE,
    )
    label_loss = F.mse_loss(
        predicted_label_velocity,
        label_target_velocity,
    )

    total_loss = image_weight * image_loss + label_weight * label_loss
    return FlowLosses(total_loss, image_loss, label_loss)


# -----------------------------------------------------------------------------
# 4. Training
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_autoencoder(
    model: TinyViTAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for images, _ in loader:
        images = images.to(device)
        reconstructions, _ = model(images)
        loss = F.mse_loss(reconstructions, images, reduction="sum")
        total += loss.item()
        count += images.numel()
    return total / max(count, 1)


def train_autoencoder(
    model: TinyViTAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    wandb_run: Any | None = None,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    step = 1
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            reconstructions, _ = model(images)
            loss = F.mse_loss(reconstructions, images)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "autoencoder/train_loss": loss.detach().cpu().item(),
                        "step": step,
                    }
                )
            step += 1

        train_loss = running / len(train_loader)
        val_loss = evaluate_autoencoder(model, val_loader, device)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "autoencoder/train_loss": train_loss,
                    "autoencoder/val_loss": val_loss,
                    "epoch": epoch,
                }
            )

            fig = make_encoder_figure(images.cpu().numpy().squeeze(), labels.cpu().numpy(), reconstructions.detach().cpu().numpy().squeeze())
            wandb_run.log(
                {
                    "autoencoder/figure": wandb.Image(fig),
                    "epoch": epoch,
                }
            )
            plt.close(fig)
        
        print(f"[AE] epoch {epoch:03d}/{epochs} train_mse={train_loss:.6f} val_mse={val_loss:.6f}", flush=True)

        

@torch.no_grad()
def estimate_latent_statistics(
    autoencoder: TinyViTAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> LatentStandardizer:
    autoencoder.eval()
    sum_x = None
    sum_x2 = None
    count = 0

    for images, _ in loader:
        latents = autoencoder.encode(images.to(device))
        batch_sum = latents.sum(dim=(0, 1))
        batch_sum2 = (latents**2).sum(dim=(0, 1))
        batch_count = latents.shape[0] * latents.shape[1]

        sum_x = batch_sum if sum_x is None else sum_x + batch_sum
        sum_x2 = batch_sum2 if sum_x2 is None else sum_x2 + batch_sum2
        count += batch_count

    mean = sum_x / count
    variance = sum_x2 / count - mean**2
    std = variance.clamp_min(1e-6).sqrt()
    return LatentStandardizer(mean, std).to(device)


@torch.no_grad()
def evaluate_flow_model(
    flow_model: BidirectionalFlowTransformer,
    autoencoder: TinyViTAutoencoder,
    latent_standardizer: LatentStandardizer,
    label_standardizer: LabelStandardizer,
    loader: DataLoader,
    device: torch.device,
) -> FlowLosses:
    flow_model.eval()
    autoencoder.eval()
    running_total = 0.0
    running_image = 0.0
    running_label = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        image_latents = autoencoder.encode(images)
        image_latents = latent_standardizer(image_latents)
        standardized_labels = label_standardizer(labels)

        losses = flow_matching_loss(
            flow_model,
            clean_image_latents=image_latents,
            clean_labels=standardized_labels,
        )
        running_total += losses.total.item()
        running_image += losses.image.item()
        running_label += losses.label.item()

    n_batches = max(len(loader), 1)
    return FlowLosses(
        total=torch.tensor(running_total / n_batches),
        image=torch.tensor(running_image / n_batches),
        label=torch.tensor(running_label / n_batches),
    )


def train_flow_model(
    flow_model: BidirectionalFlowTransformer,
    autoencoder: TinyViTAutoencoder,
    latent_standardizer: LatentStandardizer,
    label_standardizer: LabelStandardizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    wandb_run: Any | None = None,
    epoch_callback: Callable[[int], None] | None = None,
) -> None:
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    step = 1
    for epoch in range(1, epochs + 1):
        flow_model.train()
        running_total = 0.0
        running_image = 0.0
        running_label = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                image_latents = autoencoder.encode(images)
                image_latents = latent_standardizer(image_latents)
                standardized_labels = label_standardizer(labels)

            losses = flow_matching_loss(
                flow_model,
                clean_image_latents=image_latents,
                clean_labels=standardized_labels,
            )

            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
            optimizer.step()

            running_total += losses.total.item()
            running_image += losses.image.item()
            running_label += losses.label.item()

            if wandb_run is not None:
                wandb_run.log({
                        "flow/train_loss": losses.total.detach().cpu().item(),
                        "flow/train_image_loss": losses.image.detach().cpu().item(),
                        "flow/train_label_loss": losses.label.detach().cpu().item(),
                        "step": step,
                    })
            
            step += 1


        n_batches = len(train_loader)
        train_total = running_total / n_batches
        train_image = running_image / n_batches
        train_label = running_label / n_batches
        val_losses = evaluate_flow_model(
            flow_model,
            autoencoder,
            latent_standardizer,
            label_standardizer,
            val_loader,
            device,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "flow/val_loss": val_losses.total.item(),
                    "flow/val_image_loss": val_losses.image.item(),
                    "flow/val_label_loss": val_losses.label.item(),
                    "epoch": epoch,
                }
            )
        if epoch_callback is not None:
            epoch_callback(epoch)
        print(
            f"[FLOW] epoch {epoch:03d}/{epochs} "
            f"train_total={train_total:.6f} "
            f"train_image={train_image:.6f} "
            f"train_label={train_label:.6f} "
            f"val_total={val_losses.total.item():.6f} "
            f"val_image={val_losses.image.item():.6f} "
            f"val_label={val_losses.label.item():.6f}",
            flush=True,
        )


# -----------------------------------------------------------------------------
# 5/6. ODE integration and conditional sampling
# -----------------------------------------------------------------------------


@torch.no_grad()
def integrate_image_flow(
    model: BidirectionalFlowTransformer,
    initial_state: torch.Tensor,
    label_condition: torch.Tensor,
    steps: int = 50,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """
    Fixed-step midpoint ODE integrator.

    Midpoint is more accurate than Euler while staying dependency-free.
    The same function can integrate forward or backward.
    """
    state = initial_state
    dt = (t_end - t_start) / steps
    batch_size = state.shape[0]

    for step in range(steps):
        t0 = t_start + step * dt
        t_mid = t0 + 0.5 * dt

        time_0 = torch.full((batch_size,), t0, device=state.device, dtype=state.dtype)
        velocity_0, _ = model(
            image_tokens=state,
            label_state=label_condition,
            time=time_0,
            task=BidirectionalFlowTransformer.IMAGE_GIVEN_LABEL,
        )
        midpoint = state + 0.5 * dt * velocity_0

        time_mid = torch.full((batch_size,), t_mid, device=state.device, dtype=state.dtype)
        velocity_mid, _ = model(
            image_tokens=midpoint,
            label_state=label_condition,
            time=time_mid,
            task=BidirectionalFlowTransformer.IMAGE_GIVEN_LABEL,
        )
        state = state + dt * velocity_mid

    return state


@torch.no_grad()
def integrate_label_flow(
    model: BidirectionalFlowTransformer,
    initial_state: torch.Tensor,
    image_condition: torch.Tensor,
    steps: int = 50,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    state = initial_state
    dt = (t_end - t_start) / steps
    batch_size = state.shape[0]

    for step in range(steps):
        t0 = t_start + step * dt
        t_mid = t0 + 0.5 * dt

        time_0 = torch.full((batch_size,), t0, device=state.device, dtype=state.dtype)
        _, velocity_0 = model(
            image_tokens=image_condition,
            label_state=state,
            time=time_0,
            task=BidirectionalFlowTransformer.LABEL_GIVEN_IMAGE,
        )
        midpoint = state + 0.5 * dt * velocity_0

        time_mid = torch.full((batch_size,), t_mid, device=state.device, dtype=state.dtype)
        _, velocity_mid = model(
            image_tokens=image_condition,
            label_state=midpoint,
            time=time_mid,
            task=BidirectionalFlowTransformer.LABEL_GIVEN_IMAGE,
        )
        state = state + dt * velocity_mid

    return state


@torch.no_grad()
def sample_images_given_labels(
    flow_model: BidirectionalFlowTransformer,
    autoencoder: TinyViTAutoencoder,
    latent_standardizer: LatentStandardizer,
    label_standardizer: LabelStandardizer,
    physical_labels: torch.Tensor,
    samples_per_label: int,
    ode_steps: int,
) -> torch.Tensor:
    device = next(flow_model.parameters()).device
    physical_labels = physical_labels.to(device)
    standardized_labels = label_standardizer(physical_labels)

    repeated_labels = standardized_labels.repeat_interleave(samples_per_label, dim=0)
    total_samples = repeated_labels.shape[0]

    image_noise = torch.randn(
        total_samples,
        autoencoder.num_patches,
        autoencoder.embed_dim,
        device=device,
    )
    normalized_latents = integrate_image_flow(
        flow_model,
        initial_state=image_noise,
        label_condition=repeated_labels,
        steps=ode_steps,
    )
    raw_latents = latent_standardizer.inverse(normalized_latents)
    generated_images = autoencoder.decode(raw_latents)

    return generated_images.view(
        physical_labels.shape[0],
        samples_per_label,
        1,
        autoencoder.image_size,
        autoencoder.image_size,
    )


@torch.no_grad()
def sample_labels_given_images(
    flow_model: BidirectionalFlowTransformer,
    autoencoder: TinyViTAutoencoder,
    latent_standardizer: LatentStandardizer,
    label_standardizer: LabelStandardizer,
    images: torch.Tensor,
    samples_per_image: int,
    ode_steps: int,
) -> torch.Tensor:
    device = next(flow_model.parameters()).device
    images = images.to(device)

    image_latents = autoencoder.encode(images)
    image_latents = latent_standardizer(image_latents)
    repeated_latents = image_latents.repeat_interleave(samples_per_image, dim=0)

    label_noise = torch.randn(
        repeated_latents.shape[0],
        2,
        device=device,
    )
    standardized_samples = integrate_label_flow(
        flow_model,
        initial_state=label_noise,
        image_condition=repeated_latents,
        steps=ode_steps,
    )
    physical_samples = label_standardizer.inverse(standardized_samples)

    return physical_samples.view(
        images.shape[0],
        samples_per_image,
        2,
    )


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def make_encoder_figure(true_image, true_label, generated_images):

    num_images = 8
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 4, 3 * 4))
    cmap = "turbo"
    
    for i in range(num_images):
        axes[0, i].pcolormesh(true_image[i], cmap=cmap)
        axes[0, i].set_title(f"Tru {true_label[i]}")
        axes[0, i].axis("off")
        
        axes[1, i].pcolormesh(generated_images[i], cmap=cmap)
        axes[1, i].set_title(f"Gen {true_label[i]}")
        axes[1, i].axis("off")

        axes[2, i].pcolormesh(true_image[i] - generated_images[i], cmap=cmap)
        axes[2, i].set_title(f"Dif {true_label[i]}")
        axes[2, i].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    return fig

def make_demo_figure(
    observed_image: torch.Tensor,
    true_label: torch.Tensor,
    sampled_labels: torch.Tensor,
    conditioning_labels: torch.Tensor,
    generated_images: torch.Tensor,
) -> None:
    observed_image = observed_image.detach().cpu().squeeze().numpy()
    true_label = true_label.detach().cpu().numpy()
    sampled_labels = sampled_labels.detach().cpu().numpy()
    conditioning_labels = conditioning_labels.detach().cpu().numpy()
    generated_images = generated_images.detach().cpu().numpy()

    n_conditions = conditioning_labels.shape[0]
    n_samples = generated_images.shape[1]

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.scatter(sampled_labels[:, 0], sampled_labels[:, 1], alpha=0.65)
    ax.scatter([true_label[0]], [true_label[1]], marker="x", s=120)
    ax.set_xlim(15, 45)
    ax.set_ylim(15, 45)
    ax.set_xlabel("sampled sigma_1")
    ax.set_ylabel("sampled sigma_2")
    ax.set_title("Samples from label flow: p(label | image)")
    ax.grid(alpha=0.25)

    cmap = "turbo"
    fig2, axes = plt.subplots(n_conditions, n_samples, figsize=(n_samples * 4, n_conditions * 4))
    for row in range(n_conditions):
        for col in range(n_samples):
            ax = axes[row, col]
            ax.pcolormesh(
                generated_images[row, col, 0],
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            ax.set_title(
                f"condition σ=({conditioning_labels[row, 0]:.0f}, "
                f"{conditioning_labels[row, 1]:.0f})"
            )
            ax.axis("off")

    fig2.subplots_adjust(hspace=0.001, wspace=0.001)
    fig2.tight_layout()
    return fig1, fig2


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-ae", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--flow-epochs", type=int, default=100)
    parser.add_argument("--ae-lr", type=float, default=2e-4)
    parser.add_argument("--flow-lr", type=float, default=2e-4)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("toy_flow_outputs"))
    parser.add_argument("--wandb", action="store_true", help="Log losses and demo figures to Weights & Biases.")
    parser.add_argument("--wandb-project", default="bidirectional-flow-blob-demo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def init_wandb(args: argparse.Namespace) -> Any | None:
    if not args.wandb:
        return None

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )


def main() -> None:

    print("Starting demo...", flush=True)
    
    args = parse_args()

    print(f"Arguments: {args}", flush=True)

    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(args)
    image_size = 128

    print(f"Using device: {device}", flush=True)

    dataset = GaussianBlobDataset(
        n_samples=args.n_samples,
        image_size=image_size,
        sigma_min=20.0,
        sigma_max=40.0,
        center_jitter=3.0,
        noise_std=0.01,
        seed=args.seed,
    )

    print(f"Dataset: {dataset}", flush=True)

    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Exact dataset label statistics are available from the generated labels.
    label_mean = dataset.sigmas[train_set.indices].mean(dim=0).to(device)
    label_std = dataset.sigmas[train_set.indices].std(dim=0).to(device)
    label_standardizer = LabelStandardizer(label_mean, label_std).to(device)

    autoencoder = TinyViTAutoencoder(
        image_size=image_size,
        patch_size=16,
        in_channels=1,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    ).to(device)
    torchinfo.summary(autoencoder, input_size=(args.batch_size, 1, image_size, image_size))

    print("\nTraining ViT autoencoder...", flush=True)

    if args.checkpoint_ae is None:

        train_autoencoder(
            autoencoder,
            train_loader,
            val_loader,
            device,
            epochs=args.ae_epochs,
            learning_rate=args.ae_lr,
            wandb_run=wandb_run,
        )

        checkpoint = {
            "autoencoder": autoencoder.state_dict(),
            "args": vars(args),
        }
        fname = args.output_dir / "toy_ae_checkpoint.pt"
        torch.save(checkpoint, fname)
        print(f"Saved checkpoint to: {fname}", flush=True)

    else:

        checkpoint = torch.load(args.checkpoint_ae, weights_only=False, map_location=device)
        autoencoder.load_state_dict(checkpoint["autoencoder"])
        print(f"Loaded checkpoint from: {args.checkpoint_ae}", flush=True)


    latent_standardizer = estimate_latent_statistics(
        autoencoder,
        train_loader,
        device,
    )

    flow_model = BidirectionalFlowTransformer(
        num_image_tokens=autoencoder.num_patches,
        image_token_dim=autoencoder.embed_dim,
        num_labels=2,
        model_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
    ).to(device)

    observed_image, true_label = val_set[0]
    conditioning_labels = torch.tensor(
        [
            [22.0, 22.0],
            [22.0, 38.0],
            [38.0, 22.0],
            [38.0, 38.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    def log_demo_figure(epoch: int) -> None:

        label_samples = sample_labels_given_images(
            flow_model,
            autoencoder,
            latent_standardizer,
            label_standardizer,
            images=observed_image.unsqueeze(0),
            samples_per_image=100,
            ode_steps=args.ode_steps,
        )[0]
        generated_images = sample_images_given_labels(
            flow_model,
            autoencoder,
            latent_standardizer,
            label_standardizer,
            physical_labels=conditioning_labels,
            samples_per_label=8,
            ode_steps=args.ode_steps,
        )
        fig1, fig2 = make_demo_figure(
            observed_image,
            true_label,
            label_samples,
            conditioning_labels,
            generated_images,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "demo/figure1": wandb.Image(fig1),
                    "demo/figure2": wandb.Image(fig2),
                    "epoch": epoch,
                }
            )
            print('Saved figures to wandb', flush=True)
        plt.close(fig1)
        plt.close(fig2)



    print("\nTraining bidirectional flow-matching model...", flush=True)
    train_flow_model(
        flow_model,
        autoencoder,
        latent_standardizer,
        label_standardizer,
        train_loader,
        val_loader,
        device,
        epochs=args.flow_epochs,
        learning_rate=args.flow_lr,
        wandb_run=wandb_run,
        epoch_callback=log_demo_figure,
    )

    checkpoint = {
        "autoencoder": autoencoder.state_dict(),
        "flow_model": flow_model.state_dict(),
        "latent_mean": latent_standardizer.mean,
        "latent_std": latent_standardizer.std,
        "label_mean": label_standardizer.mean,
        "label_std": label_standardizer.std,
        "args": vars(args),
    }
    torch.save(checkpoint, args.output_dir / "toy_bidirectional_flow.pt")

    # -------------------------------------------------------------------------
    # 5. Demonstrate sampling labels given one observed image.
    # -------------------------------------------------------------------------
    label_samples = sample_labels_given_images(
        flow_model,
        autoencoder,
        latent_standardizer,
        label_standardizer,
        images=observed_image.unsqueeze(0),
        samples_per_image=100,
        ode_steps=args.ode_steps,
    )[0]

    print("\nTrue label:", true_label.tolist(), flush=True)
    print(
        "Posterior sample mean:",
        label_samples.mean(dim=0).cpu().tolist(),
        flush=True
    )
    print(
        "Posterior sample std:",
        label_samples.std(dim=0).cpu().tolist(),
        flush=True,
    )

    # -------------------------------------------------------------------------
    # 6. Demonstrate sampling images given labels.
    # -------------------------------------------------------------------------
    generated_images = sample_images_given_labels(
        flow_model,
        autoencoder,
        latent_standardizer,
        label_standardizer,
        physical_labels=conditioning_labels,
        samples_per_label=3,
        ode_steps=args.ode_steps,
    )

    figure_path = args.output_dir / "bidirectional_sampling_demo.png"
    make_demo_figure(
        observed_image,
        true_label,
        label_samples,
        conditioning_labels,
        generated_images,
    )

    print(f"\nSaved checkpoint to: {args.output_dir / 'toy_bidirectional_flow.pt'}", flush=True)
    print(f"Saved demonstration figure to: {figure_path}", flush=True)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    print("Starting main...", flush=True)
    main()
