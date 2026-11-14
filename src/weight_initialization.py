"""Selective orthogonal initialization for freshly constructed models."""

import math

import torch
from torch import nn


def make_initializer(architecture, method="native", *, seed=42, gain=1.0):
    """Return a builder callback, or None to use the model's defaults."""
    if architecture not in {"resnet", "vit"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    if method == "native":
        return None
    if method != "orthogonal":
        raise ValueError(f"Unknown initialization: {method}")
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("Orthogonal gain must be finite and positive")

    @torch.no_grad()
    def initialize(model):
        generators = {}
        for name, layer in model.named_modules():
            if architecture == "resnet":
                selected = isinstance(layer, nn.Conv2d)
            else:
                selected = isinstance(layer, nn.Linear) and name.endswith(
                    (".mlp.fc1", ".mlp.fc2")
                )
            if not selected:
                continue
            weight = layer.weight
            if weight.device not in generators:
                generators[weight.device] = torch.Generator(
                    device=weight.device
                ).manual_seed(seed)
            rank = min(weight.size(0), weight[0].numel())
            # Match the native tensor's RMS at gain=1, independently of shape.
            scale = gain * torch.linalg.vector_norm(weight).item() / math.sqrt(rank)
            nn.init.orthogonal_(weight, gain=scale, generator=generators[weight.device])

    return initialize
