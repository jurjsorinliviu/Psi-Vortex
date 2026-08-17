"""Exact, sampled-density, and random-Fourier BIC effective-DoF estimators."""
from __future__ import annotations

import math
import torch

from .bic import exact_effective_dof


def minibatch_effective_dof(
    weights: torch.Tensor,
    gamma: float = 0.1,
    sample_size: int = 512,
    seed: int = 0,
) -> torch.Tensor:
    flat = weights.flatten()
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    indices = torch.randint(
        0, flat.numel(), (min(sample_size, flat.numel()),), generator=generator, device=flat.device
    )
    sampled = flat[indices]
    density = (flat.numel() / sampled.numel()) * torch.exp(
        -((flat[:, None] - sampled[None, :]).square()) / (2 * gamma * gamma)
    ).sum(dim=1)
    return (1.0 / density.clamp_min(1e-8)).sum()


def rff_effective_dof(
    weights: torch.Tensor,
    gamma: float = 0.1,
    features: int = 512,
    seed: int = 0,
) -> torch.Tensor:
    flat = weights.flatten()
    if features < 1:
        raise ValueError("features must be positive")
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    omega = torch.randn(features, generator=generator, device=flat.device, dtype=flat.dtype) / gamma
    phase = 2 * math.pi * torch.rand(
        features, generator=generator, device=flat.device, dtype=flat.dtype
    )
    mapped = math.sqrt(2.0 / features) * torch.cos(torch.outer(flat, omega) + phase)
    density = mapped @ mapped.sum(dim=0)
    return (1.0 / density.clamp_min(1e-3)).sum()


def gradient_cosine(reference: torch.Tensor, approximation: torch.Tensor, weights: torch.Tensor) -> float:
    reference_gradient = torch.autograd.grad(reference, weights, retain_graph=True)[0]
    approximate_gradient = torch.autograd.grad(approximation, weights)[0]
    return float(
        torch.nn.functional.cosine_similarity(
            reference_gradient.flatten(), approximate_gradient.flatten(), dim=0
        )
    )


def compare_estimators(
    weights: torch.Tensor,
    *,
    gamma: float = 0.1,
    sample_size: int = 512,
    features: int = 512,
    seed: int = 0,
) -> dict[str, float]:
    variable = weights.detach().clone().requires_grad_(True)
    exact = exact_effective_dof(variable, gamma)
    sampled = minibatch_effective_dof(variable, gamma, sample_size, seed)
    rff = rff_effective_dof(variable, gamma, features, seed)
    exact_value = float(exact.detach())
    return {
        "exact_dof": exact_value,
        "minibatch_dof": float(sampled.detach()),
        "rff_dof": float(rff.detach()),
        "minibatch_relative_error": abs(float(sampled.detach()) - exact_value) / exact_value,
        "rff_relative_error": abs(float(rff.detach()) - exact_value) / exact_value,
        "minibatch_gradient_cosine": gradient_cosine(exact, sampled, variable),
        "rff_gradient_cosine": gradient_cosine(
            exact_effective_dof(variable, gamma), rff_effective_dof(variable, gamma, features, seed), variable
        ),
    }
