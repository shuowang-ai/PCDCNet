"""Prediction loss used by the released training pipeline."""

from __future__ import annotations

import torch


def prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all samples, steps, stations and channels."""
    return (prediction - target).abs().mean()
