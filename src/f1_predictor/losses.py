"""Masked L1 loss for delta-regression over valid driver slots.

Mirrors the LightGBM baseline's robust ``regression_l1`` objective: the delta
distribution (places gained) is heavy-tailed, so absolute error avoids the
front-of-grid over-correction that squared loss causes. Only slots active at the
snapshot lap contribute, so padded/retired drivers cannot affect the gradient.
"""
from __future__ import annotations

import torch


def masked_l1_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error between predicted and target delta over valid slots.

    pred / target / valid are [B, N]. Returns a scalar.
    """
    m = valid.float()
    abs_err = (pred - target).abs() * m
    return abs_err.sum() / m.sum().clamp(min=1.0)
