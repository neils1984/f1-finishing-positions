"""Unit tests for the masked L1 delta-regression loss."""
import pytest
import torch

from f1_predictor.losses import masked_l1_loss


def test_zero_when_exact():
    pred = torch.tensor([[1.0, -2.0, 0.5]])
    tgt = pred.clone()
    valid = torch.tensor([[True, True, True]])
    assert masked_l1_loss(pred, tgt, valid).item() == pytest.approx(0.0)


def test_matches_manual_mean_abs_error():
    pred = torch.tensor([[1.0, 0.0]])
    tgt = torch.tensor([[0.0, 2.0]])
    valid = torch.tensor([[True, True]])
    assert masked_l1_loss(pred, tgt, valid).item() == pytest.approx(1.5)  # (1+2)/2


def test_ignores_padded_slots():
    valid = torch.tensor([[True, True, False]])
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[1.0, 0.0, 999.0]])
    tgt = torch.tensor([[0.0, 0.0, 0.0]])
    assert masked_l1_loss(a, tgt, valid).item() == pytest.approx(
        masked_l1_loss(b, tgt, valid).item()
    )


def test_differentiable():
    pred = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    tgt = torch.tensor([[0.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True]])
    masked_l1_loss(pred, tgt, valid).backward()
    assert torch.isfinite(pred.grad).all()
