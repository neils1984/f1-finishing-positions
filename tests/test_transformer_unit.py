"""Unit tests for the cross-driver Transformer (delta head)."""
import torch

from f1_predictor.models.transformer import DriverDeltaNet


def _inputs(B=2, N=20, F=6):
    features = torch.randn(B, N, F)
    driver_idx = torch.randint(0, 30, (B, N))
    valid = torch.ones(B, N, dtype=torch.bool)
    valid[:, 15:] = False  # last 5 slots padded
    return features, driver_idx, valid


def test_forward_output_shape():
    model = DriverDeltaNet(num_features=6, d_model=32, n_heads=4, n_layers=2,
                           num_drivers=30)
    features, driver_idx, valid = _inputs()
    out = model(features, driver_idx, valid)
    assert out.shape == (2, 20)


def test_permutation_equivariance_on_valid_slots():
    # No positional encoding => permuting drivers permutes the outputs.
    torch.manual_seed(0)
    model = DriverDeltaNet(num_features=6, d_model=32, n_heads=4, n_layers=2,
                           num_drivers=30).eval()
    features = torch.randn(1, 5, 6)
    driver_idx = torch.tensor([[1, 2, 3, 4, 5]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    perm = torch.tensor([2, 0, 4, 1, 3])
    with torch.no_grad():
        base = model(features, driver_idx, valid)[0]
        permd = model(features[:, perm], driver_idx[:, perm], valid[:, perm])[0]
    assert torch.allclose(base[perm], permd, atol=1e-5)


def test_padded_slots_do_not_change_valid_outputs():
    # Padding must not leak through attention into the active slots' deltas.
    torch.manual_seed(0)
    model = DriverDeltaNet(num_features=6, d_model=32, n_heads=4, n_layers=2,
                           num_drivers=30).eval()
    feats = torch.randn(1, 5, 6)
    didx = torch.tensor([[1, 2, 3, 4, 5]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    with torch.no_grad():
        base = model(feats, didx, valid)[0]
        feats2 = torch.cat([feats, torch.randn(1, 3, 6)], dim=1)
        didx2 = torch.cat([didx, torch.zeros(1, 3, dtype=torch.long)], dim=1)
        valid2 = torch.cat([valid, torch.zeros(1, 3, dtype=torch.bool)], dim=1)
        out2 = model(feats2, didx2, valid2)[0][:5]
    assert torch.allclose(base, out2, atol=1e-5)
