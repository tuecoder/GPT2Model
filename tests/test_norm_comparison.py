import pytest
import torch
import torch.nn.functional as F
from gpt2.model import GPTModel

BASE_CFG = {
    "vocab_size": 50257,
    "context_length": 16,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "ff_expansion": 4,
}
LN_CFG  = {**BASE_CFG, "norm_type": "layer_norm"}
RMS_CFG = {**BASE_CFG, "norm_type": "rms_norm"}


@pytest.fixture
def batch():
    torch.manual_seed(42)
    x = torch.randint(0, BASE_CFG["vocab_size"], (2, BASE_CFG["context_length"]))
    # targets are x shifted left by one token
    targets = torch.randint(0, BASE_CFG["vocab_size"], (2, BASE_CFG["context_length"]))
    return x, targets


@pytest.fixture
def ln_model():
    torch.manual_seed(0)
    return GPTModel(LN_CFG)


@pytest.fixture
def rms_model():
    torch.manual_seed(0)
    return GPTModel(RMS_CFG)


# --- Shape & validity ---

def test_output_shape_both_models(ln_model, rms_model, batch):
    x, _ = batch
    expected = (2, BASE_CFG["context_length"], BASE_CFG["vocab_size"])
    assert ln_model(x).shape == expected
    assert rms_model(x).shape == expected


def test_no_nan_output(ln_model, rms_model, batch):
    x, _ = batch
    assert not torch.isnan(ln_model(x)).any()
    assert not torch.isnan(rms_model(x)).any()
    assert not torch.isinf(ln_model(x)).any()
    assert not torch.isinf(rms_model(x)).any()


def test_loss_is_finite(ln_model, rms_model, batch):
    x, targets = batch
    ln_loss  = F.cross_entropy(ln_model(x).flatten(0, 1),  targets.flatten())
    rms_loss = F.cross_entropy(rms_model(x).flatten(0, 1), targets.flatten())
    assert torch.isfinite(ln_loss)
    assert torch.isfinite(rms_loss)


# --- Parameter count ---

def test_rms_norm_fewer_params(ln_model, rms_model):
    ln_params  = sum(p.numel() for p in ln_model.parameters())
    rms_params = sum(p.numel() for p in rms_model.parameters())
    # RMSNorm drops the shift vector: (2 norms per block * n_layers + 1 final) * emb_dim
    expected_diff = (2 * BASE_CFG["n_layers"] + 1) * BASE_CFG["emb_dim"]
    assert ln_params - rms_params == expected_diff, (
        f"Expected param diff {expected_diff}, got {ln_params - rms_params}"
    )


# --- Gradient flow ---

def test_gradients_exist_and_nonzero(ln_model, rms_model, batch):
    x, targets = batch
    for name, model in [("LayerNorm", ln_model), ("RMSNorm", rms_model)]:
        model.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
        loss.backward()
        for pname, param in model.named_parameters():
            assert param.grad is not None, f"[{name}] {pname} has no gradient"
            assert param.grad.abs().sum() > 0, f"[{name}] {pname} gradient is all zeros"


def test_gradient_norms_per_layer(ln_model, rms_model, batch):
    x, targets = batch
    results = {}
    for tag, model in [("ln", ln_model), ("rms", rms_model)]:
        model.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
        loss.backward()
        results[tag] = {
            name: param.grad.norm().item()
            for name, param in model.named_parameters()
            if param.grad is not None
        }

    print("\nGradient norm comparison (LayerNorm vs RMSNorm):")
    all_keys = sorted(set(results["ln"]) & set(results["rms"]))
    for key in all_keys:
        print(f"  {key:<50}  LN={results['ln'][key]:.4f}  RMS={results['rms'].get(key, float('nan')):.4f}")

    # Both models must have finite gradient norms for all shared params
    for key in all_keys:
        assert torch.isfinite(torch.tensor(results["ln"][key])),  f"LN  grad norm not finite: {key}"
        assert torch.isfinite(torch.tensor(results["rms"][key])), f"RMS grad norm not finite: {key}"


# --- Learning ---

def _overfit_loss(model, x, targets, steps=20, lr=1e-2):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_single_batch_overfit(ln_model, rms_model, batch):
    x, targets = batch
    ln_losses  = _overfit_loss(ln_model,  x, targets)
    rms_losses = _overfit_loss(rms_model, x, targets)
    assert ln_losses[-1]  < ln_losses[0],  "LayerNorm model did not decrease loss"
    assert rms_losses[-1] < rms_losses[0], "RMSNorm model did not decrease loss"


# --- Numerical stability ---

def test_numerical_stability_large_input(ln_model, rms_model, batch):
    x, _ = batch
    # Manually scale the token embeddings to simulate large activations
    with torch.no_grad():
        ln_model.token_emb.weight.data  *= 100
        rms_model.token_emb.weight.data *= 100
    assert not torch.isnan(ln_model(x)).any(),  "LayerNorm produced NaN on large input"
    assert not torch.isnan(rms_model(x)).any(), "RMSNorm produced NaN on large input"
