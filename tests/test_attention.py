import torch
import pytest
from gpt2.attention import MultiHeadAttention
from gpt2.model import GPTModel

TEST_CFG = {
    "vocab_size": 50257,
    "context_length": 16,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "ff_expansion": 4,
}


class TestMultiHeadAttention:
    def setup_method(self):
        self.mha = MultiHeadAttention(TEST_CFG)
        self.mha.eval()

    def test_output_shape(self):
        x = torch.randn(2, 8, 64)
        out = self.mha(x)
        assert out.shape == (2, 8, 64)

    def test_attn_weights_shape(self):
        x = torch.randn(2, 8, 64)
        out, attn_weights = self.mha(x, return_attn_weights=True)
        assert out.shape == (2, 8, 64)
        # (batch, heads, seq_len, seq_len)
        assert attn_weights.shape == (2, TEST_CFG["n_heads"], 8, 8)

    def test_causal_mask(self):
        # Positions above the diagonal must have zero attention weight
        x = torch.randn(1, 8, 64)
        _, attn_weights = self.mha(x, return_attn_weights=True)
        upper = torch.triu(attn_weights[0], diagonal=1)
        assert upper.abs().max().item() == 0.0, "causal mask violated: future tokens attended"

    def test_attn_weights_sum_to_one(self):
        x = torch.randn(2, 8, 64)
        _, attn_weights = self.mha(x, return_attn_weights=True)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_no_nan_on_large_input(self):
        # Verify 1/sqrt(d_k) scaling prevents NaN from large logits
        x = torch.randn(2, 8, 64) * 100
        out, attn_weights = self.mha(x, return_attn_weights=True)
        assert not torch.isnan(out).any(), "NaN in output for large inputs"
        assert not torch.isnan(attn_weights).any(), "NaN in attn_weights for large inputs"

    def test_default_call_unchanged(self):
        # Calling without return_attn_weights must still return a single tensor
        x = torch.randn(2, 8, 64)
        result = self.mha(x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 8, 64)


class TestGPTModelAttnWeights:
    def setup_method(self):
        self.model = GPTModel(TEST_CFG)
        self.model.eval()

    def test_default_forward_unchanged(self):
        idx = torch.randint(0, TEST_CFG["vocab_size"], (2, 8))
        logits = self.model(idx)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 8, TEST_CFG["vocab_size"])

    def test_returns_logits_and_attn_weights(self):
        idx = torch.randint(0, TEST_CFG["vocab_size"], (1, 8))
        logits, all_attn = self.model(idx, return_attn_weights=True)
        assert logits.shape == (1, 8, TEST_CFG["vocab_size"])
        assert len(all_attn) == TEST_CFG["n_layers"]

    def test_all_layer_attn_weight_shapes(self):
        idx = torch.randint(0, TEST_CFG["vocab_size"], (1, 8))
        _, all_attn = self.model(idx, return_attn_weights=True)
        for layer_idx, aw in enumerate(all_attn):
            assert aw.shape == (1, TEST_CFG["n_heads"], 8, 8), \
                f"Layer {layer_idx}: expected shape (1, {TEST_CFG['n_heads']}, 8, 8), got {aw.shape}"

    def test_all_layers_causal(self):
        idx = torch.randint(0, TEST_CFG["vocab_size"], (1, 8))
        _, all_attn = self.model(idx, return_attn_weights=True)
        for layer_idx, aw in enumerate(all_attn):
            upper = torch.triu(aw[0], diagonal=1)
            assert upper.abs().max().item() == 0.0, \
                f"Layer {layer_idx}: causal mask violated"
