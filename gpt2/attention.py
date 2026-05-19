import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.W_query = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1),
        )

    def forward(self, x):
        _, num_tokens, _ = x.shape
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)

        attn_scores = query @ key.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / key.shape[1] ** 0.5, dim=-1)

        return attn_weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0, "emb_dim must be divisible by n_heads"

        self.d_out = cfg["emb_dim"]
        self.num_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]

        self.W_query = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_key = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_value = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1),
        )

    def forward(self, x, return_attn_weights = False):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        out = self.out_proj(context_vec)
        if return_attn_weights:
            return out, attn_weights
        return out
