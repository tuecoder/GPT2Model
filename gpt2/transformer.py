import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward, get_norm


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = get_norm(cfg)
        self.layer_norm2 = get_norm(cfg)
        self.mmha = MultiHeadAttention(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.feed_forward = FeedForward(cfg)

    def forward(self, x, return_attn_weights=False):
        shortcut = x
        x = self.layer_norm1(x)
        if return_attn_weights:
            x, attn_weights = self.mmha(x, return_attn_weights=True)
        else:
            x = self.mmha(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        if return_attn_weights:
            return x, attn_weights
        return x
