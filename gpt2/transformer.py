import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import LayerNorm, FeedForward


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg)
        self.layer_norm2 = LayerNorm(cfg)
        self.mmha = MultiHeadAttention(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.feed_forward = FeedForward(cfg)

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mmha(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
