import torch
import torch.nn as nn
from .layers import LayerNorm
from .transformer import Transformer


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.trf_blocks = nn.Sequential(*[Transformer(cfg) for _ in range(cfg["n_layers"])])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.layer_norm = LayerNorm(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        tok_emb = self.token_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.dropout(tok_emb + pos_emb)
        x = self.trf_blocks(x)
        x = self.layer_norm(x)
        return self.out_head(x)
