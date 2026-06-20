import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .layers import FeedForward, get_norm
from .transformer import Transformer


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.trf_blocks = nn.Sequential(*[Transformer(cfg) for _ in range(cfg["n_layers"])])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.layer_norm = get_norm(cfg)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        self.apply(self._init_weights)
        # Residual projections accumulate across N blocks (2 per block: attn + ffn).
        # Scale their std by 1/sqrt(2*N) so residual stream variance stays O(1).
        residual_std = 0.02 / (2 * cfg["n_layers"]) ** 0.5
        for module in self.modules():
            if isinstance(module, MultiHeadAttention):
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=residual_std)
            elif isinstance(module, FeedForward):
                last = module.layers[2]
                if isinstance(last, nn.Linear):
                    nn.init.normal_(last.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx, return_attn_weights=False):
        _, seq_len = in_idx.shape
        tok_emb = self.token_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.dropout(tok_emb + pos_emb)
        if return_attn_weights:
            all_attn_weights = []
            for block in self.trf_blocks:
                x, attn_weights = block(x, return_attn_weights=True)
                all_attn_weights.append(attn_weights)
            x = self.layer_norm(x)
            return self.out_head(x), all_attn_weights
        x = self.trf_blocks(x)
        x = self.layer_norm(x)
        return self.out_head(x)
