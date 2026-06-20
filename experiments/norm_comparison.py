"""
Compares LayerNorm vs RMSNorm on perplexity convergence over a fixed batch.
Run from the project root:  python -m experiments.norm_comparison
"""
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

STEPS = 50
LR    = 1e-2


def run(cfg, x, targets):
    torch.manual_seed(0)
    model = GPTModel(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    perplexities = []
    with torch.no_grad():
        loss = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
        perplexities.append(torch.exp(loss).item())

    for _ in range(STEPS):
        opt.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
        loss.backward()
        opt.step()
        with torch.no_grad():
            eval_loss = F.cross_entropy(model(x).flatten(0, 1), targets.flatten())
            perplexities.append(torch.exp(eval_loss).item())

    return perplexities


if __name__ == "__main__":
    torch.manual_seed(42)
    x       = torch.randint(0, BASE_CFG["vocab_size"], (2, BASE_CFG["context_length"]))
    targets = torch.randint(0, BASE_CFG["vocab_size"], (2, BASE_CFG["context_length"]))

    print("Training both models for", STEPS, "steps on a fixed batch...\n")
    ln_ppl  = run(LN_CFG,  x, targets)
    rms_ppl = run(RMS_CFG, x, targets)

    print(f"{'Step':<6} {'LayerNorm PPL':>15} {'RMSNorm PPL':>13}")
    print("-" * 36)
    for step, (ln, rms) in enumerate(zip(ln_ppl, rms_ppl)):
        if step % 10 == 0 or step == STEPS:
            print(f"{step:<6} {ln:>15.2f} {rms:>13.2f}")

    print()
    print(f"Final perplexity  ->  LayerNorm: {ln_ppl[-1]:.2f}  |  RMSNorm: {rms_ppl[-1]:.2f}")
    winner = "LayerNorm" if ln_ppl[-1] < rms_ppl[-1] else "RMSNorm"
    print(f"Lower perplexity: {winner}")
