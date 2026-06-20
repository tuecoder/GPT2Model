import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_losses(tokens_seen, train_losses, val_losses, save_path="loss_curve.png"):
    eval_steps = list(range(len(train_losses)))

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(eval_steps, train_losses, color="tab:blue", label="Train loss")
    ax1.plot(eval_steps, val_losses, color="tab:orange", linestyle="--", label="Val loss")
    ax1.set_xlabel("Evaluation step")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_perplexity(train_losses, val_losses, save_path="perplexity_curve.png"):
    eval_steps = list(range(len(train_losses)))
    train_ppl = [math.exp(l) for l in train_losses]
    val_ppl = [math.exp(l) for l in val_losses]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eval_steps, train_ppl, color="tab:blue", label="Train perplexity")
    ax.plot(eval_steps, val_ppl, color="tab:orange", linestyle="--", label="Val perplexity")
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Perplexity")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_grad_norms(grad_norms, max_norm=1.0, evals_per_epoch=None, save_path="grad_norm_curve.png"):
    steps = list(range(len(grad_norms)))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, grad_norms, color="tab:green", label="Gradient norm")
    ax.axhline(y=max_norm, color="tab:red", linestyle="--", linewidth=1.2,
               label=f"Clip threshold ({max_norm})")

    if evals_per_epoch:
        for epoch_idx in range(1, len(grad_norms) // evals_per_epoch + 1):
            boundary = epoch_idx * evals_per_epoch
            if boundary < len(grad_norms):
                ax.axvline(x=boundary, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.plot([], [], color="gray", linestyle=":", linewidth=0.8, alpha=0.7,
                label="Epoch boundary")

    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Gradient L2 norm")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_lr_schedule(warmup_steps, total_steps, max_lr, save_path="lr_schedule.png"):
    steps = list(range(total_steps + 1))
    lrs = []
    for s in steps:
        if s < warmup_steps:
            lrs.append(max_lr * s / max(1, warmup_steps))
        else:
            progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
            lrs.append(max_lr * 0.5 * (1.0 + math.cos(math.pi * progress)))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, lrs, color="tab:purple")
    ax.axvspan(0, warmup_steps, alpha=0.08, color="tab:blue", label="Warmup phase")
    ax.axvspan(warmup_steps, total_steps, alpha=0.08, color="tab:orange", label="Cosine decay phase")
    ax.axvline(x=warmup_steps, color="gray", linestyle="--", linewidth=0.9)
    ax.annotate(f"Warmup end\n(step {warmup_steps})",
                xy=(warmup_steps, max_lr), xytext=(warmup_steps + total_steps * 0.04, max_lr * 0.92),
                fontsize=8, color="gray")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
