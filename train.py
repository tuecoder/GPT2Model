import argparse
import math
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tiktoken
import torch

from gpt2 import (
    GPT_CONFIG_124M,
    GPTModel,
    create_dataloader,
    train_model_simple,
)

_DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch02/01_main-chapter-code/the-verdict.txt"
)
_DEFAULT_DATA_PATH = "the-verdict.txt"

TRAIN_CONFIG = {
    "num_epochs": 10,
    "batch_size": 2,
    "max_length": 256,
    "stride": 128,
    "lr": 4e-4,
    "weight_decay": 0.1,
    "eval_freq": 5,
    "eval_iter": 5,
    "train_ratio": 0.90,
    "start_context": "Every effort moves you",
    "warmup_steps": 100,
    "accumulation_steps": 1,
}


def load_text(path: str) -> str:
    if not os.path.exists(path):
        print(f"Downloading dataset -> {path}")
        urllib.request.urlretrieve(_DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def plot_losses(tokens_seen, train_losses, val_losses, save_path="loss_curve.png"):
    eval_steps = list(range(len(train_losses)))

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(eval_steps, train_losses, color="tab:blue", label="Train loss")
    ax1.plot(eval_steps, val_losses, color="tab:orange", linestyle="--", label="Val loss")
    ax1.set_xlabel("Evaluation step")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # invisible — sets the scale
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
        # proxy artist keeps the axis range clean (no real line at x=-1)
        ax.plot([], [], color="gray", linestyle=":", linewidth=0.8, alpha=0.7,
                label="Epoch boundary")

    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Gradient L2 norm")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def make_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=_DEFAULT_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    args = parser.parse_args()

    cfg = {**TRAIN_CONFIG, "num_epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_text = load_text(args.data)
    split = int(len(raw_text) * cfg["train_ratio"])
    train_loader = create_dataloader(
        raw_text[:split],
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
        stride=cfg["stride"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        raw_text[split:],
        batch_size=cfg["batch_size"],
        max_length=cfg["max_length"],
        stride=cfg["stride"],
        shuffle=False,
        drop_last=False,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    n_decay = sum(p.numel() for p in model.parameters() if p.dim() >= 2)
    n_nodecay = sum(p.numel() for p in model.parameters() if p.dim() < 2)
    print(f"Decay params: {n_decay:,} | No-decay params (biases/norms): {n_nodecay:,}")

    optimizer = make_optimizer(model, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    steps_per_epoch = math.ceil(len(train_loader) / cfg["accumulation_steps"])
    total_steps = cfg["num_epochs"] * steps_per_epoch
    scheduler = get_lr_scheduler(optimizer, cfg["warmup_steps"], total_steps)
    print(f"LR schedule: {cfg['warmup_steps']} warmup steps, {total_steps} total steps")

    tokenizer = tiktoken.get_encoding("gpt2")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("GPT2-from-scratch")

    with mlflow.start_run(run_name="gpt2-124m"):
        mlflow.log_params({
            **GPT_CONFIG_124M,
            **cfg,
            "device": str(device),
            "n_params": n_params,
            "n_decay_params": n_decay,
            "n_nodecay_params": n_nodecay,
            "optimizer_betas": "(0.9, 0.95)",
            "total_steps": total_steps,
            "dataset": os.path.basename(args.data),
        })

        train_losses, val_losses, track_tokens_seen, grad_norms = train_model_simple(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=cfg["num_epochs"],
            eval_freq=cfg["eval_freq"],
            eval_iter=cfg["eval_iter"],
            start_context=cfg["start_context"],
            tokenizer=tokenizer,
            accumulation_steps=cfg["accumulation_steps"],
            scheduler=scheduler,
        )

        for step, (tl, vl, tokens, gn) in enumerate(
            zip(train_losses, val_losses, track_tokens_seen, grad_norms)
        ):
            mlflow.log_metrics(
                {
                    "train_loss": tl,
                    "val_loss": vl,
                    "train_perplexity": math.exp(tl),
                    "val_perplexity": math.exp(vl),
                    "tokens_seen": tokens,
                    "grad_norm": gn
                },
                step=step,
            )

        # ── Plots ─────────────────────────────────────────────────────────────
        lr_plot = plot_lr_schedule(cfg["warmup_steps"], total_steps, cfg["lr"])
        mlflow.log_artifact(lr_plot, artifact_path="plots")

        evals_per_epoch = len(train_loader) // cfg["eval_freq"]
        loss_plot = plot_losses(track_tokens_seen, train_losses, val_losses)
        ppl_plot = plot_perplexity(train_losses, val_losses)
        grad_plot = plot_grad_norms(grad_norms, evals_per_epoch=evals_per_epoch)
        mlflow.log_artifact(loss_plot, artifact_path="plots")
        mlflow.log_artifact(ppl_plot, artifact_path="plots")
        mlflow.log_artifact(grad_plot, artifact_path="plots")

        # ── Save model weights ────────────────────────────────────────────────
        weights_path = "gpt2_weights.pt"
        torch.save(model.state_dict(), weights_path)
        mlflow.log_artifact(weights_path, artifact_path="model")

        # ── Summary metrics ───────────────────────────────────────────────────
        if train_losses:
            mlflow.log_metrics({
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "final_train_perplexity": math.exp(train_losses[-1]),
                "final_val_perplexity": math.exp(val_losses[-1]),
            })

        run_id = mlflow.active_run().info.run_id
        print(f"\nDone. MLflow run: {run_id}")
        print("Run `mlflow ui` and open http://localhost:5000 to explore results.")


if __name__ == "__main__":
    main()
