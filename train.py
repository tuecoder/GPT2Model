import argparse
import math
import os
import urllib.request

import mlflow
import tiktoken
import torch

from gpt2 import (
    GPT_CONFIG_124M,
    GPTModel,
    create_dataloader,
    train_model_simple,
)
from gpt2.plotting import (
    plot_losses,
    plot_perplexity,
    plot_grad_norms,
    plot_lr_schedule,
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


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
        os.makedirs("docs", exist_ok=True)
        lr_plot = plot_lr_schedule(cfg["warmup_steps"], total_steps, cfg["lr"], save_path="docs/lr_schedule.png")
        mlflow.log_artifact(lr_plot, artifact_path="plots")

        evals_per_epoch = len(train_loader) // cfg["eval_freq"]
        loss_plot = plot_losses(track_tokens_seen, train_losses, val_losses, save_path="docs/loss_curve.png")
        ppl_plot = plot_perplexity(train_losses, val_losses, save_path="docs/perplexity_curve.png")
        grad_plot = plot_grad_norms(grad_norms, evals_per_epoch=evals_per_epoch, save_path="docs/grad_norm_curve.png")
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
