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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    tokenizer = tiktoken.get_encoding("gpt2")

    mlflow.set_experiment("GPT2-from-scratch")

    with mlflow.start_run(run_name="gpt2-124m"):
        mlflow.log_params({
            **GPT_CONFIG_124M,
            **cfg,
            "device": str(device),
            "n_params": n_params,
            "dataset": os.path.basename(args.data),
        })

        train_losses, val_losses, track_tokens_seen = train_model_simple(
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
        )

        for step, (tl, vl, tokens) in enumerate(
            zip(train_losses, val_losses, track_tokens_seen)
        ):
            mlflow.log_metrics(
                {
                    "train_loss": tl,
                    "val_loss": vl,
                    "train_perplexity": math.exp(tl),
                    "val_perplexity": math.exp(vl),
                    "tokens_seen": tokens,
                },
                step=step,
            )

        # ── Plots ─────────────────────────────────────────────────────────────
        loss_plot = plot_losses(track_tokens_seen, train_losses, val_losses)
        ppl_plot = plot_perplexity(train_losses, val_losses)
        mlflow.log_artifact(loss_plot, artifact_path="plots")
        mlflow.log_artifact(ppl_plot, artifact_path="plots")

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
