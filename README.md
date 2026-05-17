# GPT-2 Implementation

A GPT-2 (124M) implementation from scratch in PyTorch, with training on custom text and MLflow experiment tracking.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Forward pass demo
Verify the model architecture with a quick forward pass:
```bash
python demo.py
# Expected output: torch.Size([2, 4, 50257])
```

### Training with MLflow tracking
```bash
python train.py                        # downloads dataset automatically, trains for 10 epochs
python train.py --epochs 20            # custom epoch count
python train.py --data my_text.txt     # use your own text file
python train.py --lr 3e-4 --batch-size 4
```

### Viewing results
```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

MLflow tracks per-step train/val loss, perplexity, tokens seen, generated text samples, loss curves, and model weights across runs.

## Project Structure

```
GPT2Implementation/
├── gpt2/
│   ├── config.py       # Model config (GPT_CONFIG_124M)
│   ├── layers.py       # LayerNorm, GELU, FeedForward
│   ├── attention.py    # CausalAttention, MultiHeadAttention
│   ├── transformer.py  # TransformerBlock
│   ├── model.py        # GPTModel
│   ├── data.py         # GPTDataset, create_dataloader
│   ├── training.py     # train_model_simple, evaluate_model, generate, ...
│   └── __init__.py
├── train.py            # Training script with MLflow integration
├── demo.py             # Forward pass sanity check
└── requirements.txt
```

## Model

Default config (`GPT_CONFIG_124M`) matches GPT-2 small:

| Parameter       | Value   |
|----------------|---------|
| Parameters      | 124M    |
| Vocab size      | 50,257  |
| Context length  | 1,024   |
| Embedding dim   | 768     |
| Attention heads | 12      |
| Layers          | 12      |
| Dropout         | 0.1     |

## Training details

- **Dataset:** any plain text file; defaults to *The Verdict* (Edith Wharton, public domain)
- **Objective:** next-token prediction (cross-entropy loss)
- **Optimizer:** AdamW (`lr=4e-4`, `weight_decay=0.1`)
- **Logged metrics:** train loss, val loss, train perplexity, val perplexity, tokens seen
- **Artifacts:** loss curve plot, perplexity curve plot, model weights (`.pt`)