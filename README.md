# GPT-2 Implementation

A GPT-2 implementation from scratch in PyTorch.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the demo to verify the model forward pass:

```bash
python demo.py
```

Expected output shape: `torch.Size([2, 4, 50257])`

## Project Structure

```
GPT2Implementation/
├── gpt2/
│   ├── config.py       # Model configurations (GPT_CONFIG_124M, ...)
│   ├── layers.py       # LayerNorm, Gelu, FeedForward
│   ├── attention.py    # CausalAttention, MultiHeadAttention
│   ├── transformer.py  # Transformer block
│   └── model.py        # GPTModel
├── demo.py             # Forward pass demo
└── requirements.txt
```

## Model

The default config (`GPT_CONFIG_124M`) matches the original GPT-2 small:

| Parameter       | Value  |
|----------------|--------|
| Vocab size      | 50257  |
| Context length  | 1024   |
| Embedding dim   | 768    |
| Attention heads | 12     |
| Layers          | 12     |
| Dropout         | 0.1    |
