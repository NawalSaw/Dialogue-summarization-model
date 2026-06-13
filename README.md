# Transformer from Scratch

A production-grade PyTorch implementation of the Transformer architecture (Vaswani et al., 2017) for abstractive dialogue summarization on the SAMSum dataset.

## Overview

This project implements the full encoder-decoder Transformer from *"Attention Is All You Need"* with modern best practices (Pre-LayerNorm, label smoothing, Xavier initialization). The model is trained to generate concise summaries of conversational dialogues.

### Key Features

- **Full Transformer** — encoder-decoder with multi-head self-attention, cross-attention, and position-wise feed-forward networks
- **Pre-LayerNorm** — LayerNorm applied *before* each sub-layer for more stable training
- **Sinusoidal Positional Encoding** — fixed position encodings as in the original paper
- **BPE Tokenizer** — trained on the SAMSum corpus from scratch using HuggingFace `tokenizers`
- **Label Smoothing** — 0.1 cross-entropy smoothing for better generalization
- **Greedy Decoding** — autoregressive inference with `[BOS]`/`[EOS]` control tokens
- **TensorBoard Logging** — real-time training loss visualization
- **Checkpointing** — resume training from saved `.pt` checkpoints

## Architecture

```
Input (dialogue tokens)
    ↓
src_embedding (Token IDs → d_model) × √d_model
    ↓
src_positional_encoding (sin/cos)
    ↓
Encoder (×3 layers)
    ├── LayerNorm → Multi-Head Self-Attention → Residual + Dropout
    └── LayerNorm → Feed-Forward (d_model → d_ff → d_model) → Residual + Dropout
    ↓
Encoder output (memory)
    ↓
Decoder (×3 layers)
    ├── LayerNorm → Masked Self-Attention → Residual + Dropout
    ├── LayerNorm → Cross-Attention (Q=decoder, K/V=encoder) → Residual + Dropout
    └── LayerNorm → Feed-Forward → Residual + Dropout
    ↓
ProjectionLayer (d_model → vocab_size) + LogSoftmax
    ↓
Output (summary tokens)
```

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | 2.12.0 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| transformers | 5.10.2 |
| datasets | latest |
| tokenizers | latest |
| tensorboard | latest |

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/transformer-from-scratch.git
cd transformer-from-scratch

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Train the model on the SAMSum dialogue summarization dataset:

```bash
python src/train.py
```

The script will:
1. Load the SAMSum dataset from HuggingFace
2. Train a BPE tokenizer (saved to `models/tokenizer.json`)
3. Build the Transformer model
4. Train for the configured number of epochs
5. Log loss to TensorBoard (`runs/tmodel`)
6. Save checkpoints to `weights/`

### Monitoring with TensorBoard

```bash
tensorboard --logdir runs
```

### Resuming from a Checkpoint

Set `preload` in `src/config.py` to the desired epoch number:

```python
"preload": 5  # resumes from weights/tmodel_05.pt
```

### Inference

Use `greedy_decode()` from `src/train.py` to generate summaries:

```python
from train import greedy_decode, get_model
from config import get_config
from tokeniser.tokeniser import get_or_create_tokenizer
from datasets import load_dataset

config = get_config()
dataset = load_dataset("knkarthick/samsum")
tokenizer = get_or_create_tokenizer(config, dataset["train"], config["batch_size"])
model = get_model(config, tokenizer.get_vocab_size()).to("cuda")
model.load_state_dict(torch.load("weights/tmodel_final_09.pt")["model_state_dict"])

# Tokenize input
src_text = "Hi, how are you? I'm fine, thanks!"
src_ids = tokenizer.encode(src_text).ids
src_tensor = torch.tensor([tokenizer.token_to_id("[BOS]")] + src_ids + [tokenizer.token_to_id("[EOS]")]).unsqueeze(0)
src_mask = (src_tensor != tokenizer.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int()

# Generate summary
output_ids = greedy_decode(model, tokenizer, src_tensor.to("cuda"), src_mask.to("cuda"), max_len=1400, device="cuda")
print(tokenizer.decode(output_ids.detach().cpu().numpy()))
```

## Configuration

All hyperparameters are defined in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `lr` | 1e-4 | Adam learning rate |
| `d_model` | 512 | Embedding dimension |
| `heads_num` | 4 | Number of attention heads |
| `dropout` | 0.2 | Dropout probability |
| `num_layers` | 3 | Number of encoder/decoder layers |
| `seq_len` | 1400 | Maximum sequence length |
| `batch_size` | 8 | Training batch size |
| `num_epochs` | 10 | Number of training epochs |
| `label_smoothing` | 0.1 | Cross-entropy label smoothing |

## Project Structure

```
├── src/
│   ├── attention/
│   │   └── multi_head_attention.py    # Scaled dot-product multi-head attention
│   ├── decoder/
│   │   ├── decoder.py                 # Decoder stack
│   │   └── decoder_layer.py           # Single decoder layer (masked self-attn + cross-attn + FFN)
│   ├── embedding/
│   │   └── embedding.py               # Token embedding with √d_model scaling
│   ├── encoder/
│   │   ├── encoder.py                 # Encoder stack
│   │   └── encoder_layer.py           # Single encoder layer (self-attn + FFN)
│   ├── feed_forward_net/
│   │   ├── feed_forward.py            # Position-wise FFN
│   │   └── projection_layer.py        # Output projection d_model → vocab_size
│   ├── normalization/
│   │   └── layer_normalization.py      # Custom LayerNorm
│   ├── positional_encoding/
│   │   └── sinosuidal.py              # Sinusoidal positional encoding
│   ├── tokeniser/
│   │   └── tokeniser.py               # BPE tokenizer training/loading
│   ├── config.py                      # Hyperparameters and paths
│   ├── dataset.py                     # SAMSum dataset + causal mask
│   ├── data.ipynb                     # Exploratory notebook
│   ├── train.py                       # Training loop and inference
│   └── transformer.py                 # Top-level Transformer module
├── models/
│   └── tokenizer.json                 # Pre-trained BPE tokenizer
├── requirements.txt
└── README.md
```

## Dataset

The model is trained on [SAMSum](https://huggingface.co/datasets/knkarthick/samsum), a collection of ~16,000 messenger-style conversations with summaries. Each example contains:

- `dialogue`: the full conversation
- `summary`: a human-written abstractive summary

## Implementation Details

- **Weight Initialization**: Xavier uniform for all weight matrices with `dim() > 1`
- **Label Smoothing**: 0.1 smooths the target distribution to reduce overconfidence
- **Padding Masking**: `[PAD]` tokens are masked in attention and ignored in loss computation
- **Causal Masking**: the decoder uses a lower-triangular mask to prevent attending to future tokens
- **Pre-LayerNorm**: normalization is applied before each sub-layer (unlike the original Post-LN), improving training stability

## License

MIT
