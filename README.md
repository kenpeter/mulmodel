# mulmodel

Train a GPT-style language model on code + reasoning data.

---

## Model

GPT-124M architecture:
- `emb_dim`: 768
- `n_heads`: 12
- `n_layers`: 12
- `context_length`: 256
- `vocab_size`: 50257 (GPT-2 tokenizer)

---

## Setup

```bash
pip install torch numpy tiktoken
```

---

## Train

From scratch:
```bash
python -m big_model.pretrain --epochs 1000 > run.log 2>&1
```

Resume from checkpoint:
```bash
python -m big_model.pretrain --epochs 1000 --resume checkpoints/latest_checkpoint.pt > run.log 2>&1
```

Monitor:
```bash
tail -f run.log
```

---

## Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 10 | Training epochs |
| `--lr FLOAT` | 5e-5 | Learning rate |
| `--batch-size N` | 2 | Micro batch size |
| `--effective-batch-size N` | 32 | Effective batch (grad accumulation) |
| `--save-every N` | 5 | Checkpoint every N epochs |
| `--checkpoint-dir PATH` | `checkpoints` | Where to save checkpoints |
| `--time-limit N` | — | Stop after N seconds |

---

## Structure

```
mulmodel/
├── big_model/
│   ├── pretrain.py       # training script (entry point)
│   ├── transformer.py    # model architecture
│   ├── dataloader.py     # data loading
│   └── ...
├── data/                 # training data
└── checkpoints/          # saved checkpoints
```

---

## Data

Training data lives in `data/`. The pretrain script loads it automatically.

---

## Goal

Get `train_loss` below 0.5. Current best checkpoint is in `big_model_data/`.
