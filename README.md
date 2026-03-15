# mulmodel

Train a 202M param transformer on Codeforces chain-of-thought data.

## Setup

```bash
conda create -n mulmodel python=3.12 -y
conda activate mulmodel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 10 --warmup-steps 100 --lr 3e-4
```

Resume:
```bash
python train.py --epochs 10 --resume checkpoints/latest.pt
```

Monitor:
```bash
tail -f run.log
```

## Options

| Flag | Default | |
|------|---------|-|
| `--epochs` | 10 | passes over data |
| `--lr` | 3e-4 | peak learning rate |
| `--warmup-steps` | 1000 | linear warmup before cosine decay |
| `--batch-size` | 8 | |
| `--resume` | — | path to checkpoint |
| `--time-limit` | — | stop after N seconds |

## Model

- 202M params, emb_dim=1024, n_heads=16, n_layers=16, context=512
- Byte-level tokenizer (vocab=256)
- SageAttention (falls back to standard if not installed)
