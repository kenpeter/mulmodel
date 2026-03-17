# mulmodel

Train a 202M param transformer on Codeforces chain-of-thought data.

## Setup

```bash
conda create -n mulmodel python=3.12 -y
conda activate mulmodel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Recommended: reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Train

```bash
python train.py --epochs 10 --lr 3e-4
```

For faster feedback (step increments every batch instead of every 8 batches):

```bash
python train.py --grad-accum 1
```

Resume:
```bash
python train.py --epochs 10 --resume checkpoints/latest.pt
```

Monitor:
```bash
tail -f run.log
```

### GPU Memory

If you encounter OOM, reduce context length or use gradient accumulation:

```bash
python train.py --batch-size 8 --grad-accum 8   # effective batch = 64
python train.py --batch-size 4 --grad-accum 16  # lower memory, same effective batch
python train.py --ctx-len 256   # shorter sequences = less memory
```

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.

## Options

| Flag | Default | |
|------|---------|-|
| `--epochs` | 10 | passes over data |
| `--lr` | 3e-4 | peak learning rate |
| `--warmup-steps` | 0 | linear warmup before cosine decay |
| `--batch-size` | 8 | per-step batch size |
| `--grad-accum` | 8 | gradient accumulation steps (effective batch = batch_size × grad_accum) |
| `--log-every` | 1 | print loss every N steps |
| `--time-limit` | — | stop after N seconds |
| `--ctx-len` | 512 | context length (lower = less memory) |

## Model

- 202M params, emb_dim=1024, n_heads=16, n_layers=16, context=512
- Byte-level tokenizer (vocab=256)
- SageAttention (falls back to standard if not installed)
