# Attention Residuals (AttnRes)

**ArXiv**: 2603.15031
**Authors**: Kimi Team (Moonshot AI)
**Date**: March 2026

## Key Idea

Replace fixed additive residual connections (`h_l = h_{l-1} + f(h_{l-1})`) with learned
softmax attention over all preceding layer outputs. Each layer selectively aggregates prior
representations using a single learned d-dimensional pseudo-query, enabling content-aware
depth-wise information retrieval — analogous to how attention replaced RNNs over sequences.

## Method

**Full AttnRes** — each layer l computes:

```
h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
alpha_{i->l} = softmax( q_l^T * RMSNorm(k_i) )   # q_l = learned w_l in R^d
k_i = v_i = { h_1               if i=0
            { f_i(h_i)           otherwise
```

**Block AttnRes** (scalable variant) — groups L layers into N blocks:
- Within each block: sum layer outputs into a block representation `b_n`
- Across blocks: each layer attends only over N block summaries (not all L outputs)
- Memory: O(Ld) → O(Nd); paper recommends N≈8

**Implementation** (from paper pseudocode):
- Each layer has 2 pseudo-queries (q_attn, q_mlp) — one before attention, one before FFN
- At block boundaries: current partial sum becomes a new block; partial resets to None
- `block_attn_res(blocks, partial, query, key_norm)` = softmax(q @ RMSNorm(V)) @ V

## Results

- Block AttnRes (N=8, S=4) ≡ baseline trained with **1.25× more compute**
- Consistent gains across all 25 architecture configurations tested
- 48B Kimi Linear model pre-trained on 1.4T tokens: improves all downstream benchmarks
- Overhead: <2% inference latency, marginal training cost
- Mitigates PreNorm dilution: hidden state magnitudes stay bounded across depth

## Relevance to This Project

**Direct implementation in `transformer.py`:**

The standard `Block` class has been replaced by `AttnResBlock`. Key changes:

| Component | Before | After |
|-----------|--------|-------|
| `Block.forward` | `x = x + attn(x); x = x + ffn(x)` | AttnRes aggregates from block history |
| `BigModel.forward` | `for b in blocks: x = b(x)` | `blocks=[emb]; for b: blocks,p = b(blocks,p)` |
| New params | — | `q_attn`, `q_mlp` [D] + 2×RMSNorm per layer |

**Configuration for our 16-layer model:**
- `n_blocks=8` → `n_per_block=2` (2 transformer layers per block group)
- Block boundaries at layers 2, 4, 6, 8, 10, 12, 14
- Each layer attends over up to 9 tensors (8 blocks + partial)

**Expected benefit:** The model currently shows PreNorm dilution symptoms (layer pruning
tests would likely pass easily). AttnRes should produce more uniform gradient flow across
all 16 layers and allow later layers to selectively retrieve early-layer features, which
is particularly useful for code generation where both low-level syntax (early layers) and
high-level structure (later layers) matter.

**Files changed:** `transformer.py` (full rewrite of Block + BigModel)

**Breaking change:** Old checkpoints from the additive-residual model are incompatible.
Training must restart from scratch.

## Key References

- He et al. (2015) — ResNet residual connections
- Xiong et al. (2020) — PreNorm (standard in modern LLMs)
- Li et al. (2026) — SiameseNorm: analysis of PreNorm dilution
- Zhang et al. (2025) — Kimi Linear architecture (48B MoE model)
- Milakov & Gimelshein (2018) — Online softmax for inference efficiency
