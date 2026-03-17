import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from sageattention import sageattn

    HAS_SAGE = True
except ImportError:
    HAS_SAGE = False

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "vocab_size": 256,  # byte-level tokenizer
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 16,
    "n_blocks": 4,  # AttnRes block groups (4 groups of 4 layers each)
    "drop_rate": 0.1,
}

# ── RMSNorm ───────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale


# ── Block AttnRes ─────────────────────────────────────────────────────────────


def block_attn_res(blocks, partial_block, query, key_norm):
    """
    Softmax attention over block representations (depth-wise).

    blocks:        list of N tensors [B, T, D] — completed block summaries
    partial_block: tensor [B, T, D]            — current intra-block partial sum
    query:         tensor [D]                  — learned pseudo-query (w_l)
    key_norm:      RMSNorm                     — normalises keys before dot-product

    Returns aggregated hidden state [B, T, D].
    """
    V = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
    K = key_norm(V)  # [N+1, B, T, D]
    logits = torch.einsum("d, nbtd -> nbt", query, K)  # [N+1, B, T]
    weights = logits.softmax(dim=0)  # [N+1, B, T]
    return torch.einsum("nbt, nbtd -> btd", weights, V)  # [B, T, D]


# ── Attention ─────────────────────────────────────────────────────────────────


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, drop_rate=0.0):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.drop_rate = drop_rate
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: [B, T, n_heads, head_dim]

        if HAS_SAGE and x.dtype in (torch.float16, torch.bfloat16):
            out = sageattn(q, k, v, tensor_layout="NHD", is_causal=True)
        else:
            q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2)

        return self.proj(out.reshape(B, T, C))


# ── AttnRes Transformer Block ─────────────────────────────────────────────────


class AttnResBlock(nn.Module):
    """
    Single transformer block (attention + FFN) with Block Attention Residuals.

    Replaces standard additive residuals with softmax attention over preceding
    block-level representations, following AttnRes (arxiv 2603.15031).

    layer_idx:    global index of this block (0-based)
    n_per_block:  how many transformer blocks form one AttnRes block group
    """

    def __init__(self, emb_dim, n_heads, drop_rate, layer_idx, n_per_block):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_block_start = (layer_idx % n_per_block == 0) and (layer_idx > 0)

        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, n_heads, drop_rate)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim, bias=False),
            nn.Dropout(drop_rate),
        )

        # Learned pseudo-queries (one per sub-layer)
        self.q_attn = nn.Parameter(torch.zeros(emb_dim))
        self.q_mlp = nn.Parameter(torch.zeros(emb_dim))
        nn.init.normal_(self.q_attn, std=0.02)
        nn.init.normal_(self.q_mlp, std=0.02)

        # RMSNorm applied to keys before attention weight computation
        self.k_norm_attn = RMSNorm(emb_dim)
        self.k_norm_mlp = RMSNorm(emb_dim)

    def forward(self, blocks, partial_block):
        """
        blocks:        list of completed block-level representations [B, T, D]
        partial_block: current intra-block running sum [B, T, D]

        Returns (blocks, partial_block).
        """
        # Compute attention input via depth-wise AttnRes (BEFORE boundary transition)
        h = block_attn_res(blocks, partial_block, self.q_attn, self.k_norm_attn)

        # At block boundaries: seal off the previous block, start fresh
        if self.is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None

        # Self-attention sub-layer
        attn_out = self.attn(self.ln1(h))
        partial_block = (
            (partial_block + attn_out) if partial_block is not None else attn_out
        )

        # Compute FFN input via depth-wise AttnRes
        h = block_attn_res(blocks, partial_block, self.q_mlp, self.k_norm_mlp)

        # FFN sub-layer
        mlp_out = self.ffn(self.ln2(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


# ── Model ─────────────────────────────────────────────────────────────────────


class BigModel(nn.Module):
    def __init__(self, cfg=MODEL_CONFIG):
        super().__init__()
        V, C, T = cfg["vocab_size"], cfg["emb_dim"], cfg["context_length"]
        L = cfg["n_layers"]
        N = cfg["n_blocks"]
        assert L % N == 0, f"n_layers ({L}) must be divisible by n_blocks ({N})"
        n_per_block = L // N

        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(T, C)
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(
            [
                AttnResBlock(C, cfg["n_heads"], cfg["drop_rate"], idx, n_per_block)
                for idx in range(L)
            ]
        )
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, V, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        blocks = [x]
        partial = x

        for block in self.blocks:
            blocks, partial = block(blocks, partial)

        return self.lm_head(self.ln_f(partial))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
