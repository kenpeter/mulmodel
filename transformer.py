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
    "vocab_size":     256,   # byte-level tokenizer
    "context_length": 512,
    "emb_dim":        1024,
    "n_heads":        16,
    "n_layers":       16,
    "drop_rate":      0.1,
}

# ── Attention ─────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, drop_rate=0.0):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        self.drop_rate = drop_rate
        self.qkv  = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: [B, T, n_heads, head_dim]

        if HAS_SAGE and x.dtype in (torch.float16, torch.bfloat16):
            # sageattn expects [B, T, n_heads, head_dim] with layout="NHD"
            out = sageattn(q, k, v, tensor_layout="NHD", is_causal=True)
        else:
            q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn  = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask  = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn  = attn.masked_fill(mask, float("-inf"))
            attn  = F.softmax(attn, dim=-1)
            out   = torch.matmul(attn, v).transpose(1, 2)

        return self.proj(out.reshape(B, T, C))

# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, emb_dim, n_heads, drop_rate):
        super().__init__()
        self.ln1  = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, n_heads, drop_rate)
        self.ln2  = nn.LayerNorm(emb_dim)
        self.ffn  = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim, bias=False),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# ── Model ─────────────────────────────────────────────────────────────────────

class BigModel(nn.Module):
    def __init__(self, cfg=MODEL_CONFIG):
        super().__init__()
        V, C, T = cfg["vocab_size"], cfg["emb_dim"], cfg["context_length"]

        self.tok_emb = nn.Embedding(V, C)
        self.pos_emb = nn.Embedding(T, C)
        self.drop    = nn.Dropout(cfg["drop_rate"])
        self.blocks  = nn.ModuleList([
            Block(C, cfg["n_heads"], cfg["drop_rate"])
            for _ in range(cfg["n_layers"])
        ])
        self.ln_f    = nn.LayerNorm(C)
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
        x   = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
