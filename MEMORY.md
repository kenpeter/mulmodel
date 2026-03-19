# MEMORY.md - Long-term Memory

## Project: mulmodel - BigModel Codeforces Training

### Architecture
- BigModel: 202M params (byte) / 253M params (tiktoken), emb_dim=1024, n_layers=16, n_heads=16, n_blocks=4, drop_rate=0.1, AttnRes (arxiv 2603.15031), RMSNorm, SageAttention
- ctx_len=384 is sweet spot: fits in VRAM, fast enough (~270 steps/5min)
- VRAM: 11.6 GB available

### Checkpoints (as of 2026-03-19)
| File | Format | Vocab | Ctx | Step | Loss |
|------|--------|-------|-----|------|------|
| `latest.pt` | tiktoken [CODE] | 50257 | 384 | 500 | 1.70 |
| `byte_cot_final.pt` | byte CoT | 256 | 384 | 1800 | 0.86 |
| `code_c384_step750.pt` | tiktoken [CODE] | 50257 | 384 | 250 | 3.10 |
| `tiktoken_backup.pt` | tiktoken [CODE] | 50257 | 128 | 1600 | 1.62 |

### Key Findings (2026-03-19)

**THE [CODE] FORMAT WORKS.** Greedy decoding on ctx_len=128 [CODE] model produces structured C++.

**Original 80% Result:** Byte-level CoT model got 80% pass_rate at loss 2.03. The byte-level `bytes(ids).decode("utf-8", errors="replace")` silently replaced invalid UTF-8 bytes with `\ufffd`, but g++ could still compile it because the corrupted bytes were in non-critical positions.

**Tiktoken spacing:** Tokenization introduces spacing like `vector<int> a` → `vector< int > a`. Simple string replacements in postprocess_code() handle common cases.

**Critical eval bug:** `tokenizer.decode()` replaces invalid bytes differently than `bytes().decode("utf-8", errors="replace")`. Current eval uses tiktoken decode which hides corruption.

### Data
- `data/code/codeforces_cots/` — 47,780 entries, description + CoT + code (arrow format)
- Training format: `description.strip() + "\n[CODE]\n" + extracted_code` (for [CODE] format)
- Training format: `description + "\n" + generation` (for CoT format)
- Code extraction: extract_code() from generation using ```cpp blocks

### Training Commands
```bash
# Resume [CODE] tiktoken training:
python train.py --time-limit 300 --tokenizer tiktoken --format code --ctx-len 384 --batch-size 4 --grad-accum 8 --resume

# Resume byte-level CoT training:
python train.py --time-limit 300 --tokenizer byte --format cot --ctx-len 384 --batch-size 2 --grad-accum 16 --resume

# Eval:
python eval_cf_real.py --tokenizer tiktoken --ctx-len 384
```

### Files
- `train.py` — supports `--format auto|cot|code` and `--tokenizer byte|tiktoken`
- `eval_cf_real.py` — active eval script with postprocessing
- `start.py` — uses opencode
- `checkpoints/` — model checkpoints

### Lessons Learned
1. [CODE] format is correct direction — greedy decoding confirms C++ generation works
2. Need ~2000+ steps on [CODE] format before good code quality
3. ctx_len=384 is max viable for tiktoken training
4. Byte-level CoT only generates explanations, not code blocks
5. Perplexity is NOT a good proxy for code generation quality
