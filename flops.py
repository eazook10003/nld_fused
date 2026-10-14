"""FLOP counting from the real per-forward shapes.

1 multiply-add = 2 FLOPs. Counted: all linear projections (q/k/v/o, gate/up/down), the LM head on every position
where the loop computes it (all positions of a block forward, last position only at prefill), the LoRA delta on
positions where the adapter is active, and attention QK^T + PV. Not counted: embeddings, RMSNorm, RoPE, softmax,
SiLU, residual adds (all well under 1% for this model).
Attention is counted dense: every query row against the full key length given to the kernel (cache + all input rows),
which is what a masked SDPA kernel computes. Discarded candidate rows are counted in full.
"""

CFG = dict(hidden=4096, layers=34, heads=32, kv_heads=8, head_dim=128, inter=14336, vocab=131072, lora_r=128)
H, L, NH, KVH, HD, I, V, R = (CFG[k] for k in ["hidden", "layers", "heads", "kv_heads", "head_dim", "inter", "vocab", "lora_r"])
LIN_PER_POS = 2 * L * (H * H + 2 * H * KVH * HD + H * H + 3 * H * I)          # FLOPs per position, all layers
HEAD_PER_POS = 2 * H * V                                                        # LM head per position
LORA_PER_POS = 2 * L * (H * R + R * H)                                          # o_proj LoRA per active position
ATT_PER_PAIR = 2 * L * (2 * NH * HD)                                            # QK^T + PV per (query,key) pair, all layers


def fwd(n_pos, kv_len, allowed_pairs, lora_pos, head_pos):
    dense_pairs = n_pos * kv_len
    return dict(linear=n_pos * LIN_PER_POS, head=head_pos * HEAD_PER_POS, lora=lora_pos * LORA_PER_POS,
                att_dense=dense_pairs * ATT_PER_PAIR, att_allowed=allowed_pairs * ATT_PER_PAIR)


def add(tot, f, cat):
    for k, v in f.items():
        tot[k] = tot.get(k, 0) + v
    tot["cat_" + cat] = tot.get("cat_" + cat, 0) + f["linear"] + f["head"] + f["lora"] + f["att_dense"]


def prefill_flops(prompt_len):
    tot = {}
    add(tot, fwd(prompt_len, prompt_len, prompt_len * (prompt_len + 1) // 2, 0, 1), "prefill")
    return tot["cat_prefill"]


def upstream_parts(blocks):
    """Baseline decode FLOPs per category from spec_trace blocks: one draft (bidirectional, LoRA on) + one verify (causal) per block."""
    tot = {}
    for b in blocks:
        c = b["cache_len"]
        add(tot, fwd(32, c + 32, 32 * (c + 32), 32, 32), "draft"); add(tot, fwd(32, c + 32, sum(c + p + 1 for p in range(32)), 0, 32), "verify")
    return {k[4:]: v for k, v in tot.items() if k.startswith("cat_")}


def fused_parts(records):
    """Fused decode FLOPs per category from linear_spec_generate_fused(record=...)."""
    tot = {}
    for rec in records:
        if rec["kind"] == "prefill": continue
        elif rec["kind"] == "draft": c = rec["cache_len"]; add(tot, fwd(32, c + 32, 32 * (c + 32), 32, 32), "fallback_draft")
        else:
            c = rec["cache_len"]; n_pos = rec["n_pos"]
            add(tot, fwd(32, c + n_pos, sum(c + p + 1 for p in range(32)), 0, 32), "verify_rows")
            for (m, v) in rec["cands"]:
                used = rec.get("hit", False) and (m == rec["matched"])
                add(tot, fwd(32, c + n_pos, 32 * (c + m + 1 + 32), 32, 32), "cand_used" if used else "cand_discarded")
            if rec.get("fa"):  # full-accept candidate: 33 rows, see cache + verify cols 0..30 + their own 33 cols
                # used = a prepared draft was taken AND the block passed whole (with other candidates beside it, a hit can be theirs)
                add(tot, fwd(33, c + n_pos, 33 * (c + 31 + 33), 33, 33), "fa_used" if rec.get("hit", False) and rec.get("matched") == 31 else "fa_discarded")
    return {k[4:]: v for k, v in tot.items() if k.startswith("cat_")}
