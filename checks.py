#!/usr/bin/env python3
"""Correctness checks for the full-accept candidate and the joint (M, full-accept) choice. Run once before timing.
 T1 structure (synthetic; candidates m = 3, 10, 25 + the full-accept candidate):
    verify rows causal and blind to every candidate column; each candidate sees cache + verify cols 0..m + its own 32 cols;
    full-accept rows see cache + verify cols 0..30 + their own 33 cols; no candidate sees another; positions; seeds / MASK;
    LoRA gate 1 on candidate rows, 0 on verify rows
 T2 real states (first blocks of 3 GSM8K problems):
    leak, bit-exact: changing the full-accept seed leaves verify and candidate rows unchanged; changing a candidate seed
    leaves verify and full-accept rows unchanged; changing verify cell 31 leaves the full-accept rows unchanged while
    changing cell 10 does not; the C/32-row spec path equals fused_forward
    reference: full-accept rows vs verify -> crop to cache_len + 31 -> unmasked draft forward of [cell 31, MASK x 32]
    (argmax agreement and max |logit diff|; kernel numerics, not bit-exact)
 T3 generation (first 10 GSM8K, cap 1024):
    (a) a cost table where the full-accept option never wins gives exactly the dyn outputs, nfe and dyn_log
    (b) with a realistic table: bookkeeping invariants, and full-accept candidates appear next to M > 0 candidates
The cost tables here are fixed numbers; only structure and invariants are checked.
Exit code 1 if T1, the bit-exact parts of T2, or T3 fail.

Usage: python checks.py
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_fused import (linear_spec_generate_fused, fused_forward, fused_forward_specs, build_inputs, make_specs,
                        baseline_draft, baseline_verify, draft_stats, accept, set_diffusion_lm, set_adapters)
from spec_trace import _upstream_module
from common import BL, load_model, load_prompts, chat_ids
from transformers.cache_utils import DynamicCache

DYN = (39.6, 40.1, {1: 46.4, 2: 45.9, 3: 47.3, 4: 53.1, 5: 56.9, 6: 69.1})
FA_NEVER = {k: 1e9 for k in [0, 1, 2, 3, 4, 5, 6]}
FA_REAL = {0: 46.4, 1: 47.0, 2: 47.5, 3: 53.1, 4: 56.9, 5: 62.0, 6: 72.0}
CANDS = [(3, 101), (10, 102), (25, 103)]


def t1_structure():
    dev = "cuda"; cache_len = 50; mask_id = 7
    block = torch.arange(100, 132, device=dev).unsqueeze(0)
    specs = make_specs("C", BL, CANDS, block) + make_specs("P", BL + 1, [(BL - 1, None)], block)
    ids, pos, mask, gate, spans = build_inputs(block, specs, cache_len, mask_id, dev)
    mk = mask[0, 0]; L = ids.shape[1]; ok = True
    ok &= bool((~mk[:BL, cache_len + BL:]).all())                                                    # verify blind to candidates
    ok &= bool((mk[:BL, cache_len:cache_len + BL] == torch.tril(torch.ones(BL, BL, dtype=torch.bool, device=dev))).all()) and bool(mk[:BL, :cache_len].all())
    ok &= bool((gate[0, :BL, 0] == 0).all())
    for s, (r0, r1) in zip(specs, spans):
        exp = torch.zeros(cache_len + L, dtype=torch.bool, device=dev); exp[:cache_len] = True
        exp[cache_len:cache_len + s["vis"]] = True; exp[cache_len + r0:cache_len + r1] = True
        vis_ok = bool((mk[r0:r1] == exp).all())
        pos_ok = bool((pos[0, r0:r1] == torch.arange(cache_len + s["pos_rel"], cache_len + s["pos_rel"] + s["rows"], device=dev)).all())
        seed_ok = int(ids[0, r0]) == s["seed"] and bool((ids[0, r0 + 1:r1] == mask_id).all())
        gate_ok = bool((gate[0, r0:r1, 0] == 1).all())
        exp_vis = s["m"] + 1 if s["kind"] == "C" else BL - 1
        print(f"T1 {s['kind']} m={s['m']} rows {r1 - r0}: sees verify 0..{s['vis'] - 1} (expected 0..{exp_vis - 1}) visibility {vis_ok} positions {pos_ok} seed/MASK {seed_ok} gate {gate_ok}", flush=True)
        ok &= vis_ok and pos_ok and seed_ok and gate_ok and s["vis"] == exp_vis
    print(f"T1 rows {L} (=32 + 3x32 + 33): {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


@torch.no_grad()
def real_states(tok, model, n_prompts=3, blocks_per_prompt=3):
    """Yield (pkv, cache_len, block) after a baseline draft, at the first blocks of a few prompts (baseline trajectory)."""
    _crop = _upstream_module(model)._crop_dynamic_cache; mask_id = model.config.mask_token_id
    for row in load_prompts("gsm8k", n_prompts):
        p = chat_ids(tok, row["user"])
        set_diffusion_lm(model, False); set_adapters(model, False)
        enc = model.encoder(input_ids=p, past_key_values=DynamicCache(), use_cache=True, use_causal_mask=True); pkv = enc.past_key_values
        nt = model.diffusion_head(enc.last_hidden_state[:, -1:, :]).squeeze(1).argmax(dim=-1, keepdim=True)
        for bi in range(blocks_per_prompt):
            cache_len = pkv.get_seq_length()
            block = torch.full((1, BL), mask_id, dtype=torch.long, device="cuda"); block[0, 0] = nt.item()
            dt, conf, _, _ = draft_stats(baseline_draft(model, pkv, block), 8); block[0, 1:] = dt[1:]
            yield row["id"], bi, pkv, cache_len, block
            _crop(pkv, cache_len)
            pkv, vlog = baseline_verify(model, pkv, block)
            ar, _, acc = accept(vlog, block)
            _crop(pkv, cache_len + acc); nt = ar[:, acc - 1:acc]


@torch.no_grad()
def t2(tok, model):
    _crop = _upstream_module(model)._crop_dynamic_cache; mask_id = model.config.mask_token_id; ok = True
    agv, agc, agf, agr, maxd = [], [], [], [], []
    for pid, bi, pkv, cache_len, block in real_states(tok, model):
        cs = make_specs("C", BL, CANDS, block); fs = make_specs("P", BL + 1, [(BL - 1, None)], block)
        other = (int(block[0, BL - 1]) + 17) % 1000 + 1000
        _, v0, c0 = fused_forward_specs(model, pkv, block, cs + fs, cache_len, mask_id); _crop(pkv, cache_len)
        _, v1, c1 = fused_forward_specs(model, pkv, block, cs + [dict(fs[0], seed=other)], cache_len, mask_id); _crop(pkv, cache_len)
        _, v2, c2 = fused_forward_specs(model, pkv, block, [dict(cs[0], seed=other)] + cs[1:] + fs, cache_len, mask_id); _crop(pkv, cache_len)
        a = torch.equal(v0, v1) and all(torch.equal(c0[i], c1[i]) for i in range(3))                 # full-accept seed leaks nowhere
        b = torch.equal(v0, v2) and torch.equal(c0[3], c2[3]) and all(torch.equal(c0[i], c2[i]) for i in (1, 2))  # candidate seed leaks nowhere
        _, _, fo = fused_forward_specs(model, pkv, block, fs, cache_len, mask_id); _crop(pkv, cache_len)   # full-accept candidate alone
        b31 = block.clone(); b31[0, BL - 1] = other                                                     # cell 31: not visible to the FA rows
        _, _, f31 = fused_forward_specs(model, pkv, b31, fs, cache_len, mask_id); _crop(pkv, cache_len)
        b10 = block.clone(); b10[0, 10] = other                                                         # cell 10: visible to the FA rows
        _, _, f10 = fused_forward_specs(model, pkv, b10, fs, cache_len, mask_id); _crop(pkv, cache_len)
        c = torch.equal(fo[0], f31[0]) and not torch.equal(fo[0], f10[0])
        _, vn, cn = fused_forward_specs(model, pkv, block, cs, cache_len, mask_id); _crop(pkv, cache_len)  # same candidates, no FA
        _, vo, co = fused_forward(model, pkv, block, CANDS, cache_len, mask_id); _crop(pkv, cache_len)
        same_old = torch.equal(vn, vo) and all(torch.equal(cn[i], co[i]) for i in range(3))               # C/32 spec path == fused_forward
        agv.append((v0.argmax(-1) == vn.argmax(-1)).float().mean().item())
        agc.append(sum((c0[i].argmax(-1) == cn[i].argmax(-1)).float().mean().item() for i in range(3)) / 3)
        agf.append((c0[3].argmax(-1) == fo[0].argmax(-1)).float().mean().item())
        # reference: baseline verify -> keep positions 0..30 -> unmasked draft forward of [cell 31, MASK x 32]
        pkv, _ = baseline_verify(model, pkv, block); _crop(pkv, cache_len + BL - 1)
        ref_in = torch.full((1, BL + 1), mask_id, dtype=torch.long, device="cuda"); ref_in[0, 0] = block[0, BL - 1]
        ref = baseline_draft(model, pkv, ref_in); _crop(pkv, cache_len)
        agr.append((ref[0, 1:].argmax(-1) == fo[0][0, 1:].argmax(-1)).float().mean().item())
        maxd.append((ref[0, 1:].float() - fo[0][0, 1:].float()).abs().max().item())
        print(f"T2 {pid} blk{bi} cache {cache_len}: FA seed leaks nowhere {a} | cand seed leaks nowhere {b} | FA blind to cell 31, sees cell 10 {c} "
              f"| C/32 spec path == fused_forward {same_old} || FA vs reference: argmax agree {agr[-1]:.3f} max|dlogit| {maxd[-1]:.3f}", flush=True)
        ok &= a and b and c and same_old
    print(f"T2 argmax agreement with vs without the FA rows (shape effect only): verify {sum(agv)/len(agv):.4f} cands {sum(agc)/len(agc):.4f} "
          f"FA mixed vs alone {sum(agf)/len(agf):.4f}", flush=True)
    print(f"T2 FA rows vs sequential reference: argmax agreement {sum(agr)/len(agr):.3f}, max |dlogit| {max(maxd):.3f} (masked kernel vs unmasked draft)", flush=True)
    print(f"T2 bit-exact parts: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def t3(tok, model, n=10):
    eos = tok.eos_token_id; ok_a = ok_b = True; mpos = nfa = fh = 0; same = 0
    for row in load_prompts("gsm8k", n):
        p = chat_ids(tok, row["user"])
        o0, s0 = linear_spec_generate_fused(model, p, max_new_tokens=1024, block_length=BL, eos_token_id=eos, timing=False, dyn=DYN)
        o1, s1 = linear_spec_generate_fused(model, p, max_new_tokens=1024, block_length=BL, eos_token_id=eos, timing=False, dyn=DYN + (FA_NEVER,))
        a = torch.equal(o0, o1) and s0["nfe"] == s1["nfe"] and s0["dyn_log"] == s1["dyn_log"] and sum(s1["fa_sel"]) == 0
        o2, s2 = linear_spec_generate_fused(model, p, max_new_tokens=1024, block_length=BL, eos_token_id=eos, timing=False, dyn=DYN + (FA_REAL,))
        nb = len(s2["dyn_log"])
        inv1 = s2["hits"] + s2["miss_pred"] + s2["miss_full"] == nb - 1  # the last block (EOS / cap) ends the loop before its bookkeeping
        inv2 = all(x[1] == 1 for x in s2["fa_log"] if x[2] == 1) and s2["fa_hits"] == sum(1 for x in s2["fa_log"] if x[2] == 1)
        inv3 = len(s2["fa_sel"]) == nb and sum(s2["fa_sel"]) == s2["fa_blocks"]
        inv4 = s2["nfe"] == 1 + nb + s2["draft_fwd"]
        mpos += sum(1 for x in s2["fa_log"] if x[4] > 0); nfa += s2["fa_blocks"]; fh += s2["fa_hits"]; same += int(torch.equal(o0, o2))
        print(f"T3 {row['id']}: (a) never-FA == dyn {a} | (b) blocks {nb} FA blocks {s2['fa_blocks']} (with M>0: {sum(1 for x in s2['fa_log'] if x[4] > 0)}) "
              f"FA hits {s2['fa_hits']} hits {s2['hits']} draft {s2['draft_fwd']} | invariants {inv1} {inv2} {inv3} {inv4} | same output as dyn {torch.equal(o0, o2)}", flush=True)
        ok_a &= a; ok_b &= inv1 and inv2 and inv3 and inv4
    print(f"T3 total: (a) {'PASS' if ok_a else 'FAIL'} | (b) invariants {'PASS' if ok_b else 'FAIL'}, FA blocks {nfa} (with M>0: {mpos}), FA hits {fh}, identical outputs vs dyn {same}/{n}", flush=True)
    return ok_a and ok_b and mpos > 0


if __name__ == "__main__":
    ok1 = t1_structure()
    tok, model = load_model()
    ok2 = t2(tok, model)
    ok3 = t3(tok, model)
    print("CHECKS", "PASS" if ok1 and ok2 and ok3 else "FAIL", flush=True)
    sys.exit(0 if ok1 and ok2 and ok3 else 1)
