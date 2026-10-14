"""Fused verify + draft for Nemotron-Labs-Diffusion linear self-speculation.

The baseline loop runs two forwards per block: a bidirectional draft forward (LoRA on) and a causal AR verify
forward (LoRA off). Here the verify forward also drafts candidate next blocks, so when a candidate matches the
verify result, the next draft forward is skipped.

Layout of one fused forward (block length BL = 32, M candidates):
  input rows   : [ verify block (32) | cand_1 (32) | ... | cand_M (32) ]
  verify block : [seed, d_0..d_30], the same block the baseline verify forward sees
  cand_c       : [v_c, MASK x 31], the block the next baseline draft forward would see if the first rejection
                 is at draft index m_c and the AR correction is v_c
Positions (RoPE): verify -> cache_len + 0..31, cand_c -> cache_len + (m_c + 1) + 0..31
Attention mask (True = may attend):
  verify row p : all cache, verify cols <= p (causal), never a candidate column
  cand_c rows  : all cache, verify cols 0..m_c, its own 32 cols (bidirectional), never another candidate
LoRA (draft-only adapter) is gated per row: 0 on verify rows, 1 on candidate rows.
KV: after the forward the cache is cropped to cache_len + accepted, so only the accepted verify rows are kept.

Full-accept candidate: 33 rows [cell 31, MASK x 32] at positions cache_len + 31.., seeing the cache and verify
cols 0..30 (a draft forward started one token earlier). Row 1 guesses the bonus token, rows 2..32 draft the 31
cells after it. It is used only when the whole block is accepted and the guess equals the AR bonus token.

Per block, the number of candidates M and whether to add the full-accept candidate are chosen together from a
measured cost table (dyn_choose_fa): expected time = forward time + P(a separate draft forward is needed) * draft time.
"""
import copy, time, types
from contextlib import contextmanager
import torch
from transformers.cache_utils import DynamicCache
from spec_trace import _upstream_module, _sync_ms

# ---------------------------------------------------------------- position-gated LoRA
_GATE = {"t": None}


def _gated_forward(self, x, *args, **kwargs):
    gate = _GATE["t"]
    if gate is None or self.disable_adapters or self.merged:
        return self._nld_orig_forward(x, *args, **kwargs)
    kwargs.pop("adapter_names", None)
    result = self.base_layer(x, *args, **kwargs)
    dtype = result.dtype
    g = gate.to(result.dtype)
    for a in self.active_adapters:
        if a not in self.lora_A:
            continue
        xa = self._cast_input_dtype(x, self.lora_A[a].weight.dtype)
        result = result + g * (self.lora_B[a](self.lora_A[a](self.lora_dropout[a](xa))) * self.scaling[a])
    return result.to(dtype)


def install_gated_lora(model):
    """Patch every peft LoRA Linear so that, while a gate is set, the LoRA delta is multiplied per position."""
    n = 0
    for mod in model.modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and not hasattr(mod, "_nld_orig_forward"):
            mod._nld_orig_forward = mod.forward
            mod.forward = types.MethodType(_gated_forward, mod)
            n += 1
    return n


def set_adapters(model, enable: bool):
    for mod in model.modules():
        if hasattr(mod, "_disable_adapters"):
            mod._disable_adapters = not enable


def set_diffusion_lm(model, val: bool):
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = val


@contextmanager
def lora_gate(model, gate):
    set_adapters(model, True)  # adapters must be active for the gated path; gate==0 rows get exactly the base output
    _GATE["t"] = gate
    try:
        yield
    finally:
        _GATE["t"] = None


# ---------------------------------------------------------------- fused inputs
def build_fused_inputs(block, cands, cache_len, mask_id, device):
    BL = block.shape[1]
    M = len(cands)
    L = BL * (1 + M)
    ids = torch.full((1, L), mask_id, dtype=torch.long, device=device)
    ids[0, :BL] = block[0]
    pos = torch.empty(L, dtype=torch.long, device=device)
    pos[:BL] = torch.arange(cache_len, cache_len + BL, device=device)
    gate = torch.zeros(1, L, 1, device=device)
    kv = cache_len + L
    mask = torch.zeros(L, kv, dtype=torch.bool, device=device)
    mask[:BL, :cache_len] = True
    mask[:BL, cache_len:cache_len + BL] = torch.tril(torch.ones(BL, BL, dtype=torch.bool, device=device))
    for c, (m, v) in enumerate(cands):
        r0 = BL * (c + 1)
        ids[0, r0] = v
        pos[r0:r0 + BL] = torch.arange(cache_len + m + 1, cache_len + m + 1 + BL, device=device)
        gate[0, r0:r0 + BL, 0] = 1.0
        mask[r0:r0 + BL, :cache_len] = True
        mask[r0:r0 + BL, cache_len:cache_len + m + 1] = True
        mask[r0:r0 + BL, cache_len + r0:cache_len + r0 + BL] = True
    return ids, pos.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0), gate


@torch.no_grad()
def fused_forward(model, past_key_values, block, cands, cache_len, mask_id):
    """One forward. Returns (past_key_values with [verify|cands] appended, verify_logits (1,32,V), [cand_logits (1,32,V)...])."""
    BL = block.shape[1]
    ids, pos, mask, gate = build_fused_inputs(block, cands, cache_len, mask_id, block.device)
    set_diffusion_lm(model, False)  # mask-driven attention path
    cache_position = torch.arange(cache_len, cache_len + ids.shape[1], device=block.device)
    with lora_gate(model, gate):
        enc_out = model.encoder(input_ids=ids, position_ids=pos, cache_position=cache_position, attention_mask=mask,
                                past_key_values=past_key_values, use_cache=True, use_causal_mask=True)
    logits = model.diffusion_head(enc_out.last_hidden_state)
    verify_logits = logits[:, :BL]
    cand_logits = [logits[:, BL * (c + 1):BL * (c + 2)] for c in range(len(cands))]
    return enc_out.past_key_values, verify_logits, cand_logits


# ---------------------------------------------------------------- candidate rows given as specs (full-accept candidate)
def make_specs(kind, rows, pairs, block):
    """pairs: [(m, v)].
    C: seed v at relative position m+1, sees verify cols 0..m (same rows as fused_forward's candidates when rows = 32).
    P: seed block[m] at relative position m, sees verify cols 0..m-1. The full-accept candidate is P with m = 31, 33 rows."""
    if kind == "C":
        return [dict(kind="C", m=m, v=v, seed=v, pos_rel=m + 1, vis=m + 1, rows=rows) for m, v in pairs]
    return [dict(kind="P", m=m, v=v, seed=int(block[0, m]), pos_rel=m, vis=m, rows=rows) for m, v in pairs]


def build_inputs(block, specs, cache_len, mask_id, device):
    BL = block.shape[1]; L = BL + sum(s["rows"] for s in specs)
    ids = torch.full((1, L), mask_id, dtype=torch.long, device=device); ids[0, :BL] = block[0]
    pos = torch.empty(L, dtype=torch.long, device=device); pos[:BL] = torch.arange(cache_len, cache_len + BL, device=device)
    gate = torch.zeros(1, L, 1, device=device)
    mask = torch.zeros(L, cache_len + L, dtype=torch.bool, device=device)
    mask[:BL, :cache_len] = True
    mask[:BL, cache_len:cache_len + BL] = torch.tril(torch.ones(BL, BL, dtype=torch.bool, device=device))
    r0 = BL; spans = []
    for s in specs:
        n = s["rows"]
        ids[0, r0] = s["seed"]
        pos[r0:r0 + n] = torch.arange(cache_len + s["pos_rel"], cache_len + s["pos_rel"] + n, device=device)
        gate[0, r0:r0 + n, 0] = 1.0
        mask[r0:r0 + n, :cache_len] = True
        mask[r0:r0 + n, cache_len:cache_len + s["vis"]] = True
        mask[r0:r0 + n, cache_len + r0:cache_len + r0 + n] = True
        spans.append((r0, r0 + n)); r0 += n
    return ids, pos.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0), gate, spans


@torch.no_grad()
def fused_forward_specs(model, past_key_values, block, specs, cache_len, mask_id):
    """One forward over [verify 32 | candidate rows...]. Returns (pkv, verify_logits (1,32,V), [cand_logits (1,rows,V)])."""
    BL = block.shape[1]
    ids, pos, mask, gate, spans = build_inputs(block, specs, cache_len, mask_id, block.device)
    set_diffusion_lm(model, False)
    cache_position = torch.arange(cache_len, cache_len + ids.shape[1], device=block.device)
    with lora_gate(model, gate):
        enc_out = model.encoder(input_ids=ids, position_ids=pos, cache_position=cache_position, attention_mask=mask,
                                past_key_values=past_key_values, use_cache=True, use_causal_mask=True)
    logits = model.diffusion_head(enc_out.last_hidden_state)
    return enc_out.past_key_values, logits[:, :BL], [logits[:, a:b] for a, b in spans]


def suffix_logits(spec, clog):
    """(1,32,V) logits laid out like a next block: row 0 = placeholder for the seed cell, rows 1..31 = the 31 suffix cells."""
    if spec["kind"] == "C":
        return clog[:, :32]                      # row 0 = seed row (unused), rows 1..31 = suffix ; row 32 (if any) dropped
    return torch.cat([clog[:, 1:2], clog[:, 2:33]], dim=1)  # row 0 placeholder = P's guess row, rows 1..31 = clog rows 2..32


# ---------------------------------------------------------------- baseline single steps
@torch.no_grad()
def baseline_draft(model, past_key_values, block):
    set_diffusion_lm(model, True); set_adapters(model, True)
    enc_out = model.encoder(input_ids=block, past_key_values=past_key_values, use_cache=False)
    return model.diffusion_head(enc_out.last_hidden_state)


@torch.no_grad()
def baseline_verify(model, past_key_values, block):
    set_diffusion_lm(model, False); set_adapters(model, False)
    enc_out = model.encoder(input_ids=block, past_key_values=past_key_values, use_cache=True, use_causal_mask=True)
    return enc_out.past_key_values, model.diffusion_head(enc_out.last_hidden_state)


def accept(verify_logits, block):
    """Baseline acceptance rule. Returns ar_tokens (1,32), matched, accepted."""
    ar = verify_logits.argmax(dim=-1)
    BL = block.shape[1]
    matched = 0
    for i in range(BL - 1):
        if ar[0, i].item() == block[0, i + 1].item():
            matched += 1
        else:
            break
    return ar, matched, matched + 1


def draft_stats(draft_logits, topk):
    """From draft logits (1,32,V): draft tokens (32), top-1 conf (32), top-k ids/probs (32,k)."""
    draft_tokens = draft_logits.argmax(dim=-1)[0]
    probs = torch.softmax(draft_logits, dim=-1)[0]
    conf = torch.gather(probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1).float()
    tk_ids = torch.topk(draft_logits[0].float(), topk, dim=-1).indices
    tk_p = torch.gather(probs.float(), -1, tk_ids)
    return draft_tokens, conf, tk_ids, tk_p


# ---------------------------------------------------------------- candidate selection and the per-block choice
def select_candidates(conf, draft_tokens, tk_ids, tk_p, M, exclude=None, with_scores=False):
    """Joint score S(j, v) = prod_{i<j} c_i * q_j(v), v != draft token, over drafted cells j=0..30 (block cells 1..31).
    Uses only pre-verification draft information. Returns top-M (m, v) with ties -> lower j, lower rank
    (and, with_scores=True, also their joint scores S)."""
    c = conf[1:].double().cpu()
    prefix = torch.cat([torch.ones(1, dtype=torch.double), torch.cumprod(c, 0)[:-1]])
    ids = tk_ids[1:].cpu(); ps = tk_p[1:].double().cpu(); dt = draft_tokens[1:].cpu()
    pairs = []
    for j in range(ids.shape[0]):
        rank = 0
        for r in range(ids.shape[1]):
            v = int(ids[j, r])
            if v == int(dt[j]):
                continue
            pairs.append((-(float(prefix[j]) * float(ps[j, r])), j, rank, v))
            rank += 1
    pairs.sort()
    out = []; sc = []
    for s, j, _, v in pairs:
        if exclude is not None and (j, v) in exclude:
            continue
        out.append((j, v)); sc.append(-s)
        if len(out) == M:
            break
    return (out, sc) if with_scores else out


def dyn_choose(sc, dyn):
    """dyn = (verify_ms, draft_ms, {M: fused_ms}); sc = joint scores of the top candidates, best first.
    Expected block time: M=0 -> verify + draft; M>0 -> fused[M] + (1 - h(M)) * draft, h(M) = sum of the top-M scores
    (a hit skips the next draft forward). Returns the M with the smallest expected time (ties -> fewer candidates)."""
    v, d, f = dyn
    best_M, best_t = 0, v + d
    for M in sorted(f):
        t = f[M] + (1 - sum(sc[:M])) * d
        if t < best_t:
            best_M, best_t = M, t
    return best_M


def dyn_choose_fa(sc, pfa, dyn):
    """dyn = (verify_ms, draft_ms, {M: fused_ms}, {M: fused_ms with the full-accept candidate});
    pfa = predicted P(whole block accepted) = product of the 31 draft confidences. Expected block time per option:
      (0, no FA): verify + draft ; (0, FA): fa[0] + (1 - pfa) * draft
      (M, no FA): fused[M] + (1 - h(M)) * draft ; (M, FA): fa[M] + (1 - h(M) - pfa) * draft
    h(M) = sum of the top-M joint scores. Full accept and every (m, v) outcome are disjoint, so h(M) + pfa <= 1.
    Returns (M, use_fa) with the smallest expected time (ties -> the option listed first: fewer rows, no FA before FA)."""
    v, d, f, ffa = dyn
    best, best_t = (0, False), v + d
    if 0 in ffa and ffa[0] + (1 - pfa) * d < best_t:
        best, best_t = (0, True), ffa[0] + (1 - pfa) * d
    for M in sorted(f):
        h = sum(sc[:M])
        t = f[M] + (1 - h) * d
        if t < best_t:
            best, best_t = (M, False), t
        if M in ffa:
            t = ffa[M] + (1 - h - pfa) * d
            if t < best_t:
                best, best_t = (M, True), t
    return best


# ---------------------------------------------------------------- generation loop
@torch.no_grad()
def linear_spec_generate_fused(model, prompt_ids, max_new_tokens=128, block_length=32, eos_token_id=None,
                               M=1, topk=8, mask_token_id=None, timing=True, record=None, dyn=None, fa=False):
    """Baseline linear self-speculation where every verify forward also drafts candidate next blocks.
    On a candidate hit the prepared draft becomes the next block (no draft forward); otherwise a normal draft
    forward follows. Accepted tokens follow the baseline acceptance rule.

      dyn=None                 : fixed M candidates per block
      dyn=(v, d, f)            : M chosen per block by dyn_choose (M = 0 uses the plain verify forward)
      dyn=(v, d, f, ffa)       : M and the full-accept candidate chosen per block by dyn_choose_fa (final method)
      dyn=(v, d, f) + fa=True  : full-accept candidate only on blocks where dyn_choose picks M = 0
    Cost table entries are in ms: v = verify, d = draft, f[M] = fused forward with M candidates,
    ffa[M] = the same forward plus the full-accept candidate.

    Returns (output ids incl. prompt, stats). stats["dyn_log"] has one entry per block:
    [M, rank of the true (m, v) among the top candidates (0 = not among them, -1 = whole block accepted), scores...].
    stats["fa_log"] has one entry per block that carried the full-accept candidate:
    [predicted P(full accept), full (0/1), guess == bonus (-1 if not full), guess prob, M].
    record (list) collects per-forward shapes for FLOP counting."""
    _crop = _upstream_module(model)._crop_dynamic_cache
    mask_id = mask_token_id if mask_token_id is not None else model.config.mask_token_id
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    device = prompt_ids.device
    stats = dict(nfe=0, fused=0, draft_fwd=0, hits=0, miss_pred=0, miss_full=0, blocks=0,
                 acc_prepared=[], acc_fresh=[], t_prefill=0.0, t_fused=0.0, t_draft=0.0, t_select=0.0)
    if dyn is not None: stats.update(verify_plain=0, dyn_log=[])
    fa_tab = dyn is not None and len(dyn) > 3  # cost table has full-accept entries: choose M and FA together
    if fa_tab: assert not fa, "dyn with an FA table decides the FA candidate itself; fa must be False"
    if fa or fa_tab:
        assert dyn is not None, "fa=True replaces the dyn M=0 blocks"
        stats.update(fa_blocks=0, fa_hits=0, fa_log=[], acc_fa=[])
        if fa_tab: stats.update(fa_sel=[])
    _ms = _sync_ms if timing else (lambda t0: 0.0)  # timing=False: no per-block CUDA synchronization
    prompt_len = int(prompt_ids.shape[1])

    set_diffusion_lm(model, False); set_adapters(model, False)
    t0 = time.perf_counter()
    enc_out = model.encoder(input_ids=prompt_ids, past_key_values=DynamicCache(), use_cache=True, use_causal_mask=True)
    pkv = enc_out.past_key_values
    next_token = model.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1).argmax(dim=-1, keepdim=True)
    stats["nfe"] += 1; stats["t_prefill"] = _ms(t0)
    if record is not None: record.append(dict(kind="prefill", cache_len=0, n_pos=prompt_len))
    if eos_token_id is not None and next_token.item() == eos_token_id:
        return torch.cat([prompt_ids, next_token], dim=1), stats
    generated = [next_token]; total_gen = 1
    prepared = None  # (block tensor (1,32), conf, tk_ids, tk_p, draft_tokens) for the next block, if a candidate hit
    prepared_fa = False  # the prepared block came from the full-accept candidate

    while total_gen < max_new_tokens:
        cache_len = pkv.get_seq_length()
        if prepared is None:
            block = torch.full((1, block_length), mask_id, dtype=torch.long, device=device)
            block[0, 0] = next_token.item()
            t0 = time.perf_counter()
            dl = baseline_draft(model, pkv, block)
            stats["nfe"] += 1; stats["draft_fwd"] += 1; stats["t_draft"] += _ms(t0)
            if record is not None: record.append(dict(kind="draft", cache_len=int(cache_len), n_pos=block_length))
            draft_tokens, conf, tk_ids, tk_p = draft_stats(dl, topk)
            block[0, 1:] = draft_tokens[1:]
            from_prepared = False
        else:
            block, conf, tk_ids, tk_p, draft_tokens = prepared; prepared = None
            from_prepared = True
        from_fa = prepared_fa; prepared_fa = False
        t0 = time.perf_counter()
        if dyn is None:
            cands = select_candidates(conf, draft_tokens, tk_ids, tk_p, M)
        elif not fa_tab:
            top, sc = select_candidates(conf, draft_tokens, tk_ids, tk_p, max(dyn[2]), with_scores=True)
            cands = top[:dyn_choose(sc, dyn)]
        else:  # joint choice of M and the full-accept candidate
            top, sc = select_candidates(conf, draft_tokens, tk_ids, tk_p, max(dyn[2]), with_scores=True)
            pfa = float(conf[1:].double().prod())
            Mx, fa_pick = dyn_choose_fa(sc, pfa, dyn)
            cands = top[:Mx]; stats["fa_sel"].append(int(fa_pick))
        stats["t_select"] += (time.perf_counter() - t0) * 1000 if timing else 0.0
        t0 = time.perf_counter()
        use_fa = (fa and not cands) or (fa_tab and fa_pick)
        plain = dyn is not None and not cands and not use_fa
        if plain:  # no candidates: plain verify forward, next block drafts normally
            pkv, vlog = baseline_verify(model, pkv, block); clogs = []
            stats["verify_plain"] += 1
        elif use_fa and cands:  # verify + M candidates (32 rows each, same as fused_forward) + the 33-row full-accept candidate
            fa_spec = make_specs("P", block_length + 1, [(block_length - 1, None)], block)
            pkv, vlog, all_clogs = fused_forward_specs(model, pkv, block, make_specs("C", block_length, cands, block) + fa_spec, cache_len, mask_id)
            clogs, fa_clogs = all_clogs[:-1], all_clogs[-1:]
            stats["fused"] += 1; stats["fa_blocks"] += 1
        elif use_fa:  # verify + the full-accept candidate only
            fa_spec = make_specs("P", block_length + 1, [(block_length - 1, None)], block)
            pkv, vlog, fa_clogs = fused_forward_specs(model, pkv, block, fa_spec, cache_len, mask_id); clogs = []
            stats["fused"] += 1; stats["fa_blocks"] += 1
        else:
            pkv, vlog, clogs = fused_forward(model, pkv, block, cands, cache_len, mask_id)
            stats["fused"] += 1
        stats["nfe"] += 1; stats["blocks"] += 1
        ar, matched, accepted = accept(vlog, block)
        stats["t_fused"] += _ms(t0)
        if record is not None: record.append(dict(kind="fused", cache_len=int(cache_len), n_pos=block_length * (1 + len(cands)) + (block_length + 1 if use_fa else 0), M=len(cands), cands=cands, matched=int(matched), from_prepared=from_prepared, plain=plain, fa=use_fa))
        (stats["acc_prepared"] if from_prepared else stats["acc_fresh"]).append(accepted)
        if from_fa: stats["acc_fa"].append(accepted)
        accepted_toks = ar[:, :accepted]
        generated.append(accepted_toks); total_gen += accepted
        _crop(pkv, cache_len + accepted)
        next_token = ar[:, accepted - 1:accepted]
        if dyn is not None:
            true = None if matched == block_length - 1 else (int(matched), next_token.item())
            stats["dyn_log"].append([len(cands), -1 if true is None else (top.index(true) + 1 if true in top else 0)] + [round(x, 5) for x in sc])
        if eos_token_id is not None and (accepted_toks[0] == eos_token_id).any():
            first_eos = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0][0].item()
            generated[-1] = accepted_toks[:, :first_eos + 1]; total_gen = total_gen - accepted + first_eos + 1
            break
        if total_gen >= max_new_tokens:
            break
        if use_fa:  # usable only if the whole block passed AND its bonus-slot guess equals the AR bonus
            sl = suffix_logits(fa_spec[0], fa_clogs[0])  # (1,32,V): row 0 = guess for the bonus slot, rows 1..31 = following cells
            full = matched == block_length - 1
            dt, cf, ti, tp = draft_stats(sl, topk)
            guess = int(dt[0]); hit = full and guess == next_token.item()
            entry = [round(float(conf[1:].double().prod()), 5), int(full), (int(hit) if full else -1), round(float(cf[0]), 5)]
            if fa_tab: entry.append(len(cands))
            stats["fa_log"].append(entry)
            if hit:
                nb = torch.full((1, block_length), mask_id, dtype=torch.long, device=device)
                nb[0, 0] = next_token.item(); nb[0, 1:] = dt[1:]
                prepared = (nb, cf, ti, tp, dt); prepared_fa = True; stats["hits"] += 1; stats["fa_hits"] += 1
            elif full:
                stats["miss_full"] += 1
            elif not cands:
                stats["miss_pred"] += 1
        if use_fa and (matched == block_length - 1 or not cands):
            pass  # settled by the full-accept candidate above (with other candidates, a rejection falls through to them below)
        elif matched == block_length - 1:
            stats["miss_full"] += 1  # whole block accepted: next seed is the bonus, no candidate covers it
        else:
            corr = next_token.item()
            for c, (m, v) in enumerate(cands):
                if m == matched and v == corr:
                    dt, cf, ti, tp = draft_stats(clogs[c], topk)
                    nb = torch.full((1, block_length), mask_id, dtype=torch.long, device=device)
                    nb[0, 0] = v; nb[0, 1:] = dt[1:]
                    prepared = (nb, cf, ti, tp, dt); stats["hits"] += 1
                    break
            if prepared is None:
                stats["miss_pred"] += 1
        if record is not None: record[-1]["hit"] = prepared is not None; record[-1]["full"] = (matched == block_length - 1)
    return torch.cat([prompt_ids] + generated, dim=1), stats
