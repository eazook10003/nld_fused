#!/usr/bin/env python3
"""Cost table for the per-block choice, measured on the GPU that will run the benchmark.

Median forward times (ms) at the first block of one GSM8K prompt (test index 1):
  v    : baseline verify forward (32 rows)
  d    : baseline draft forward (32 rows)
  M    : fused forward with M candidates (32 + 32M rows), M = 1..6
  faM  : the same forward plus the 33-row full-accept candidate, M = 0..6
CUDA events, 3 warm-up + --reps measured runs per point, cache cropped back after every forward.
Writes OUT.json (all measured rows) and OUT.txt (the --dyn string for bench.py / flops_record.py).

Usage: python cost_table.py --out runs/gsm8k/cost_table
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_fused import fused_forward, fused_forward_specs, make_specs, baseline_draft, baseline_verify, draft_stats, set_diffusion_lm, set_adapters
from spec_trace import _upstream_module
from common import MODEL, BL, UPSTREAM_GSM8K, load_model
from transformers.cache_utils import DynamicCache


def cuda_ms(fn):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record(); out = fn(); e.record(); torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def table_string(results):
    R = results
    v = [r["median_ms"] for r in R if r["kind"] == "upstream_verify"][0]; d = [r["median_ms"] for r in R if r["kind"] == "upstream_draft"][0]
    f = {r["M"]: r["median_ms"] for r in R if r["kind"] == "fused_masked" and r["M"] > 0}
    fa = {r["M"]: r["median_ms"] for r in R if r["kind"] == "fused_fa"}
    return ",".join([f"v={v:.1f}", f"d={d:.1f}"] + [f"{m}={f[m]:.1f}" for m in sorted(f)] + [f"fa{m}={fa[m]:.1f}" for m in sorted(fa)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--Ms", default="0,1,2,3,4,5,6"); ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True, help="output prefix (writes .json and .txt)")
    a = ap.parse_args(); Ms = [int(x) for x in a.Ms.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tok, model = load_model(a.model)
    _crop = _upstream_module(model)._crop_dynamic_cache
    mask_id = model.config.mask_token_id
    from datasets import load_dataset
    user = UPSTREAM_GSM8K + load_dataset("openai/gsm8k", "main", split="test")[1]["question"]
    p = tok(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.to("cuda")
    set_diffusion_lm(model, False); set_adapters(model, False)
    enc = model.encoder(input_ids=p, past_key_values=DynamicCache(), use_cache=True, use_causal_mask=True); pkv = enc.past_key_values
    nt = model.diffusion_head(enc.last_hidden_state[:, -1:, :]).squeeze(1).argmax(dim=-1, keepdim=True)
    cache_len = pkv.get_seq_length()
    block = torch.full((1, BL), mask_id, dtype=torch.long, device="cuda"); block[0, 0] = nt.item()
    dl = baseline_draft(model, pkv, block); dt, conf, tk_ids, tk_p = draft_stats(dl, 8); block[0, 1:] = dt[1:]

    def timeit(fn, crop=True):
        ts = []
        for r in range(3 + a.reps):
            _, t = cuda_ms(fn)
            if crop: _crop(pkv, cache_len)
            if r >= 3: ts.append(t)
        return float(np.mean(ts)), float(np.median(ts))

    results = []
    m_, md_ = timeit(lambda: baseline_verify(model, pkv, block)); results.append(dict(cache_len=cache_len, kind="upstream_verify", M=None, n_pos=32, mean_ms=m_, median_ms=md_))
    m_, md_ = timeit(lambda: baseline_draft(model, pkv, block), crop=False); results.append(dict(cache_len=cache_len, kind="upstream_draft", M=None, n_pos=32, mean_ms=m_, median_ms=md_))
    for M in Ms:
        ms_ = [int(round(x)) for x in np.linspace(0, 30, M)] if M > 0 else []  # fixed rejection positions, spread over the block
        cands = [(m, int(tk_ids[m + 1, 1])) for m in ms_]
        torch.cuda.reset_peak_memory_stats()
        m_, md_ = timeit(lambda: fused_forward(model, pkv, block, cands, cache_len, mask_id))
        results.append(dict(cache_len=cache_len, kind="fused_masked", M=M, n_pos=BL * (1 + M), mean_ms=m_, median_ms=md_, peak_gb=torch.cuda.max_memory_allocated() / 2**30))
        print(results[-1], flush=True)
        specs = make_specs("C", BL, cands, block) + make_specs("P", BL + 1, [(BL - 1, None)], block)  # same M candidates + full-accept candidate
        torch.cuda.reset_peak_memory_stats()
        m_, md_ = timeit(lambda: fused_forward_specs(model, pkv, block, specs, cache_len, mask_id))
        results.append(dict(cache_len=cache_len, kind="fused_fa", M=M, n_pos=BL * (1 + M) + BL + 1, mean_ms=m_, median_ms=md_, peak_gb=torch.cuda.max_memory_allocated() / 2**30))
        print(results[-1], flush=True)
    s = table_string(results)
    json.dump(dict(gpu=torch.cuda.get_device_name(), table=s, rows=results), open(a.out + ".json", "w"), indent=1)
    open(a.out + ".txt", "w").write(s + "\n")
    print("cost table", s)
    print("written", a.out + ".json", a.out + ".txt")


if __name__ == "__main__":
    main()
