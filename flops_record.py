#!/usr/bin/env python3
"""Untimed pass for FLOPs: regenerates every prompt of one arm with per-forward records and counts FLOPs (flops.py).
Kept separate from bench.py so that recording never touches the timed region. analyze.py checks that the outputs
here equal the timed outputs before using the numbers.
  upstream : spec_trace.linear_spec_generate_traced (same tokens as model.linear_spec_generate) -> per-block shapes
  others   : linear_spec_generate_fused(record=...) -> per-forward shapes

Usage: python flops_record.py --dataset gsm8k --arm dynfat --max-new-tokens 8192 --dyn "$(cat runs/gsm8k/cost_table.txt)" --out runs/gsm8k/flops_dynfat
"""
import argparse, json, os, sys, socket, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_fused import linear_spec_generate_fused
from spec_trace import linear_spec_generate_traced
from common import MODEL, load_model, load_prompts, chat_ids, round_to_block, parse_cost_table, smi
from flops import prefill_flops, upstream_parts, fused_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--arm", required=True, help="upstream | dynfat | dyn | dynfa | fused<M>")
    ap.add_argument("--dyn", default=None, help="cost table string (dyn / dynfa / dynfat)")
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", required=True, help="output prefix (writes .json)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    meta = dict(host=socket.gethostname(), start=datetime.datetime.now().isoformat(), gpu_start=smi(), args=vars(a))
    print(json.dumps(meta), flush=True)
    tok, model = load_model(a.model)
    eos = tok.eos_token_id; n = round_to_block(a.max_new_tokens)
    prompts = load_prompts(a.dataset, a.limit)
    arm = a.arm
    if arm in ("dyn", "dynfa", "dynfat"): fkw = dict(dyn=parse_cost_table(a.dyn, arm), fa=(arm == "dynfa"))
    elif arm != "upstream": fkw = dict(M=int(arm[5:]))
    recs = []
    for i, r in enumerate(prompts):
        p = chat_ids(tok, r["user"], a.thinking)
        if arm == "upstream":
            o, nfe, tr = linear_spec_generate_traced(model, p, max_new_tokens=n, block_length=32, eos_token_id=eos)
            parts = upstream_parts(tr["blocks"])
        else:
            rec = []
            o, st = linear_spec_generate_fused(model, p, max_new_tokens=n, block_length=32, eos_token_id=eos, timing=False, record=rec, **fkw)
            nfe = st["nfe"]; parts = fused_parts(rec)
        g = o[0, p.shape[1]:].tolist()
        recs.append(dict(id=r["id"], arm=arm, prompt_len=int(p.shape[1]), n_tok=len(g), nfe=int(nfe), ids=g,
                         term=("eos" if eos in g else ("cap" if len(g) >= n else "other")),
                         flops_prefill=prefill_flops(int(p.shape[1])), flops_dec=sum(parts.values()), flops_parts=parts))
        if (i + 1) % 50 == 0: print(f"record {i + 1}/{len(prompts)} {datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
    meta.update(end=datetime.datetime.now().isoformat())
    json.dump(dict(meta=meta, records=recs), open(a.out + ".json", "w"))
    print(f"{arm}: decode TFLOP {sum(r['flops_dec'] for r in recs) / 1e12:,.1f}, prefill TFLOP {sum(r['flops_prefill'] for r in recs) / 1e12:,.1f}")
    print("written", a.out + ".json")


if __name__ == "__main__":
    main()
