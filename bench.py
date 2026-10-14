#!/usr/bin/env python3
"""Timing + accuracy run: baseline (model.linear_spec_generate) vs the fused method on one GPU.

Measurement rules
  - same model / LoRA / dtype / greedy decoding / length cap / EOS for every arm
  - warm-up: 2 prompts x every arm, twice, not recorded
  - prefill time measured in a separate pass (identical work for every arm)
  - no per-block synchronization inside the measured region; one cuda.synchronize before and after each prompt
  - prompts run in batches of --batch; the arm order is reversed every batch (and every repeat), so both arms
    see the same node conditions
  - --repeats R runs the whole set R times; analyze.py drops the first pass and averages the rest
Output tokens = tokens after the prompt as returned by the loop (EOS included when generated).

Arms: upstream (baseline) | dynfat (final method: M and the full-accept candidate chosen per block from the cost table)
      also available: dyn (M only), dynfa (full-accept candidate only on M = 0 blocks), fused<M> (fixed M)

Usage: python bench.py --dataset gsm8k --max-new-tokens 8192 --dyn "$(cat runs/gsm8k/cost_table.txt)" --repeats 6 --out runs/gsm8k/timing
"""
import argparse, json, os, sys, time, socket, datetime
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_fused import linear_spec_generate_fused, set_diffusion_lm, set_adapters
from common import MODEL, load_model, load_prompts, chat_ids, round_to_block, parse_cost_table, smi
from scoring import is_correct
from transformers.cache_utils import DynamicCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N problems only (default: all)")
    ap.add_argument("--arms", default="upstream,dynfat"); ap.add_argument("--batch", type=int, default=20); ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, required=True); ap.add_argument("--out", required=True, help="output prefix (writes .json)")
    ap.add_argument("--thinking", action="store_true", help="chat template enable_thinking=True")
    ap.add_argument("--dyn", default=None, help="cost table string from cost_table.py (needed for dyn / dynfa / dynfat)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    meta = dict(host=socket.gethostname(), start=datetime.datetime.now().isoformat(), gpu_start=smi(), args=vars(a))
    print(json.dumps(meta), flush=True)
    tok, model = load_model(a.model)
    eos = tok.eos_token_id; n = round_to_block(a.max_new_tokens)
    prompts = load_prompts(a.dataset, a.limit)
    P = [chat_ids(tok, r["user"], a.thinking) for r in prompts]
    arms = a.arms.split(",")
    DYN = {arm: parse_cost_table(a.dyn, arm) for arm in arms if arm in ("dyn", "dynfa", "dynfat")}

    def run(arm, p):
        if arm == "upstream":
            set_diffusion_lm(model, False); set_adapters(model, False)
            out, nfe = model.linear_spec_generate(p, max_new_tokens=n, block_length=32, eos_token_id=eos)
            return out, dict(nfe=int(nfe))
        if arm in DYN:
            return linear_spec_generate_fused(model, p, max_new_tokens=n, block_length=32, eos_token_id=eos, timing=False, dyn=DYN[arm], fa=(arm == "dynfa"))
        M = int(arm.replace("fused", ""))
        return linear_spec_generate_fused(model, p, max_new_tokens=n, block_length=32, eos_token_id=eos, M=M, timing=False)

    # warm-up
    for _ in range(2):
        for arm in arms:
            for p in P[:2]: run(arm, p)
    torch.cuda.synchronize()
    # prefill pass
    set_diffusion_lm(model, False); set_adapters(model, False); t_pref = []
    for p in P:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.encoder(input_ids=p, past_key_values=DynamicCache(), use_cache=True, use_causal_mask=True)
        torch.cuda.synchronize(); t_pref.append(time.perf_counter() - t0)
    print(f"prompts {len(P)} warm-up done, prefill total {sum(t_pref):.2f}s", flush=True)

    batches = [list(range(i, min(i + a.batch, len(P)))) for i in range(0, len(P), a.batch)]
    results = {arm: [[None] * len(P) for _ in range(a.repeats)] for arm in arms}

    def make_recs(reps):  # per-prompt records (built outside the measured region)
        recs = []
        for arm in arms:
            for rep in reps:
                for i, r in enumerate(prompts):
                    x = results[arm][rep][i]; ids = x["ids"]; st = x["st"]
                    text = tok.decode(ids, skip_special_tokens=True)
                    term = "eos" if eos in ids else ("cap" if len(ids) >= n else "other")
                    rec = dict(dataset=a.dataset, id=r["id"], arm=arm, gold=r["gold"], rep=rep, prompt_len=int(P[i].shape[1]), t_total=x["t_total"], t_prefill=t_pref[i],
                               t_decode=x["t_total"] - t_pref[i], n_tok=len(ids), term=term, correct=is_correct(a.dataset, text, r["gold"], term),
                               nfe=int(st["nfe"]), tpf=len(ids) / max(int(st["nfe"]), 1),
                               hits=st.get("hits"), miss_pred=st.get("miss_pred"), miss_full=st.get("miss_full"), fused_fwd=st.get("fused"), draft_fwd=st.get("draft_fwd"),
                               acc_prepared=st.get("acc_prepared"), acc_fresh=st.get("acc_fresh"), verify_plain=st.get("verify_plain"), dyn_log=st.get("dyn_log"),
                               fa_blocks=st.get("fa_blocks"), fa_hits=st.get("fa_hits"), fa_log=st.get("fa_log"), acc_fa=st.get("acc_fa"), fa_sel=st.get("fa_sel"),
                               identical_to_upstream=(ids == results["upstream"][rep][i]["ids"]) if "upstream" in results else None, ids=ids, text=text)
                    recs.append(rec)
        return recs

    for rep in range(a.repeats):
        for bi, batch in enumerate(batches):
            order = arms if (bi + rep) % 2 == 0 else arms[::-1]
            for arm in order:
                for i in batch:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    out, st = run(arm, P[i])
                    torch.cuda.synchronize(); dt = time.perf_counter() - t0
                    ids = out[0, P[i].shape[1]:].tolist()
                    results[arm][rep][i] = dict(t_total=dt, ids=ids, st={k: v for k, v in st.items()})
            done = sum(x is not None for x in results[arms[-1]][rep])
            print(f"rep {rep+1} batch {bi+1}/{len(batches)} done ({done}/{len(P)}) {datetime.datetime.now().strftime('%H:%M:%S')}", flush=True)
        if a.repeats > 1:  # checkpoint after every repeat, so a time-limit kill keeps the finished repeats
            R = make_recs([rep])
            json.dump(dict(meta=dict(meta, rep_done=rep, t_rep_done=datetime.datetime.now().isoformat(), gpu_rep_done=smi()), records=R), open(a.out + f".rep{rep}.json", "w"))
            print(f"rep {rep+1} written {a.out}.rep{rep}.json: " + " | ".join(f"{arm} {sum(x['t_total'] for x in R if x['arm'] == arm):.1f}s" for arm in arms), flush=True)
    recs = make_recs(range(a.repeats))
    meta.update(end=datetime.datetime.now().isoformat(), gpu_end=smi())
    json.dump(dict(meta=meta, records=recs), open(a.out + ".json", "w"))
    for arm in arms:
        for rep in range(a.repeats):
            R = [x for x in recs if x["arm"] == arm and x["rep"] == rep]
            T = sum(x["t_total"] for x in R); NT = sum(x["n_tok"] for x in R); TD = sum(x["t_decode"] for x in R)
            print(f"{arm} rep{rep}: n={len(R)} total {T:.1f}s tokens {NT} tok/s(all) {NT/T:.1f} decode tok/s {NT/TD:.1f} nfe {sum(x['nfe'] for x in R)} "
                  f"cap {sum(x['term'] == 'cap' for x in R)} correct {sum(x['correct'] for x in R)} identical {sum(bool(x['identical_to_upstream']) for x in R)}", flush=True)
    print("written", a.out + ".json")


if __name__ == "__main__":
    main()
