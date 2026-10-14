#!/usr/bin/env python3
"""Result table for one run directory: baseline (upstream) vs the fused arm.

Reads RUN/timing.json (or, while bench.py is still running, the per-repeat files RUN/timing.rep<k>.json) and, if present,
RUN/flops_<arm>.json from flops_record.py. Writes RUN/table.md.
With 2+ repeats the first pass is dropped and times are averaged per problem over the rest.
Tables: (1) usual metrics with paired 95% bootstrap CIs (over problems, 4000 resamples, seed 0), (2) full-accept blocks,
(3) wall per repeat, (4) first-pass effect, (5) result vs number of averaged repeats, (6) output identity.
Accuracy: GSM8K = authors' SGLang scorer, MATH-500 = math_verify (scoring.py); an output without EOS counts as wrong.

Usage: python analyze.py runs/gsm8k
"""
import glob, json, math, os, sys
import numpy as np

RUN = sys.argv[1]
NAME = {"upstream": "baseline", "dyn": "dynamic", "dynfa": "dynamic + 통째수락(0개 기준)", "dynfat": "dynamic + 통째수락(시간표 기준)"}
fp = os.path.join(RUN, "timing.json")
if os.path.exists(fp):
    J = json.load(open(fp)); R = J["records"]; meta = J["meta"]; src = "timing.json"
else:
    files = sorted(glob.glob(os.path.join(RUN, "timing.rep*.json")), key=lambda f: int(f.rsplit(".rep", 1)[1][:-5]))
    assert files, f"no timing.json or timing.rep*.json in {RUN}"
    R, meta = [], None
    for f in files:
        J = json.load(open(f)); R += J["records"]; meta = J["meta"]
    src = f"회차별 저장 파일 {len(files)}개 (실행 중)"
ARMS = meta["args"]["arms"].split(",")
u = "upstream"; d = [a for a in ARMS if a != u][0]
name = lambda a: NAME.get(a, f"고정 M={a[5:]}" if a.startswith("fused") else a)
DSET = meta["args"]["dataset"]; reps = sorted({r["rep"] for r in R})
MR = reps[1:] if len(reps) >= 2 else reps  # measured repeats (pass 0 is dropped when there are 2+)
D = {(a, k): {r["id"]: r for r in R if r["arm"] == a and r["rep"] == k} for a in (u, d) for k in reps}
ids = list(D[(u, reps[0])])
k1 = MR[0]


def boot(f, n=4000):
    rng = np.random.default_rng(0); k = np.arange(len(ids)); v = [f(k[rng.integers(0, len(k), len(k))]) for _ in range(n)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


pct = lambda x: f"{100 * (x - 1):+.1f}%"
ci = lambda lo, hi: f"[{100 * (lo - 1):+.1f}, {100 * (hi - 1):+.1f}]"
T = {a: np.array([[D[(a, k)][i]["t_total"] for i in ids] for k in reps]) for a in (u, d)}   # [repeat, problem]
Tm = {a: T[a][[reps.index(k) for k in MR]].mean(axis=0) for a in (u, d)}                    # mean over measured repeats
N = {a: np.array([D[(a, k1)][i]["n_tok"] for i in ids], dtype=float) for a in (u, d)}
NF = {a: np.array([D[(a, k1)][i]["nfe"] for i in ids], dtype=float) for a in (u, d)}
A = {a: {i: bool(D[(a, k1)][i]["correct"]) for i in ids} for a in (u, d)}

# ---------------------------------------------------------------- FLOPs (only if flops_record.py was run)
FL, same_fl = {}, {}
for a in (u, d):
    p = os.path.join(RUN, f"flops_{a}.json")
    if os.path.exists(p):
        X = {r["id"]: r for r in json.load(open(p))["records"]}
        if all(i in X for i in ids):
            same_fl[a] = sum(X[i]["ids"] == D[(a, k1)][i]["ids"] for i in ids)
            pre = sum(X[i]["flops_prefill"] for i in ids); dec = sum(X[i]["flops_dec"] for i in ids); nt = sum(X[i]["n_tok"] for i in ids)
            FL[a] = dict(gft_all=(pre + dec) / nt / 1e9, gft_dec=dec / nt / 1e9, tfl_all=(pre + dec) / 1e12)

# ---------------------------------------------------------------- (1) usual metrics
W = {a: Tm[a].sum() for a in (u, d)}
m = {a: dict(thr=N[a].sum() / W[a], tpt=1000 * W[a] / N[a].sum(), tpf=N[a].sum() / NF[a].sum(), acc=100 * np.mean(list(A[a].values())), wall=W[a],
             **(FL.get(a) or dict(gft_all=None, gft_dec=None, tfl_all=None))) for a in (u, d)}
rep_txt = f"{len(MR)}번 평균" if len(MR) > 1 else "1번"
L = [f"# {DSET} {len(ids)}문제: {name(u)} vs {name(d)} ({meta['host']}, {src})", ""]
if len(reps) >= 2:
    L += [f"측정 회차 = {len(MR)}번 평균 (1번째 실행은 버림). 시간은 문제마다 {len(MR)}번 평균을 낸 뒤 합한 값.", ""]
L += [f"| 지표 | {name(u)} | {name(d)} | 변화 [95% CI] |", "|---|---|---|---|"]
rows = [("Throughput (tok/s)", "thr", "{:.1f}", lambda j: (N[d][j].sum() / Tm[d][j].sum()) / (N[u][j].sum() / Tm[u][j].sum())),
        ("토큰당 시간 (ms)", "tpt", "{:.2f}", lambda j: (Tm[d][j].sum() / N[d][j].sum()) / (Tm[u][j].sum() / N[u][j].sum())),
        ("TPF", "tpf", "{:.2f}", None), ("Acc" + (" (math_verify)" if DSET == "math500" else ""), "acc", "{:.2f}", None),
        ("GFLOP/token (prefill 포함)", "gft_all", "{:.1f}", None), ("GFLOP/token (decode만)", "gft_dec", "{:.1f}", None),
        ("전체 TFLOP (prefill 포함)", "tfl_all", "{:,.0f}", None),
        (f"Wall time (s, {rep_txt})", "wall", "{:,.1f}", lambda j: Tm[d][j].sum() / Tm[u][j].sum())]
for lab, k, f, bf in rows:
    if m[u][k] is None or m[d][k] is None: L.append(f"| {lab} | – | – | flops_record.py 결과 없음 |"); continue
    if k == "acc":
        x = sum(A[d][i] and not A[u][i] for i in ids); y = sum(A[u][i] and not A[d][i] for i in ids); n = x + y
        p = min(1.0, 2 * sum(math.comb(n, t) for t in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
        ch = f"{m[d][k] - m[u][k]:+.2f}%p ({x} / {y}, McNemar p={p:.2f})"
    else:
        ch = pct(m[d][k] / m[u][k]) + (f" {ci(*boot(bf))}" if bf else "")
    L.append(f"| {lab} | {f.format(m[u][k])} | {f.format(m[d][k])} | {ch} |")
cap = {a: sum(D[(a, k1)][i]["term"] == "cap" for i in ids) for a in (u, d)}
L += ["", f"정확도 괄호 = ({name(d)}만 정답 / {name(u)}만 정답). 길이 한도 도달: {name(u)} {cap[u]}개, {name(d)} {cap[d]}개."]
if FL:
    L.append("FLOPs 기록 런 출력이 시간 측정 출력과 같은 문제 수: " + ", ".join(f"{name(a)} {same_fl[a]}/{len(ids)}" for a in same_fl))

# ---------------------------------------------------------------- (2) full-accept blocks (fused arm)
X = [D[(d, k1)][i] for i in ids]
if X[0].get("fa_log") is not None:
    nb = full = cov = 0; used = sum(r["fa_hits"] or 0 for r in X); nfa = sum(r["fa_blocks"] or 0 for r in X)
    for r in X:  # the last block of each output ends the loop before its bookkeeping, so it is left out
        Lg = r["dyn_log"][:-1]
        sel = r["fa_sel"][:-1] if r.get("fa_sel") is not None else [int(x[0] == 0) for x in Lg]
        for x, s in zip(Lg, sel):
            f = x[1] == -1; nb += 1; full += f; cov += bool(s and f)
    L += ["", "## 통째수락 블록", "", "| 항목 | 값 |", "|---|---|",
          f"| 전체 블록 | {nb:,} |", f"| 통째수락된 블록 (32칸 전부 수락) | {full:,} ({100 * full / max(nb, 1):.1f}%) |",
          f"| 통째수락 후보를 넣은 블록 | {nfa:,} |",
          f"| 통째수락된 블록 중 후보가 들어 있던 비율 | {100 * cov / max(full, 1):.1f}% |",
          f"| 통째수락된 블록 중 후보를 재사용한 비율 | {100 * used / max(full, 1):.1f}% ({used:,}개) |"]

# ---------------------------------------------------------------- (3) wall per repeat
if len(reps) >= 2:
    L += ["", "## 회차별 wall", "", f"| 회차 | {name(u)} (s) | {name(d)} (s) | 변화 [95% CI] |", "|---|---|---|---|"]
    for r, k in enumerate(reps):
        lo, hi = boot(lambda j, r=r: T[d][r, j].sum() / T[u][r, j].sum())
        L.append(f"| {k + 1}{' (버림)' if k == 0 else ''} | {T[u][r].sum():,.1f} | {T[d][r].sum():,.1f} | {pct(T[d][r].sum() / T[u][r].sum())} {ci(lo, hi)} |")
    tot = {a: T[a][1:].sum(axis=1) for a in (u, d)}; chg = 100 * (tot[d] / tot[u] - 1)
    sd = lambda x: x.std(ddof=1) if len(x) > 1 else 0.0
    L += ["", "측정 회차끼리 흔들림: " + ", ".join(f"{name(a)} 평균 {tot[a].mean():,.1f}s, 표준편차 {sd(tot[a]):,.1f}s (변동계수 {100 * sd(tot[a]) / tot[a].mean():.2f}%)" for a in (u, d))
          + f"; 변화율 회차별 {', '.join(f'{c:+.1f}%' for c in chg)}"]

    # ------------------------------------------------------------ (4) first-pass effect
    nbat = meta["args"]["batch"]
    L += ["", "## 1번째 실행 효과 (버린 회차 vs 측정 회차 평균)", "", f"| 팔 | 1번째 실행 전체 | 1번째 실행의 첫 묶음 (문제 {nbat}개) |", "|---|---|---|"]
    for a in (u, d):
        L.append(f"| {name(a)} | {pct(T[a][0].sum() / Tm[a].sum())} | {pct(T[a][0, :nbat].sum() / Tm[a][:nbat].sum())} |")

    # ------------------------------------------------------------ (5) result vs number of averaged repeats
    L += ["", f"## 평균 낸 회차 수에 따른 결과 ({name(d)} vs {name(u)} wall)", "", "| 평균 낸 회차 | 변화 | 95% CI | CI 폭 |", "|---|---|---|---|"]
    for n in range(1, len(MR) + 1):
        a_ = T[d][1:n + 1].mean(axis=0); b_ = T[u][1:n + 1].mean(axis=0); lo, hi = boot(lambda j: a_[j].sum() / b_[j].sum())
        lab = "2번째만" if n == 1 else f"2~{n + 1}번째"
        L.append(f"| {lab} ({n}번) | {pct(a_.sum() / b_.sum())} | {ci(lo, hi)} | {100 * (hi - lo):.1f}%p |")

# ---------------------------------------------------------------- (6) output identity
same_rep = {a: sum(all(D[(a, k)][i]["ids"] == D[(a, reps[0])][i]["ids"] for k in reps) for i in ids) for a in (u, d)}
same_ud = sum(D[(d, k1)][i]["ids"] == D[(u, k1)][i]["ids"] for i in ids)
L += ["", "## 출력 동일성", "", "| 팔 | 모든 회차 출력 같음 |", "|---|---|"]
for a in (u, d):
    L.append(f"| {name(a)} | {same_rep[a]} / {len(ids)} |")
L += ["", f"두 팔의 출력이 토큰까지 같은 문제: {same_ud} / {len(ids)} (다른 경우는 커널 수치 차이로 argmax가 갈린 것)"]
open(os.path.join(RUN, "table.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
