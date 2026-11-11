# -*- coding: utf-8 -*-
"""补充诊断：三元组超额是否稳健/是否选择偏差；点位-开奖频次检查；半样本拆分。"""
import json
import numpy as np
from kl8_core import load_dataset, clustered_rate_ci, ring_prev, ring_next
from kl8_report import fast_replay, hyper_triple_probs
from kl8_methods import build_triples, select_main_groups
from kl8_core import TimeSeries, truncated

ds = load_dataset()
periods = sorted(set(ds.point_sets) & set(ds.draws_by_period))
p_all3, p_ge2 = hyper_triple_probs(3)

# --- A. all 20 candidate ring triples vs selected main groups, per period
all_hit2, all_n = [], []
t1_hit2, t1_n, t2_hit2, t2_n = [], [], [], []
sel_hit2, sel_n = [], []
for p in periods:
    draw = ds.draws_by_period[p].numbers
    tds = truncated(ds, p)
    ts = TimeSeries(tds)
    triples, _ext, _cs, info = build_triples(tds, ts, {})
    cnt = 0
    for t in triples:
        cnt += 1 if len(set(t["members"]) & draw) >= 2 else 0
    all_hit2.append(cnt); all_n.append(len(triples))
    groups = select_main_groups(triples)
    sh = 0
    t1h = t1c = t2h = t2c = 0
    for g in groups:
        hit = len(set(g["members"]) & draw) >= 2
        sh += hit
        if g.get("tier") == "T1核心":
            t1h += hit; t1c += 1
        elif g.get("tier") == "T2防守":
            t2h += hit; t2c += 1
    sel_hit2.append(sh); sel_n.append(len(groups))
    t1_hit2.append(t1h); t1_n.append(t1c)
    t2_hit2.append(t2h); t2_n.append(t2c)

def rep(name, kt, nt):
    ci = clustered_rate_ci(kt, nt, p0=p_ge2)
    print(f"{name}: {sum(kt)}/{sum(nt)}={ci['rate']*100:.3f}% clusteredCI[{ci['lo']*100:.3f},{ci['hi']*100:.3f}] "
          f"lift_vs_15.26={(ci['rate']/p_ge2):.3f}")
    return {**ci, "lift": ci["rate"] / p_ge2}

out = {}
out["all_candidates"] = rep("ALL 20 candidates", all_hit2, all_n)
out["selected"] = rep("selected main groups", sel_hit2, sel_n)
out["T1"] = rep("T1 groups", t1_hit2, t1_n)
out["T2"] = rep("T2 groups", t2_hit2, t2_n)

# --- B. half split
half = len(periods) // 2
for nm, sl in (("first_half", slice(0, half)), ("second_half", slice(half, None))):
    kt = np.array(sel_hit2)[sl]; nt = np.array(sel_n)[sl]
    out[nm] = rep(f"selected {nm} ({periods[sl][0]}..{periods[sl][-1]})", list(kt), list(nt))

# --- C. does group count / dedup size explain? selected rate by groups-per-period
for gn in (1, 2, 3, 4):
    kt = [h for h, n in zip(sel_hit2, sel_n) if n == gn]
    nt = [n for n in sel_n if n == gn]
    if nt:
        print(f"g_n={gn}: periods={len(nt)} hit2={sum(kt)}/{sum(nt)}={sum(kt)/sum(nt)*100:.2f}%")

# --- D. empirical number frequency & point selection
H = ds.draw_matrix()
freq = H.mean(axis=0)  # 80
print("draw freq: min %.4f max %.4f mean %.4f sd %.4f" % (freq.min(), freq.max(), freq.mean(), freq.std()))
# chi-square vs 0.25
chi2 = (((H.sum(axis=0) - H.shape[0] * 0.25) ** 2) / (H.shape[0] * 0.25)).sum()
print("chi2(80) =", chi2, "(df=79, 95%% crit ~100.7)")
# point inclusion frequency
inc = np.zeros(80)
for p in periods:
    for n in ds.point_sets[p]:
        inc[n - 1] += 1
inc = inc / len(periods)
print("point inclusion: min %.3f max %.3f mean %.4f" % (inc.min(), inc.max(), inc.mean()))
corr = np.corrcoef(inc, freq)[0, 1]
print("corr(point inclusion, draw freq) =", corr)

# --- E. is the point set itself correlated with the SAME draw (contemporaneous)?
core_rate = sum(len(ds.point_sets[p] & ds.draws_by_period[p].numbers) for p in periods) / (20 * len(periods))
print("core contemporaneous rate:", core_rate)

# --- F. permutation with day-clustered resampling of selected groups (sign-flip of membership?)
# sanity: rate of >=2 for a random triple from 1..80 if draws are the empirical draws
rng = np.random.default_rng(7)
B = 200
tot = int(np.sum(sel_n)); obs = int(np.sum(sel_hit2))
rates = []
for b in range(B):
    acc = 0
    for i, p in enumerate(periods):
        m = np.zeros(81, bool); m[list(ds.draws_by_period[p].numbers)] = True
        gn = int(sel_n[i])
        idx = np.argsort(rng.random((gn, 80)), axis=1)[:, :3] + 1
        acc += int((m[idx].sum(axis=1) >= 2).sum())
    rates.append(acc / tot)
rates = np.array(rates)
print(f"perm rand80 (selected count): mean={rates.mean()*100:.3f}% sd={rates.std(ddof=1)*100:.3f}% "
      f"obs={obs/tot*100:.3f}% p1={(1+(rates>=obs/tot).sum())/(B+1):.3f}")

# --- G. permutation null using groups drawn as ring triples around random 20-number point sets
# (tests whether ring-neighborhood structure of arbitrary point sets matters)
rates_ring = []
for b in range(B):
    acc = 0
    for i, p in enumerate(periods):
        m = np.zeros(81, bool); m[list(ds.draws_by_period[p].numbers)] = True
        gn = int(sel_n[i])
        pts = rng.choice(80, size=20, replace=False) + 1  # random point set
        idx = np.array([[(x - 2) % 80 + 1, x, x % 80 + 1] for x in rng.choice(pts, size=gn, replace=False)])
        acc += int((m[idx].sum(axis=1) >= 2).sum())
    rates_ring.append(acc / tot)
rates_ring = np.array(rates_ring)
print(f"perm ring-around-random-points: mean={rates_ring.mean()*100:.3f}% sd={rates_ring.std(ddof=1)*100:.3f}% "
      f"p1={(1+(rates_ring>=obs/tot).sum())/(B+1):.3f}")

out["perm_selected"] = {"mean": float(rates.mean()), "sd": float(rates.std(ddof=1)), "obs": obs / tot}
out["perm_ring_random_points"] = {"mean": float(rates_ring.mean()), "sd": float(rates_ring.std(ddof=1)), "obs": obs / tot}
with open("scratch/hitrate_extra.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)