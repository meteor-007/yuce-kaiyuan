# -*- coding: utf-8 -*-
"""命中率深度归因分析（只读数据；结果写 scratch/）

量化：结构性随机 vs 模型/实现问题
- 超几何理论基线
- recent_hits 40/100/200 面板
- 300 期回测（JSON + eval_triples_walkforward）
- 点位源上限（core / extension / 主攻三元组 / 随机三元组反事实）
- 噪声与回归均值
"""
import json
import math
import time
import numpy as np

from kl8_core import load_dataset, clustered_rate_ci, ring_prev, ring_next
from kl8_report import recent_hits, fast_replay, recent_periods, hyper_triple_probs

T0 = time.time()
ds = load_dataset()
periods = sorted(set(ds.point_sets) & set(ds.draws_by_period))
print(f"dataset: draws={ds.T} point_periods={len(periods)} range={periods[0]}..{periods[-1]}")

result = {}

# ---------------------------------------------------------------- 1. baseline
p0 = 20.0 / 80.0
N = 80.0
def hyper_var(k):
    return k * p0 * (1 - p0) * ((N - k) / (N - 1))

var9, var20 = hyper_var(9), hyper_var(20)
sd9, sd20 = math.sqrt(var9), math.sqrt(var20)
se = lambda sd, n: sd / math.sqrt(n)
result["baseline"] = {
    "expected_per_number": 0.25,
    "expected_hits_k9": round(9 * p0, 6),
    "hypergeom_sd_k9": round(sd9, 6),
    "var_k9": round(var9, 6),
    "var_k20": round(var20, 6),
    "sd_k20": round(sd20, 6),
    "se_k9": {str(n): round(se(sd9, n), 6) for n in (40, 100, 200)},
    "se_k20": {str(n): round(se(sd20, n), 6) for n in (40, 100, 200)},
}
print("baseline:", result["baseline"])

# exact triple probabilities
p_all3, p_ge2 = hyper_triple_probs(3)
# hypergeometric win odds for 5-number pick (KL8 选五)
from math import comb
pick_probs = {}
for k in (5, 4, 3, 0):
    pick_probs[k] = comb(5, k) * comb(75, 20 - k) / comb(80, 20)
pick_probs["ge4"] = pick_probs[5] + pick_probs[4]
pick_probs["ge3"] = pick_probs["ge4"] + pick_probs[3]
print("p_all3", p_all3, "p_ge2", p_ge2, "pick5", pick_probs)
result["triple_probs"] = {"p_all3": p_all3, "p_ge2": p_ge2, "pick5_probs": pick_probs}

# ---------------------------------------------------------------- 2. panel
panels = {}
for n in (40, 100, 200):
    rows, agg = recent_hits(ds, n)
    ci = agg["main_ci"]
    panels[n] = {
        "n_periods": agg["n"], "first": agg["first"], "last": agg["last"],
        "hits": agg["main_h"], "total": agg["main_n"],
        "rate": agg["main_rate"], "ci": ci,
        "lift": agg["main_rate"] / 0.25,
        "se_clustered": (ci[1] - ci[0]) / (2 * 1.96),
        "se_binomial": math.sqrt(0.25 * 0.75 / agg["main_n"]),
        "core_per_period": agg["core"], "core_n": agg["core_n"],
        "ext_rate": agg["ext_rate"], "ext_n_avg": agg["ext_n_avg"],
        "group_hit2_per_period": agg["group_hit2"], "group_n_per_period": agg["group_n"],
        "group_full3_per_period": agg["group_full3"],
        "z16_per_period": agg["z16"], "z16_n": agg["z16_n"],
    }
    print(f"panel {n}: n={agg['n']} {agg['main_h']}/{agg['main_n']}={agg['main_rate']*100:.2f}% "
          f"CI[{ci[0]*100:.2f},{ci[1]*100:.2f}] lift={agg['main_rate']/0.25:.3f} "
          f"core={agg['core']:.2f}/{agg['core_n']:.0f} ext={agg['ext_rate']*100:.2f}%")
result["panels"] = panels

# ---------------------------------------------------------------- 3. backtest
bt = json.load(open("output/backtest/kl8_backtest_result.json", encoding="utf-8"))
import os, datetime
mtime = os.path.getmtime("output/backtest/kl8_backtest_result.json")
data_mtime = max(os.path.getmtime("daily_points.txt"), os.path.getmtime("kl8_history_final.txt"))
print("backtest json mtime", datetime.datetime.fromtimestamp(mtime), "data mtime", datetime.datetime.fromtimestamp(data_mtime))
result["backtest_json"] = bt
result["backtest_json_fresh"] = bool(mtime >= data_mtime - 1)

from kl8_backtest import eval_triples_walkforward
tw = eval_triples_walkforward(ds, periods=300)
print("triple walkforward:", json.dumps(tw, ensure_ascii=False, indent=1))
result["triple_walkforward"] = tw

# ---------------------------------------------------------------- 4. point source
core_hits, core_ns = [], []
ext_hits, ext_ns = [], []
g_hit2s, g_ns, g_full3s = [], [], []
dedup_hits, dedup_ns = [], []
z16_hits, z16_ns = [], []
replay_rows = []
for p in periods:
    r = fast_replay(ds, p)
    assert r is not None, p
    replay_rows.append(r)
    draw = ds.draws_by_period[p].numbers
    core = ds.point_sets[p]
    core_hits.append(len(core & draw))
    core_ns.append(20)
    ext_hits.append(r["ext_hit"])
    ext_ns.append(r["ext_n"])
    g_hit2s.append(r["g_hit2"])
    g_ns.append(r["g_n"])
    g_full3s.append(r["g_full3"])
    dedup_hits.append(r["g_dedup"])
    dedup_ns.append(r["g_dedup_n"])
    z16_hits.append(r["z16"])
    z16_ns.append(r["z16_n"])

nP = len(periods)
core_ci = clustered_rate_ci(core_hits, core_ns)
ext_ci = clustered_rate_ci(ext_hits, ext_ns)
tri_ci = clustered_rate_ci(g_hit2s, g_ns, p0=p_ge2)
dedup_ci = clustered_rate_ci(dedup_hits, dedup_ns)
z16_ci = clustered_rate_ci(z16_hits, z16_ns)
print(f"point source: n={nP} core={core_ci['rate']*100:.3f}% CI[{core_ci['lo']*100:.3f},{core_ci['hi']*100:.3f}] "
      f"ext={ext_ci['rate']*100:.3f}% CI[{ext_ci['lo']*100:.3f},{ext_ci['hi']*100:.3f}]")
print(f"triples: >=2 rate={tri_ci['rate']*100:.3f}% CI[{tri_ci['lo']*100:.3f},{tri_ci['hi']*100:.3f}] base=15.26% "
      f"full3={sum(g_full3s)}/{sum(g_ns)}={sum(g_full3s)/sum(g_ns)*100:.3f}% base=1.3875%")
print(f"dedup: {sum(dedup_hits)}/{sum(dedup_ns)}={dedup_ci['rate']*100:.3f}% CI[{dedup_ci['lo']*100:.3f},{dedup_ci['hi']*100:.3f}]")
print(f"z16: {sum(z16_hits)}/{sum(z16_ns)}={z16_ci['rate']*100:.3f}% CI[{z16_ci['lo']*100:.3f},{z16_ci['hi']*100:.3f}]")

# window snapshots of core / ext / triples for noise discussion
def window_stats(kt, nt, k):
    sl = slice(len(kt) - k, len(kt))
    ci = clustered_rate_ci(kt[sl], nt[sl])
    return {"n": k, "hits": int(sum(kt[sl])), "total": int(sum(nt[sl])),
            "rate": ci["rate"], "ci": [ci["lo"], ci["hi"]], "nperiods": ci["n_periods"]}
core_windows = {str(k): window_stats(core_hits, core_ns, k) for k in (40, 100, 200)}
tri_windows = {str(k): window_stats(g_hit2s, g_ns, k) for k in (40, 100, 200)}
dedup_windows = {str(k): window_stats(dedup_hits, dedup_ns, k) for k in (40, 100, 200)}
print("core windows:", {k: f"{v['hits']}/{v['total']}={v['rate']*100:.2f}%" for k, v in core_windows.items()})
print("dedup windows:", {k: f"{v['hits']}/{v['total']}={v['rate']*100:.2f}%" for k, v in dedup_windows.items()})
print("tri windows:", {k: f"{v['hits']}/{v['total']}={v['rate']*100:.2f}%" for k, v in tri_windows.items()})

# random-triple counterfactual (fixed seed, 200 permutations)
B = 200
rng = np.random.default_rng(20261110)
draw_masks = {}
for p in periods:
    m = np.zeros(81, dtype=bool)
    m[list(ds.draws_by_period[p].numbers)] = True
    draw_masks[p] = m
point_arrays = {p: np.array(sorted(ds.point_sets[p]), dtype=int) for p in periods}
gn_arr = np.array(g_ns, dtype=int)
total_groups = int(gn_arr.sum())
obs_hit2 = int(sum(g_hit2s))
obs_rate = obs_hit2 / total_groups

rand80_rates, randpt_rates = [], []
for b in range(B):
    tot80 = totpt = 0
    for i, p in enumerate(periods):
        gn = int(gn_arr[i])
        dm = draw_masks[p]
        # uniform 3-subset of 1..80 per row via argsort of random ranks (distinct within row)
        idx = np.argsort(rng.random((gn, 80)), axis=1)[:, :3] + 1
        tot80 += int((dm[idx].sum(axis=1) >= 2).sum())
        # uniform 3-subset of the 20-point field
        pts = point_arrays[p]
        jdx = np.argsort(rng.random((gn, 20)), axis=1)[:, :3]
        tri_pts = pts[jdx]
        totpt += int((dm[tri_pts].sum(axis=1) >= 2).sum())
    rand80_rates.append(tot80 / total_groups)
    randpt_rates.append(totpt / total_groups)

r80 = np.array(rand80_rates)
rpt = np.array(randpt_rates)
def perm_p(perms, obs):
    p_hi = (1 + int((perms >= obs).sum())) / (len(perms) + 1)
    p_lo = (1 + int((perms <= obs).sum())) / (len(perms) + 1)
    return p_hi, p_lo, min(1.0, 2 * min(p_hi, p_lo))
phi80, plo80, p2_80 = perm_p(r80, obs_rate)
phipt, plopt, p2_pt = perm_p(rpt, obs_rate)
print(f"counterfactual: observed {obs_hit2}/{total_groups}={obs_rate*100:.3f}% | "
      f"rand80 mean={r80.mean()*100:.3f}% sd={r80.std(ddof=1)*100:.3f} p1={phi80:.3f} p2={p2_80:.3f} | "
      f"randpt mean={rpt.mean()*100:.3f}% sd={rpt.std(ddof=1)*100:.3f} p1={phipt:.3f} p2={p2_pt:.3f}")

# exact analytic null for >=2 per group and its aggregate SE (binomial reference)
null_mean = total_groups * p_ge2
null_sd = math.sqrt(total_groups * p_ge2 * (1 - p_ge2))
z_tri = (obs_hit2 - null_mean) / null_sd
print(f"triple obs vs hypergeom null: expected {null_mean:.2f}, sd {null_sd:.2f}, z={z_tri:.2f}")

result["point_source"] = {
    "n_periods": nP,
    "core": {**core_ci, "windows": core_windows},
    "extension": {**ext_ci, "windows": {str(k): window_stats(ext_hits, ext_ns, k) for k in (40, 100, 200)}},
    "triples": {**tri_ci, "full3_total": int(sum(g_full3s)), "group_total": total_groups,
                "full3_rate": sum(g_full3s) / total_groups, "windows": tri_windows,
                "z_vs_hypergeom": z_tri},
    "dedup": {**dedup_ci, "windows": dedup_windows},
    "z16": z16_ci,
    "counterfactual": {
        "B": B, "seed": 20261110, "observed_hit2": obs_hit2, "observed_rate": obs_rate,
        "rand80_mean": float(r80.mean()), "rand80_sd": float(r80.std(ddof=1)),
        "p_one_sided_80": phi80, "p_two_sided_80": p2_80,
        "randpt_mean": float(rpt.mean()), "randpt_sd": float(rpt.std(ddof=1)),
        "p_one_sided_pt": phipt, "p_two_sided_pt": p2_pt,
    },
}

# ---------------------------------------------------------------- 5. noise
p40 = panels[40]
z40_clu = (p40["rate"] - 0.25) / p40["se_clustered"]
z40_bin = (p40["rate"] - 0.25) / p40["se_binomial"]
z100_clu = (panels[100]["rate"] - 0.25) / panels[100]["se_clustered"]
z200_clu = (panels[200]["rate"] - 0.25) / panels[200]["se_clustered"]
print(f"noise: z40 clustered={z40_clu:.2f} binomial={z40_bin:.2f}; z100={z100_clu:.2f}; z200={z200_clu:.2f}")
result["noise"] = {
    "z40_clustered": z40_clu, "z40_binomial": z40_bin,
    "z100_clustered": z100_clu, "z200_clustered": z200_clu,
    "rates": {str(k): panels[k]["rate"] for k in (40, 100, 200)},
    "cis": {str(k): panels[k]["ci"] for k in (40, 100, 200)},
}

# edge scan summary
es = json.load(open("output/diagnostics/edge_scan_latest.json", encoding="utf-8"))
result["edge_scan"] = {"n_tests": es["n_tests"], "T": es["T"],
                       "n_pass_ci": len(es["candidates_ci"]),
                       "n_pass_fdr": len(es["candidates_fdr"]),
                       "n_stable": len(es["half_validation"])}

# ablation summary from backtest json
result["ablation"] = bt.get("ablation", {})

result["elapsed_s"] = time.time() - T0
with open("scratch/hitrate_analysis_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=float)
print("elapsed", result["elapsed_s"])