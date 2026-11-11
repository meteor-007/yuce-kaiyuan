# -*- coding: utf-8 -*-
"""C 选择步骤的精确条件随机化检验：
每期从合规候选池随机抽 m 个三元组（m=主攻组数），用蒙特卡洛分布计算选定组≥2码率的
单期 z 值，再 Stouffer 合并；另报精确枚举（组合数小：C(20,m)）版本。
"""
import os
import sys
import json
import math
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
from kl8_core import load_dataset, truncated, TimeSeries  # noqa: E402
from kl8_methods import build_triples, triple_picks  # noqa: E402
from itertools import combinations

rng = np.random.default_rng(777)
OUT = os.path.join(ROOT, "output", "diagnostics")


def main():
    t0 = time.time()
    ds = load_dataset()
    periods = sorted(p for p in ds.point_sets if p in ds.draws_by_period)
    zs, obs_rates, enum_p = [], [], []
    for p in periods:
        tds = truncated(ds, p)
        ts = TimeSeries(tds)
        triples, _, _, _ = build_triples(tds, ts, {})
        pk = triple_picks(triples)
        groups = pk["groups"]
        actual = ds.draws_by_period[p].numbers
        rated = [x for x in triples if "低温弃子" not in x.get("labels", [])] or list(triples)
        win_vec = np.array([1.0 if len(set(x["members"]) & actual) >= 2 else 0.0 for x in rated])
        m = len(groups)
        obs_wins = sum(1 for g in groups if len(set(g["members"]) & actual) >= 2)
        obs_rate = obs_wins / m
        # exact enumeration of C(n,m) subsets
        n = len(rated)
        if m <= n:
            cnt = 0
            tot = 0
            ge = 0
            for idx in combinations(range(n), m):
                wins = win_vec[list(idx)].sum()
                tot += 1
                if wins >= obs_wins:
                    ge += 1
            enum_p.append(ge / tot)
        else:
            enum_p.append(1.0)
        # Monte Carlo null mean/sd
        B = 400
        means = np.empty(B)
        for b in range(B):
            idx = rng.choice(n, size=m, replace=False)
            means[b] = win_vec[idx].mean()
        sd = means.std(ddof=1)
        zs.append((obs_rate - means.mean()) / sd if sd > 0 else 0.0)
        obs_rates.append(obs_rate)
    zs = np.array(zs)
    Z = zs.sum() / math.sqrt(len(zs))
    p_two = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(Z)))
    enum_p = np.array(enum_p)
    # Fisher combination of exact per-period one-sided p
    from scipy import stats
    fisher_stat = -2 * np.log(np.clip(enum_p, 1e-12, 1)).sum()
    fisher_p = stats.chi2.sf(fisher_stat, 2 * len(enum_p))
    out = {
        "n_periods": len(periods),
        "stouffer_Z": float(Z),
        "stouffer_p_two_sided": float(p_two),
        "mean_period_z": float(zs.mean()),
        "exact_enum_mean_p": float(enum_p.mean()),
        "fisher_exact_p": float(fisher_p),
        "obs_rate_mean": float(np.mean(obs_rates)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    # subset windows
    idx299 = [i for i, p in enumerate(periods) if p >= 2026001]
    out["eval299"] = {
        "stouffer_Z": float(zs[idx299].sum() / math.sqrt(len(idx299))),
        "stouffer_p_two_sided": float(2 * (1 - stats.norm.cdf(abs(zs[idx299].sum() / math.sqrt(len(idx299)))))),
        "mean_period_z": float(zs[idx299].mean()),
        "fisher_exact_p": float(stats.chi2.sf(-2 * np.log(np.clip(enum_p[idx299], 1e-12, 1)).sum(), 2 * len(idx299))),
    }
    halves = [idx299[:len(idx299) // 2], idx299[len(idx299) // 2:]]
    out["halves_299"] = []
    for h in halves:
        out["halves_299"].append({
            "n": len(h),
            "mean_z": float(zs[h].mean()),
            "stouffer_p": float(2 * (1 - stats.norm.cdf(abs(zs[h].sum() / math.sqrt(len(h)))))),
        })
    print(json.dumps(out, ensure_ascii=False, indent=1))
    with open(os.path.join(OUT, "candidates_eval_C_randomization_20261110.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[saved] elapsed={out['elapsed_s']}s")


if __name__ == "__main__":
    sys.exit(main())