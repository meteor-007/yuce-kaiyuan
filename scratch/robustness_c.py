# -*- coding: utf-8 -*-
"""C 实验稳健性复核：分半稳定、全334期、构成校正零假设、分数分位校准。
只读生产代码，不改动。结果追加到 output/diagnostics/candidates_eval_20261110.json 的同目录文件。
"""
import os
import sys
import json
import math
import time
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
from kl8_core import load_dataset, truncated, TimeSeries  # noqa: E402
from kl8_methods import build_triples, triple_picks  # noqa: E402

OUT = os.path.join(ROOT, "output", "diagnostics")
rng = np.random.default_rng(20261111)


def comb_safe(n, k):
    if k < 0 or n < 0 or k > n:
        return 0
    return math.comb(n, k)


def p_ge2_composition(a, b, h):
    """三元组含 a 个场内号、b 个场外号；本期场内命中 h 个。
    精确 P(三码中≥2)（场内/场外分别无放回超几何）。"""
    total = 0.0
    for k in (2, 3):
        s = 0.0
        for i in range(0, min(a, k) + 1):
            j = k - i
            if j < 0 or j > b:
                continue
            pf = comb_safe(a, i) * comb_safe(20 - a, h - i) / comb_safe(20, h) if h <= 20 else 0.0
            po = (comb_safe(b, j) * comb_safe(60 - b, (20 - h) - j)
                  / comb_safe(60, 20 - h)) if 0 <= 20 - h <= 60 else 0.0
            s += pf * po
        total += s
    return total


def collect(ds, periods):
    rows = []
    for p in periods:
        tds = truncated(ds, p)
        ts = TimeSeries(tds)
        triples, _, _, _ = build_triples(tds, ts, {})
        pk = triple_picks(triples)
        groups = pk["groups"]
        actual = ds.draws_by_period[p].numbers
        field = set(ds.point_sets[p])
        h = len(field & actual)
        g_field = (math.comb(h, 2) * (20 - h) + math.comb(h, 3)) / math.comb(20, 3) if h >= 2 else 0.0
        rated = [x for x in triples if "低温弃子" not in x.get("labels", [])] or list(triples)
        q = sum(1 for x in rated if len(set(x["members"]) & actual) >= 2) / max(len(rated), 1)
        # exact composition-adjusted null for each selected group
        comp_null = []
        wins = 0
        sel_field = []
        for gr in groups:
            mem = set(gr["members"])
            a = len(mem & field)
            b = 3 - a
            sel_field.append(a)
            comp_null.append(p_ge2_composition(a, b, h))
            if len(mem & actual) >= 2:
                wins += 1
        m = len(groups)
        # score calibration: quartiles of eligible candidates by score
        ranked = sorted(rated, key=lambda x: -float(x["score"]))
        qcuts = np.array_split(np.array(range(len(ranked))), 4)
        qrate = []
        for cut in qcuts:
            qs = [ranked[i] for i in cut]
            qrate.append(sum(1 for x in qs if len(set(x["members"]) & actual) >= 2) / max(len(qs), 1))
        rows.append({
            "period": int(p), "m": int(m), "wins": int(wins),
            "rate": wins / m if m else float("nan"),
            "h": int(h), "g_field": g_field, "q_pool": float(q),
            "comp_null": float(np.mean(comp_null)) if comp_null else float("nan"),
            "sel_field_mean": float(np.mean(sel_field)) if sel_field else float("nan"),
            "pool_field_mean": float(np.mean([len(set(x["members"]) & field) for x in rated])),
            "qrate": [float(x) for x in qrate],
            "scores": [float(x["score"]) for x in ranked],
        })
    return rows


def perm_p(d, B=20000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, len(d)))
    return float((1 + (np.abs((signs * d).mean(axis=1)) >= obs).sum()) / (B + 1))


def boot_ci(a, B=10000):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    idx = rng.integers(0, len(a), size=(B, len(a)))
    m = a[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(rows, label, out):
    rate = np.array([r["rate"] for r in rows])
    g = np.array([r["g_field"] for r in rows])
    q = np.array([r["q_pool"] for r in rows])
    cn = np.array([r["comp_null"] for r in rows])
    d_field = rate - g
    d_pool = rate - q
    d_comp = rate - cn
    entry = {
        "label": label,
        "n_periods": len(rows),
        "groups_total": int(sum(r["m"] for r in rows)),
        "rate_mean": float(np.nanmean(rate)),
        "field_base_mean": float(np.nanmean(g)),
        "pool_base_mean": float(np.nanmean(q)),
        "comp_base_mean": float(np.nanmean(cn)),
        "pooled_rate": float(sum(r["wins"] for r in rows) / max(sum(r["m"] for r in rows), 1)),
        "d_pool_mean": float(np.nanmean(d_pool)),
        "d_pool_ci": boot_ci(d_pool),
        "d_pool_p": perm_p(d_pool),
        "d_comp_mean": float(np.nanmean(d_comp)),
        "d_comp_ci": boot_ci(d_comp),
        "d_comp_p": perm_p(d_comp),
        "d_field_mean": float(np.nanmean(d_field)),
        "d_field_p": perm_p(d_field),
        "sel_field_mean": float(np.nanmean([r["sel_field_mean"] for r in rows])),
        "pool_field_mean": float(np.nanmean([r["pool_field_mean"] for r in rows])),
    }
    # score quartile calibration
    qr = np.array([r["qrate"] for r in rows])
    entry["quartile_ge2_rates_top_to_bottom"] = [float(x) for x in qr.mean(axis=0)]
    entry["quartile_lift_vs_pool"] = [float(x) for x in qr.mean(axis=0) / np.nanmean(q)]
    out.append(entry)
    print(f"\n[{label}] periods={entry['n_periods']} groups={entry['groups_total']}")
    print(f"  rate={entry['rate_mean']:.4f} field_base={entry['field_base_mean']:.4f} "
          f"pool_base={entry['pool_base_mean']:.4f} comp_base={entry['comp_base_mean']:.4f}")
    print(f"  d_pool={entry['d_pool_mean']:+.4f} CI{entry['d_pool_ci']} p={entry['d_pool_p']:.4f}")
    print(f"  d_comp={entry['d_comp_mean']:+.4f} CI{entry['d_comp_ci']} p={entry['d_comp_p']:.4f}")
    print(f"  sel_field_mean={entry['sel_field_mean']:.2f} vs pool_field_mean={entry['pool_field_mean']:.2f}")
    print(f"  score quartiles(top->bottom) >=2 rate: {entry['quartile_ge2_rates_top_to_bottom']}")
    return entry


def main():
    t0 = time.time()
    ds = load_dataset()
    periods_all = sorted(p for p in ds.point_sets if p in ds.draws_by_period)
    print(f"[data] T={ds.T} point periods={len(periods_all)}")
    out = []
    rows334 = collect(ds, periods_all)
    print(f"[collect all] {time.time()-t0:.1f}s")
    report(rows334, "all_point_periods_334", out)
    rows299 = [r for r in rows334 if r["period"] >= 2026001]
    report(rows299, "eval_299", out)
    half = len(rows299) // 2
    report(rows299[:half], "eval_299_first_half", out)
    report(rows299[half:], "eval_299_second_half", out)
    # 40-day recent window for comparison with the report panel
    report(rows299[-40:], "recent_40", out)
    with open(os.path.join(OUT, "candidates_eval_robustness_C_20261110.json"), "w", encoding="utf-8") as f:
        json.dump({"entries": out, "elapsed_s": round(time.time() - t0, 1)}, f,
                  ensure_ascii=False, indent=1)
    print(f"[saved] robustness json; elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())