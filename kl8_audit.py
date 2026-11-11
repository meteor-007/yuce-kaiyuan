# -*- coding: utf-8 -*-
"""模块耗时与有效性取证脚本（诊断用，可重复运行）。

输出：
1) 每个分析模块的 pipeline 单次耗时（ms）
2) 各模块输出向量与「是否被 H1–H7 评分实际引用」的接线表
3) 各模块单维 Top5 的 walk-forward 命中（近 120 期真实点位期）
"""
from __future__ import annotations

import sys
import time
import numpy as np

from kl8_core import CONFIG, N_NUM, TimeSeries, load_dataset, truncated
import kl8_methods as M
from kl8_report import H_DEF, run_pipeline, top_idx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

USED = sorted({c for _, cols, _ in H_DEF for c in cols})


def timing_one(fn, ds, ts, ctx):
    t0 = time.perf_counter()
    try:
        out = fn(ds, ts, ctx)
    except Exception as e:  # pragma: no cover
        return (time.perf_counter() - t0) * 1000, f"ERR {type(e).__name__}: {e}", None
    return (time.perf_counter() - t0) * 1000, "ok", out


def main(periods: int = 120) -> int:
    ds_full = load_dataset()
    target = ds_full.target_period
    ds = truncated(ds_full, target)
    ts = TimeSeries(ds)
    print(f"历史期数 T={ds.T}  目标期={target}")

    # ---------- 1) 单模块耗时 ----------
    print("\n[T1] 单模块耗时（基于最新一期截断数据，单次冷启动）")
    order = [
        ("m_cluster", M.m_cluster), ("m_markov", M.m_markov),
        ("m_fib", M.m_fib), ("m_linalg", M.m_linalg), ("m_complex", M.m_complex),
        ("m_graph", M.m_graph), ("m_volatility", M.m_volatility),
        ("m_entropy", M.m_entropy), ("m_calc_zone", M.m_calc_zone),
        ("m_adversarial", M.m_adversarial),
    ]
    shared_ctx = {"feat": None}
    rows = []
    for name, fn in order:
        ms_list = []
        for _ in range(3):
            ctx = {}
            ms, st, out = timing_one(fn, ds, ts, ctx)
            ms_list.append(ms)
        rows.append((name, min(ms_list), max(ms_list)))
    full = {}
    t0 = time.perf_counter()
    run_pipeline(ds)
    full_ms = (time.perf_counter() - t0) * 1000
    for name, lo, hi in rows:
        print(f"  {name:14s} {lo:8.1f}-{hi:8.1f} ms")
    print(f"  {'pipeline(全)':14s} {full_ms:8.1f} ms")

    # ---------- 2) 接线表：模块输出是否真正参与评分 ----------
    print("\n[T2] 特征列 → H1-H7 接线检查")
    feat, ctx = run_pipeline(ds)
    cols = sorted(feat.cols.keys())
    for c in cols:
        v = feat.cols[c]
        nan = float(np.isnan(v).mean())
        sd = float(np.nanstd(v))
        where = [n for n, cs, _ in H_DEF if c in cs]
        print(f"  {c:14s} std={sd:6.3f} nan%={nan*100:5.1f} 用于={','.join(where) if where else '★未接入任何评分维度'}")

    # ---------- 3) 单维 walk-forward 命中 ----------
    if periods <= 0:
        return 0
    print(f"\n[T3] 单维 Top5 walk-forward 命中（近 {periods} 个真实点位期，基线 1.25）", flush=True)
    periods_list = [p for p in sorted(ds_full.point_sets) if p in ds_full.draws_by_period]
    periods_list = periods_list[-periods:]
    dims = {}
    for t in periods_list:
        tds = truncated(ds_full, t)
        _f, c = run_pipeline(tds, point_period=t)
        actual = ds_full.draws_by_period[t].numbers
        for hn in c["H"]:
            dims.setdefault(hn, []).append(len(set(top_idx(c["H"][hn], 5)) & actual))
        dims.setdefault("_综合", []).append(len(set(top_idx(c["composite"], 5)) & actual))
    base = 5 * 0.25
    for hn, hits in dims.items():
        a = np.array(hits, float)
        se = a.std(ddof=1) / np.sqrt(len(a))
        lo, hi = a.mean() - 1.96 * se, a.mean() + 1.96 * se
        print(f"  {hn:8s} n={len(a):3d} 命中={a.mean():.3f} lift={a.mean()/base:.3f} CI[{lo:.2f},{hi:.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 120))
