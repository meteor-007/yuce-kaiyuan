# -*- coding: utf-8 -*-
"""walk-forward 回测：Top5 / Top20 / 五码 / 5+2 / 三元组Top5 命中率 vs 随机基线
用法: python kl8_backtest.py [--periods 300] [--start 1500]
"""
import sys
import os
import json
import time
import argparse
import numpy as np
from scipy import stats

from kl8_core import (CONFIG, load_dataset, Dataset, truncated,
                      backtest_result_path, backtest_stash_path, resolve_existing)
from kl8_report import run_pipeline, H_DEF

BASE = {5: 1.25, 20: 5.0, 7: 1.75}


def picks(comp, H, feat, ctx):
    """回测策略组合选号生成。
    <p>
    【业务背景与架构定位】:
    根据当前期的各维特征与综合分，生成 Top5、Top20、核心五码、5+2 推荐及三元组 Top5 选号集合，
    逻辑与报告端 100% 契约保真对齐。
    
    @param comp 综合评分向量 (80维)
    @param H    H1-H7 各维度评分字典
    @param feat Feat80 容器
    @param ctx  分析上下文
    @return dict 包含各策略推荐号码集合
    """
    top5 = list(np.argsort(-comp)[:5] + 1)
    top20 = list(np.argsort(-comp)[:20] + 1)
    from kl8_methods import triple_picks
    tri = ctx["triple_info"]["triples"]
    tp = triple_picks(tri, fill=list(np.argsort(-comp) + 1))
    cluster5 = list(np.argsort(-feat.norm["cluster"])[:5] + 1)
    return {"top5": top5, "top20": top20, "five": tp["five"], "seven": tp["seven"],
            "cluster5": cluster5}


def eval_triples_walkforward(ds, periods=300):
    """只重建三元集群（不跑七维），评估五码/5+2/当日主攻组去重号。"""
    from kl8_core import TimeSeries
    from kl8_methods import build_triples, triple_picks
    T = ds.T
    hi = T - 1
    lo = max(0, hi - periods)
    h5, h7, hg = [], [], []
    for t in range(lo, hi):
        period = ds.draws[t].period
        if period not in ds.point_sets:
            continue
        tds = truncated(ds, period)
        ts = TimeSeries(tds)
        triples, _, _, _ = build_triples(tds, ts, {})
        pk = triple_picks(triples)
        actual = ds.draws[t].numbers
        h5.append(len(set(pk["five"]) & actual))
        h7.append(len(set(pk["seven"]) & actual))
        group_nums = set()
        for g in pk["groups"]:
            group_nums.update(g["members"])
        # 与推荐一致：主攻组是实际出号用到的组，命中按该集合计
        hg.append((len(group_nums & actual), max(len(group_nums), 1)))
    g_hits = [a for a, _ in hg]
    g_k = [b for _, b in hg]
    g_base = float(np.mean(g_k)) * 0.25 if g_k else 0.25
    arr = np.array(g_hits, dtype=float)
    mean = float(arr.mean()) if len(arr) else 0.0
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    _, p = stats.ttest_1samp(arr, g_base) if len(arr) > 1 else (0.0, 1.0)
    g_sum = {
        "name": "triple_groups", "n": len(arr),
        "k": float(np.mean(g_k)) if g_k else 0.0,
        "base": g_base, "mean": mean,
        "lift": (mean / g_base) if g_base else 0.0,
        "ci": [mean - 1.96 * se, mean + 1.96 * se],
        "p": float(p),
    }
    return {
        "n": len(h5),
        "five": summarize("triple_five", h5, 5),
        "seven": summarize("triple_seven", h7, 7),
        "groups": g_sum,
    }


def eval_calc_zones_walkforward(ds, periods=300):
    """微积分八区分区：每区优2→16码 walk-forward 命中。"""
    from kl8_core import TimeSeries, CONFIG
    from kl8_methods import build_calc_zones
    T = ds.T
    W = int(CONFIG["window_calc"])
    hi = T - 1
    lo = max(W, hi - periods)
    hits16, zone_any = [], []
    for t in range(lo, hi):
        period = ds.draws[t].period
        tds = truncated(ds, period)
        if tds.T < W:
            continue
        ts = TimeSeries(tds)
        zones, picks16, _, _ = build_calc_zones(tds, ts, {})
        actual = ds.draws[t].numbers
        hits16.append(len(set(picks16) & actual))
        zone_any.append(sum(1 for z in zones if set(z["top2"]) & actual))
    s16 = summarize("calc_zone16", hits16, 16)
    za = np.asarray(zone_any, dtype=float)
    za_base = 8 * (1 - 0.75 ** 2)
    za_mean = float(za.mean()) if len(za) else 0.0
    za_se = float(za.std(ddof=1) / np.sqrt(len(za))) if len(za) > 1 else 0.0
    _, za_p = stats.ttest_1samp(za, za_base) if len(za) > 1 else (0.0, 1.0)
    return {
        "sixteen": s16,
        "zones_hit": {
            "name": "calc_zones_with_hit",
            "n": len(za),
            "mean": za_mean,
            "base": za_base,
            "lift": (za_mean / za_base) if za_base else 0.0,
            "ci": [za_mean - 1.96 * za_se, za_mean + 1.96 * za_se],
            "p": float(za_p),
        },
    }


def summarize(name, hits, k):
    arr = np.array(hits, dtype=float)
    base = BASE.get(k, k * 0.25)
    mean = arr.mean()
    sd = arr.std(ddof=1) if len(arr) > 1 else 0.0
    se = sd / np.sqrt(len(arr)) if len(arr) > 0 else 0.0
    t, p = stats.ttest_1samp(arr, base) if len(arr) > 1 else (0.0, 1.0)
    return {"name": name, "n": len(arr), "k": k, "base": base, "mean": float(mean),
            "lift": float(mean / base) if base > 0 else 0.0,
            "ci": [float(mean - 1.96 * se), float(mean + 1.96 * se)],
            "p": float(p)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def split_eval_periods(ds, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, max_per_split=None):
    """Return contiguous non-overlapping evaluation windows using only real scored periods."""
    periods = sorted(p for p in ds.point_sets if p in ds.draws_by_period)
    if len(periods) < 3:
        raise ValueError(f"Need at least 3 real scored periods, got {len(periods)}")
    if max_per_split is not None:
        max_per_split = max(1, int(max_per_split))
        periods = periods[-max_per_split * 3:]
    total = train_ratio + valid_ratio + test_ratio
    train_n = max(1, int(round(len(periods) * (train_ratio / total))))
    valid_n = max(1, int(round(len(periods) * (valid_ratio / total))))
    test_n = max(1, len(periods) - train_n - valid_n)
    while train_n + valid_n + test_n > len(periods):
        test_n -= 1
    while train_n + valid_n + test_n < len(periods):
        test_n += 1
    train = periods[:train_n]
    valid = periods[train_n:train_n + valid_n]
    test = periods[train_n + valid_n:train_n + valid_n + test_n]
    if not (train and valid and test):
        raise ValueError("Train/valid/test split produced empty segment")
    if set(train) & set(valid) or set(valid) & set(test) or set(train) & set(test):
        raise ValueError("Evaluation split overlaps across segments")
    return {"train": train, "valid": valid, "test": test}


def eval_strategy_on_periods(ds, periods, strategy_name="top5"):
    hits = []
    for period in periods:
        tds = truncated(ds, period)
        feat, ctx = run_pipeline(tds, point_period=period)
        actual = ds.draws_by_period[period].numbers
        comp = ctx["composite"]
        top5 = set(np.argsort(-comp)[:5] + 1)
        hits.append(len(top5 & actual))
    return summarize(strategy_name, hits, 5)


def strict_walkforward_eval(ds, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, max_per_split=12):
    split = split_eval_periods(ds, train_ratio, valid_ratio, test_ratio, max_per_split=max_per_split)
    out = {"split": split, "train": {}, "valid": {}, "test": {}}
    for label, periods in split.items():
        out[label] = eval_strategy_on_periods(ds, periods, strategy_name=f"top5_{label}")
    return out


def calibration(ds):
    """组件有效性校准：各信号分桶的条件命中率 vs 随机基线 0.25。

    审计修正：原实现三处分桶一律用 wilson() 把每个 (期, 号) 格子当独立样本，
    但快乐8每期恰好开出 20 个号（无放回抽样，组内负相关），独立口径的 CI 方法不成立。
    现统一改用 kl8_core.clustered_rate_ci 以「期」为聚类单元；
    连失分桶同时由 O(T×80) 的双层 Python 循环改为每期向量化，提速且口径不变。
    """
    from kl8_core import TimeSeries, ring_prev, ring_next, clustered_rate_ci
    ts = TimeSeries(ds)
    H, gap = ts.H, ts.gap
    T = ds.T
    out = {"baseline": 0.25}

    def _pack(kt, nt):
        r = clustered_rate_ci(kt, nt)
        return {"rate": r["rate"], "n": r["n"], "n_periods": r["n_periods"],
                "ci": [r["lo"], r["hi"]]}

    # 1) 连失分桶（用 t-1 时刻的连失，预测第 t 期命中数）
    bucket_defs = [
        ("0", lambda g: g == 0), ("1", lambda g: g == 1), ("2", lambda g: g == 2),
        ("3", lambda g: g == 3), ("4", lambda g: g == 4),
        ("5-6", lambda g: (g >= 5) & (g <= 6)),
        ("7-9", lambda g: (g >= 7) & (g <= 9)),
        ("10-14", lambda g: (g >= 10) & (g <= 14)),
        ("15+", lambda g: g >= 15),
    ]
    gaptab = {}
    for b, pred in bucket_defs:
        kt, nt = [], []
        for t in range(1, T):
            m = pred(gap[t - 1])
            cnt = int(m.sum())
            nt.append(cnt)
            kt.append(int(H[t, m].sum()) if cnt else 0)
        gaptab[b] = _pack(kt, nt)
    out["gap_buckets"] = gaptab
    # 2) 点位场（核心20 / 扩展环比）命中率，逐期序列
    core_kt, core_nt, ext_kt, ext_nt = [], [], [], []
    for p in sorted(ds.point_sets):
        if p not in ds.draws_by_period:
            continue
        ps = ds.point_sets[p]
        actual = ds.draws_by_period[p].numbers
        ext = set(ps)
        for m in ps:
            ext |= {ring_prev(m), ring_next(m)}
        core_kt.append(len(ps & actual)); core_nt.append(len(ps))
        ext_kt.append(len(ext & actual)); ext_nt.append(len(ext))
    out["field"] = {
        "core": _pack(core_kt, core_nt),
        "extended": _pack(ext_kt, ext_nt),
    }
    # 3) Markov P01 高分桶（滚动200期估计，取P01前30）
    W = 200
    kt30 = []
    for t in range(W, T):
        prev, cur = H[t - W:t - 1], H[t - W + 1:t]
        n0 = (prev == 0).sum(axis=0)
        p01 = (((prev == 0) & (cur == 1)).sum(axis=0) + 1) / (n0 + 2)
        kt30.append(int(H[t, np.argsort(-p01)[:30]].sum()))
    out["markov_top30"] = _pack(kt30, [30] * len(kt30))
    out["note"] = ("CI 为按期聚类的95%区间；下界>0.25 表示该信号存在可测正向edge，"
                   "否则与随机无显著差异")
    return out


def synth_dataset(T=200, alpha=1.2, mode="momentum", seed=7):
    """正对照合成数据。
    mode='momentum': 固定25个"热号"权重3.0（点位场=热号集）→ 动量类维度应能检出
    mode='omission': 命中概率∝连失（点位场=固定随机20号，与遗漏无关）→ 遗漏类维度应能检出
    """
    from kl8_core import Dataset, Draw
    rng = np.random.default_rng(seed)
    draws = []
    gap = np.full(80, 10.0)
    hot = set(range(1, 26))                       # 动量模式热号
    neutral_pts = frozenset(int(x) for x in rng.choice(np.arange(1, 81), 20, replace=False))
    for t in range(T):
        if mode == "momentum":
            w = np.ones(80) * 1.0
            w[[h - 1 for h in hot]] = 3.0
        else:
            w = 1.0 + alpha * np.minimum(gap, 15) / 15.0
        p = w / w.sum()
        nums = rng.choice(np.arange(1, 81), size=20, replace=False, p=p)
        draws.append(Draw(date="2000-01-01", period=900000 + t,
                          numbers=frozenset(int(x) for x in nums), idx=t))
        for n in range(1, 81):
            gap[n - 1] = 0 if n in nums else gap[n - 1] + 1
    ds = Dataset(draws=draws, draws_by_period={d.period: d for d in draws},
                 point_sets={}, point_dates={}, target_period=900000 + T)
    pts = frozenset(hot) if mode == "momentum" else neutral_pts
    for i in range(1, T):
        ds.point_sets[900000 + i] = pts if mode == "momentum" else (
            neutral_pts if mode == "omission" else draws[i - 1].numbers)
        ds.point_dates[900000 + i] = "2000-01-01"
    return ds


def run_synthetic(T=150, alpha=1.2, mode="momentum"):
    """正对照：分别检验 综合分 / 单维(H1动量 or H2遗漏) 能否检出植入规律"""
    ds = synth_dataset(T=T, alpha=alpha, mode=mode)
    res = {"composite": [], "single": []}
    for t in range(60, T):
        period = ds.draws[t].period
        tds = truncated(ds, period)
        feat, ctx = run_pipeline(tds, point_period=period)
        actual = ds.draws[t].numbers
        comp = ctx["composite"]
        key = "H1热度" if mode == "momentum" else "H2遗漏"
        single = ctx["H"][key]
        res["composite"].append(len(set(np.argsort(-comp)[:5] + 1) & actual))
        res["single"].append(len(set(np.argsort(-single)[:5] + 1) & actual))
    out = {"T": T, "alpha": alpha, "mode": mode, "n_eval": len(res["composite"])}
    for k, v in res.items():
        m = float(np.mean(v))
        sd = float(np.std(v, ddof=1) / np.sqrt(len(v)))
        t, p = stats.ttest_1samp(v, 1.25)
        out[k] = {"mean": m, "lift": m / 1.25, "ci": [m - 1.96 * sd, m + 1.96 * sd], "p": float(p)}
    return out


def blend_eval(stash_path=None):
    """用已存H离线评估 顺势榜/对冲榜/综合分 三套Top5（验正对照结论在真实数据上的表现）"""
    from kl8_report import TREND_DIMS, HEDGE_DIMS
    stash_path = stash_path or resolve_existing(backtest_stash_path(), "kl8_H_stash.npz")
    if not stash_path:
        raise FileNotFoundError("未找到 kl8_H_stash.npz（请先跑回测）")
    z = np.load(stash_path)
    Hs, hit = z["H"], z["hit"].astype(float)
    names = [n for n, _, _ in H_DEF]
    idx = {n: i for i, n in enumerate(names)}
    a_all = np.array([w for _, _, w in H_DEF])
    a_tr = np.zeros(len(names)); a_hd = np.zeros(len(names))
    for n, w in TREND_DIMS.items():
        a_tr[idx[n]] = w
    for n, w in HEDGE_DIMS.items():
        a_hd[idx[n]] = w
    out = {}
    for label, a in [("综合分", a_all), ("顺势榜", a_tr), ("对冲榜", a_hd)]:
        comp = np.tensordot(Hs, a, axes=([1], [0]))
        order = np.argsort(-comp, axis=1)[:, :5]
        h = np.take_along_axis(hit, order, axis=1).sum(axis=1)
        m = h.mean(); sd = h.std(ddof=1) / np.sqrt(len(h))
        t, p = stats.ttest_1samp(h, 1.25)
        out[label] = {"mean": float(m), "lift": float(m / 1.25),
                      "ci": [float(m - 1.96 * sd), float(m + 1.96 * sd)], "p": float(p)}
    # 简化样本外验证：前1/3选榜(比 lift)，后2/3按选定榜评估
    n3 = len(hit) // 3

    def top5_hits(a, lo, hi):
        comp = np.tensordot(Hs[lo:hi], a, axes=([1], [0]))
        order = np.argsort(-comp, axis=1)[:, :5]
        return np.take_along_axis(hit[lo:hi], order, axis=1).sum(axis=1)
    cands = {"综合分": a_all, "顺势榜": a_tr, "对冲榜": a_hd}
    best = max(cands, key=lambda k: top5_hits(cands[k], 0, n3).mean())
    h_oos = top5_hits(cands[best], n3, len(hit))
    m = h_oos.mean(); sd = h_oos.std(ddof=1) / np.sqrt(len(h_oos))
    t, p = stats.ttest_1samp(h_oos, 1.25)
    out["_选榜OOS"] = {"chosen": best, "n_test": len(h_oos), "mean": float(m),
                       "lift": float(m / 1.25),
                       "ci": [float(m - 1.96 * sd), float(m + 1.96 * sd)], "p": float(p)}
    return out


def _print_calib(c):
    print("[校准] 基线0.25（CI下界>0.25 表示存在可测正向edge）")
    for b, d in c["gap_buckets"].items():
        print(f"  连失{b:5s} 命中率{d['rate']:.3f} n={d['n']} CI[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]")
    print(f"  点位核心20 命中率{c['field']['core']['rate']:.3f} n={c['field']['core']['n']} CI[{c['field']['core']['ci'][0]:.3f},{c['field']['core']['ci'][1]:.3f}]")
    print(f"  扩展环比   命中率{c['field']['extended']['rate']:.3f} n={c['field']['extended']['n']} CI[{c['field']['extended']['ci'][0]:.3f},{c['field']['extended']['ci'][1]:.3f}]")
    print(f"  MarkovTop30 命中率{c['markov_top30']['rate']:.3f} n={c['markov_top30']['n']} CI[{c['markov_top30']['ci'][0]:.3f},{c['markov_top30']['ci'][1]:.3f}]")


def explore(stash_path=None, topn=5):
    """离线调权：前半段网格选权，后半段样本外验证（避免自我欺骗）"""
    stash_path = stash_path or resolve_existing(backtest_stash_path(), "kl8_H_stash.npz")
    if not stash_path:
        raise FileNotFoundError("未找到 kl8_H_stash.npz（请先跑回测）")
    z = np.load(stash_path)
    Hs, hit, lo = z["H"], z["hit"], int(z["lo"])   # (n,7,80), (n,80)
    n = Hs.shape[0]
    half = n // 2
    a0 = np.array([w for _, _, w in H_DEF])
    cands = {"基准(等权原版)": a0}
    for h1 in (0.0, 0.20):
        for h2 in (0.16, 0.26):
            for h6 in (0.10, 0.24):
                a = a0.copy()
                a[0] = h1; a[1] = h2; a[5] = h6
                if a.sum() > 0:
                    a = a / a.sum()
                cands[f"H1={h1:.2f}/H2={h2:.2f}/H6={h6:.2f}"] = a
    hits = hit.astype(float)

    def eval_w(a, lo_i, hi_i):
        comp = np.tensordot(Hs[lo_i:hi_i], a, axes=([1], [0]))  # (m,80)
        order = np.argsort(-comp, axis=1)[:, :topn]
        h = np.take_along_axis(hits[lo_i:hi_i], order, axis=1).sum(axis=1)
        return h

    base = topn * 0.25
    res = {}
    best_name, best_lift = None, -1
    for name, a in cands.items():
        h_in = eval_w(a, 0, half)
        lift_in = h_in.mean() / base
        res[name] = {"in_lift": float(lift_in), "weights": [float(x) for x in a]}
        if lift_in > best_lift:
            best_lift, best_name = lift_in, name
    h_oos = eval_w(cands[best_name], half, n)
    mean = h_oos.mean(); sd = h_oos.std(ddof=1); se = sd / np.sqrt(len(h_oos))
    t, p = stats.ttest_1samp(h_oos, base)
    res["_选优"] = {"name": best_name, "in_lift": float(best_lift),
                    "oos_mean": float(mean), "oos_lift": float(mean / base),
                    "oos_ci": [float(mean - 1.96 * se), float(mean + 1.96 * se)],
                    "oos_p": float(p), "n_tune": half, "n_test": n - half,
                    "weights": res[best_name]["weights"]}
    # 基准的样本外表现
    h_base = eval_w(a0, half, n)
    mb = h_base.mean(); sb = h_base.std(ddof=1) / np.sqrt(len(h_base))
    res["_基准OOS"] = {"oos_mean": float(mb), "oos_lift": float(mb / base),
                       "oos_ci": [float(mb - 1.96 * sb), float(mb + 1.96 * sb)]}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", type=int, default=300)
    ap.add_argument("--start", type=int, default=1500)
    ap.add_argument("--ablate", type=int, default=60)
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="正对照：合成数据检验框架能否检出已知规律")
    ap.add_argument("--syn-T", type=int, default=150)
    ap.add_argument("--syn-mode", default="momentum", choices=["momentum", "omission"])
    ap.add_argument("--strict-eval", action="store_true",
                    help="严格三段时间切分评估：train/valid/test，不允许重叠和伪点位")
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--valid-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--strict-eval-max", type=int, default=12,
                    help="每段最多评估多少个真实点位期，防止完整 HMM 在大数据上耗尽计算资源")
    args = ap.parse_args(argv)

    ds0 = load_dataset()
    if args.synthetic:
        for mode in (["momentum", "omission"] if args.syn_mode == "all" else [args.syn_mode]):
            r = run_synthetic(T=args.syn_T, mode=mode)
            key = "H1热度" if mode == "momentum" else "H2遗漏"
            print(f"[正对照·{mode}] T={r['T']} α={r['alpha']} 评估{r['n_eval']}期")
            for k, label in [("composite", "综合分Top5"), ("single", f"{key}单维Top5")]:
                s = r[k]
                ok = "✓检出" if s["ci"][0] > 1.25 else "✗未检出"
                print(f"    {label:14s} 命中{s['mean']:.3f} lift={s['lift']:.3f} "
                      f"CI[{s['ci'][0]:.2f},{s['ci'][1]:.2f}] p={s['p']:.1e} {ok}")
        return 0
    if args.strict_eval:
        try:
            strict = strict_walkforward_eval(ds0, args.train_ratio, args.valid_ratio, args.test_ratio,
                                            max_per_split=args.strict_eval_max)
            print(f"[严格评估] train/valid/test 分段：每段最多 {args.strict_eval_max} 期，非重叠且不使用伪点位")
            for label in ("train", "valid", "test"):
                s = strict[label]
                print(f"  {label:5s} n={s['n']} mean={s['mean']:.3f} lift={s['lift']:.3f} "
                      f"ci=[{s['ci'][0]:.3f},{s['ci'][1]:.3f}] p={s['p']:.3f}")
            return 0
        except Exception as e:
            print(f"[严格评估] 无法完成：{e}")
            return 2

    if args.calib_only or args.explore:
        out = {}
        result_path = resolve_existing(backtest_result_path(), "kl8_backtest_result.json")
        if result_path:
            try:
                with open(result_path, encoding="utf-8") as f:
                    out = json.load(f)
            except Exception:
                out = {}
        if args.calib_only:
            out["calibration"] = calibration(ds0)
            _print_calib(out["calibration"])
        stash = resolve_existing(backtest_stash_path(), "kl8_H_stash.npz")
        if args.explore and stash:
            out["weight_explore"] = explore(stash)
            out["blend_eval"] = blend_eval(stash)
            e = out["weight_explore"]["_选优"]
            print(f"[调权] 前半段最优={e['name']} 样本内lift={e['in_lift']:.3f}")
            print(f"[调权] 样本外 lift={e['oos_lift']:.3f} CI[{e['oos_ci'][0]:.3f},{e['oos_ci'][1]:.3f}] p={e['oos_p']:.3f}")
            b = out["weight_explore"]["_基准OOS"]
            print(f"[对照] 基准样本外 lift={b['oos_lift']:.3f} CI[{b['oos_ci'][0]:.3f},{b['oos_ci'][1]:.3f}]")
            if "blend_eval" in out:
                print("[双榜] 真实数据Top5:")
                for k, s in out["blend_eval"].items():
                    if k == "_选榜OOS":
                        continue
                    print(f"    {k} lift={s['lift']:.3f} CI[{s['ci'][0]:.2f},{s['ci'][1]:.2f}] p={s['p']:.2f}")
                v = out["blend_eval"]["_选榜OOS"]
                print(f"    选榜OOS(前半选[{v['chosen']}]): lift={v['lift']:.3f} CI[{v['ci'][0]:.2f},{v['ci'][1]:.2f}] p={v['p']:.2f}")
        out_path = backtest_result_path()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"[已写入] {out_path}")
        return 0

    ds = load_dataset()
    T = ds.T
    hi = T - 1
    lo = max(args.start, hi - args.periods)
    print(f"[回测] 预测期区间 idx {lo}..{hi-1}（共 {hi-lo} 期）")
    t0 = time.time()
    recs = []
    rec_idx = []          # 与 recs 逐行对齐的历史索引，供消融取回真实开奖期
    H_stash = []
    hits_mat = []
    n_pseudo = 0
    for t in range(lo, hi):
        period = ds.draws[t].period
        if period not in ds.point_sets:
            continue
        tds = truncated(ds, period)
        feat, ctx = run_pipeline(tds, point_period=period)
        if period in ctx["ds"].pseudo_point_periods:
            n_pseudo += 1
        comp = ctx["composite"]
        H = ctx["H"]
        p = picks(comp, H, feat, ctx)
        actual = ds.draws[t].numbers
        rec = {k: len(set(v) & actual) for k, v in p.items()}
        recs.append(rec)
        rec_idx.append(t)
        H_stash.append({hn: H[hn].copy() for hn, _, _ in H_DEF})
        hits_mat.append([1 if (n + 1) in actual else 0 for n in range(80)])
        if (t - lo) % 50 == 0:
            el = time.time() - t0
            print(f"  ...{t-lo+1}/{hi-lo} 用时{el:.0f}s 剩余约{el/(t-lo+1)*((hi-lo)-(t-lo+1)):.0f}s", flush=True)
    el = time.time() - t0
    print(f"[回测] 完成 {len(recs)} 期，用时 {el:.0f}s，其中伪点位 {n_pseudo} 期", flush=True)
    stash_path = backtest_stash_path()
    try:
        np.savez_compressed(stash_path,
                            H=np.stack([[H_stash[j][hn] for hn, _, _ in H_DEF]
                                        for j in range(len(H_stash))]).astype(np.float32),
                            hit=np.array(hits_mat, dtype=np.uint8),
                            lo=lo, idx=np.array(rec_idx, dtype=np.int64))
        print(f"[已写入] {stash_path}（供离线调权/样本外验证）", flush=True)
    except Exception as e:
        print("[警告] H_stash 保存失败:", e)

    out = {"n_periods": len(recs), "start_idx": lo, "end_idx": hi - 1,
           "pseudo_periods": n_pseudo, "elapsed_s": round(el, 1), "strategies": {}}
    for key, k in [("top5", 5), ("top20", 20), ("five", 5), ("seven", 7),
                   ("cluster5", 5)]:
        hits = [r[key] for r in recs]
        out["strategies"][key] = summarize(key, hits, k)
        s = out["strategies"][key]
        print(f"  {key:9s} 命中{s['mean']:.3f} 基线{s['base']:.2f} lift={s['lift']:.3f} "
              f"CI[{s['ci'][0]:.2f},{s['ci'][1]:.2f}] p={s['p']:.3f}")

    # 消融：置零单个H维后按权重重算Top5（复用已存H，零额外pipeline开销）
    # 必须用与 H_stash 逐行对齐的真实索引取回开奖期；早期版本用 lo+j 反推，
    # 一旦区间内有期因缺点位被跳过就会整体错位，把预测对到错误的开奖期上。
    out["ablation"] = {}
    base_hits = [r["top5"] for r in recs]
    ab_rows = max(1, min(args.ablate, len(recs)))
    ab_slice = slice(len(recs) - ab_rows, len(recs))
    ab_idx = rec_idx[ab_slice]
    base_mean = float(np.mean(base_hits[ab_slice])) if base_hits else 0.0
    for hn, _, _ in H_DEF:
        hits = []
        for j, t in zip(range(len(recs) - ab_rows, len(recs)), ab_idx):
            comp = np.zeros(80)
            for hn2, _, w in H_DEF:
                comp += (0.0 if hn2 == hn else w) * H_stash[j][hn2]
            top5 = set(np.argsort(-comp)[:5] + 1)
            hits.append(len(top5 & ds.draws[t].numbers))
        out["ablation"][hn] = summarize(f"drop_{hn}", hits, 5)
        s = out["ablation"][hn]
        print(f"  消融去{hn} (近{len(hits)}期) 命中{s['mean']:.3f} lift={s['lift']:.3f} "
              f"CI[{s['ci'][0]:.2f},{s['ci'][1]:.2f}]  (基准{base_mean:.3f})")

    t5 = out["strategies"]["top5"]
    out.update({"top5_mean": t5["mean"], "lift_top5": t5["lift"], "ci_top5": t5["ci"],
                "p_top5": t5["p"]})
    out_path = backtest_result_path()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[已写入] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())