# -*- coding: utf-8 -*-
"""条件命中率全扫描（只读诊断）

目标：在全历史开奖上扫描常见分层/信号的条件命中率，相对基线 0.25，
用「按期聚类的比率估计 95% CI」+ BH-FDR 校正，筛「CI下界>0.25 且 q<0.05」的候选。
对入选候选再做时间对半分稳定复检；空集则结论为无可用 edge。

统计口径：快乐8每期恰好开出 20 个号（无放回超几何抽样），同一期内各号码格子负相关，
因此不能把每个 (期, 号) 格子当独立伯努利试验——那会误估不确定度；
本脚本一律以「期」为聚类单元计算标准误（kl8_core.clustered_rate_ci，
原 Wilson 独立口径已按审计结论替换）。

用法:
  python kl8_edge_scan.py
  python kl8_edge_scan.py --history kl8_history_final.txt --points daily_points.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from math import erfc
from pathlib import Path

import numpy as np

from kl8_core import (
    CONFIG,
    N_NUM,
    TimeSeries,
    ensure_output_dirs,
    load_dataset,
    ring_next,
    ring_prev,
)

BASE = 0.25


def _ensure_utf8():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg q-values（与 p 等长）。"""
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        # rank from largest p: i = m, m-1, ...
        i = m - rank + 1
        val = pvals[idx] * m / i
        prev = min(prev, val)
        q[idx] = min(prev, 1.0)
    # 单调化：从最大 p 向最小 p 累加取前缀最小值
    running = 1.0
    for idx in reversed(order):
        running = min(running, q[idx])
        q[idx] = running
    return q.tolist()


def _ratio_stats(kt, nt, p0: float = BASE):
    """按期聚类的比率估计（委托 kl8_core.clustered_rate_ci，全项目单一实现）。

    @return (rate, ci_lo, ci_hi, se, naive_se, n_periods, k, n)
    """
    from kl8_core import clustered_rate_ci
    kt = np.asarray(kt, dtype=float)
    nt = np.asarray(nt, dtype=float)
    r = clustered_rate_ci(kt, nt, p0=p0)
    n = r["n"]
    naive = float(np.sqrt(max(r["rate"] * (1 - r["rate"]), 1e-12) / n)) if n else 1.0
    return (r["rate"], r["lo"], r["hi"], r["se"], naive,
            r["n_periods"], r["k"], r["n"])


def record_series(name: str, kt, nt, family: str, p0: float = BASE) -> dict:
    """按期聚类的比率估计显著性检验（组内格子不独立的正确口径）。

    审计修正：原 record() 用 Wilson/binomial 把每个 (期, 号) 格子当独立试验，
    CI 与 p 的统计方法都错了；改为聚类标准误。逐期序列留存于 "_series"，
    供半分稳定复检直接切分复用（不再按名称重算，杜绝原先「一半检验无法半分
    而被 else: continue 静默跳过」的漏检）。
    """
    kt = np.asarray(kt, dtype=float)
    nt = np.asarray(nt, dtype=float)
    keep = nt > 0
    kt, nt = kt[keep], nt[keep]
    T_ = int(len(kt))
    k = float(kt.sum())
    n = float(nt.sum())
    rate = k / n if n else 0.0
    if T_ > 1:
        e = kt - rate * nt
        var = float((e ** 2).sum() / ((T_ - 1) * (nt.mean() ** 2) * T_))
        se = float(np.sqrt(max(var, 0.0)))
    else:
        se = float(np.sqrt(max(rate * (1 - rate), 1e-12) / n)) if n else 1.0
    lo, hi = rate - 1.96 * se, rate + 1.96 * se
    z = (rate - p0) / se if se > 0 else 0.0
    p = float(erfc(abs(z) / np.sqrt(2.0)))
    return {
        "name": name,
        "family": family,
        "k": int(k),
        "n": int(n),
        "n_periods": T_,
        "rate": float(rate),
        "ci": [float(lo), float(hi)],
        "edge_lo": float(lo - p0),
        "se": se,
        "p": float(p),
        "pass_ci": bool(lo > p0 and n >= 200 and T_ >= 30),
        "_series": (kt, nt),
    }


def detect_runs(nums_sorted: list[int]) -> list[list[int]]:
    if not nums_sorted:
        return []
    runs, cur = [], [nums_sorted[0]]
    for x in nums_sorted[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)
    return runs


def scan_all(ds) -> dict:
    ts = TimeSeries(ds)
    H, gap = ts.H, ts.gap
    T = ds.T
    rows: list[dict] = []

    # ---------- 1) 连失分桶 ----------
    gap_defs = [
        ("gap=0", lambda g: g == 0),
        ("gap=1", lambda g: g == 1),
        ("gap=2", lambda g: g == 2),
        ("gap=3", lambda g: g == 3),
        ("gap=4", lambda g: g == 4),
        ("gap=5-6", lambda g: (g >= 5) & (g <= 6)),
        ("gap=7-9", lambda g: (g >= 7) & (g <= 9)),
        ("gap=10-14", lambda g: (g >= 10) & (g <= 14)),
        ("gap>=15", lambda g: g >= 15),
    ]
    for label, pred in gap_defs:
        kt, nt = [], []
        for t in range(1, T):
            m = pred(gap[t - 1])
            nt.append(int(m.sum()))
            kt.append(int(H[t, m].sum()))
        rows.append(record_series(label, kt, nt, "连失"))

    # ---------- 2) 近窗热度分桶（用 t-1 截止的滚动命中） ----------
    cum = np.cumsum(H, axis=0)
    for W, tag in ((3, "近3"), (10, "近10"), (30, "近30")):
        for lo_c, hi_c, lab in ((0, 0, "0次"), (1, 1, "1次"), (2, 2, "2次"),
                                (3, 4, "3-4次"), (5, 99, "5+次")):
            kt, nt = [], []
            for t in range(W, T):
                # hits in [t-W, t) = cum[t-1] - cum[t-W-1]
                prev = cum[t - 1] - (cum[t - W - 1] if t - W - 1 >= 0 else 0)
                m = (prev >= lo_c) & (prev <= hi_c)
                nt.append(int(m.sum()))
                kt.append(int(H[t, m].sum()))
            # 不可达空桶：W 期窗口内命中数不可能 >W（如「近3命中=5+次」恒为空），
            # 这类检验既无信息量又白白占用 BH 的次数 m，直接跳过不入表。
            if not any(x > 0 for x in nt):
                continue
            rows.append(record_series(f"{tag}命中={lab}", kt, nt, "热度"))

    # ---------- 3) 上期是否重号 ----------
    kt = [int(H[t, H[t - 1] > 0].sum()) for t in range(1, T)]
    nt = [20] * len(kt)
    rows.append(record_series("上期已开(重号候选)", kt, nt, "重号"))
    kt = [int(H[t, H[t - 1] == 0].sum()) for t in range(1, T)]
    nt = [60] * len(kt)
    rows.append(record_series("上期未开", kt, nt, "重号"))

    # ---------- 4) 静态区位（号码固有属性，应≈0.25，作负对照） ----------
    nums = np.arange(1, 81)
    static = [
        ("小号1-40", nums <= 40),
        ("大号41-80", nums > 40),
        ("奇数", nums % 2 == 1),
        ("偶数", nums % 2 == 0),
        ("黄金段1-30", nums <= 30),
        ("黄金段31-54", (nums >= 31) & (nums <= 54)),
        ("黄金段55-80", nums >= 55),
    ]
    for lab, mask80 in static:
        m = mask80.astype(bool)
        f = int(m.sum())
        kt = [int(H[t, m].sum()) for t in range(T)]
        nt = [f] * T
        rows.append(record_series(lab, kt, nt, "静态区位"))

    # ---------- 5) 上期连号成员 ----------
    kt, nt = [], []
    for t in range(1, T):
        opened = sorted(int(i + 1) for i in np.where(H[t - 1] > 0)[0])
        in_run = set()
        for r in detect_runs(opened):
            if len(r) >= 2:
                in_run.update(r)
        if not in_run:
            kt.append(0); nt.append(0)
            continue
        idx = np.array([x - 1 for x in in_run], dtype=int)
        nt.append(len(idx))
        kt.append(int(H[t, idx].sum()))
    rows.append(record_series("上期连号成员下期", kt, nt, "连号"))

    # ---------- 6) 期级语境：连号温度分位后追上期热号 ----------
    adj_pairs = []
    for t in range(T):
        opened = sorted(int(i + 1) for i in np.where(H[t] > 0)[0])
        adj_pairs.append(sum(max(0, len(r) - 1) for r in detect_runs(opened)))
    adj_pairs = np.array(adj_pairs, dtype=float)

    def period_bucket_hits(series, high: bool, family: str, name: str):
        """期级语境分桶 → 检验「追上期热号」口径。

        审计修正：原实现另有「·全体」分支（每期 n+=80、k+=H[t].sum()），
        而快乐8每期恒定开出 20 个号，故 rate ≡ 20/80 ≡ 0.25 为数学恒等式，
        既不可能成为信号，又白白占用 BH 多重比较次数 m（稀释真信号的 q 阈值），
        并在 Top20 表里塞入 4 行伪候选；该分支及其调用已按「移除无出口的实现」删除。
        """
        # 用 t-1 的期级特征分位，预测 t
        feat = series[:-1]
        recent = feat[max(0, len(feat) - 500):]
        thr = np.quantile(recent, 0.75 if high else 0.25)
        kt, nt = [], []
        for t in range(1, T):
            f = series[t - 1]
            ok = (f >= thr) if high else (f <= thr)
            if not ok:
                continue
            m = H[t - 1] > 0
            kt.append(int(H[t, m].sum()))
            nt.append(int(m.sum()))
        rows.append(record_series(name, kt, nt, family))

    period_bucket_hits(adj_pairs, True, "期级语境", "上期连号偏热后·追上期热号")
    period_bucket_hits(adj_pairs, False, "期级语境", "上期连号偏冷后·追上期热号")

    # ---------- 7) Markov P01 Top/Bottom ----------
    W = 200
    kt_hi, kt_lo = [], []
    for t in range(W, T):
        prev, cur = H[t - W:t - 1], H[t - W + 1:t]
        n0 = (prev == 0).sum(axis=0)
        p01 = (((prev == 0) & (cur == 1)).sum(axis=0) + 1) / (n0 + 2)
        kt_hi.append(int(H[t, np.argsort(-p01)[:30]].sum()))
        kt_lo.append(int(H[t, np.argsort(p01)[:30]].sum()))
    nt30 = [30] * len(kt_hi)
    rows.append(record_series("Markov P01 Top30", kt_hi, nt30, "Markov"))
    rows.append(record_series("Markov P01 Bottom30", kt_lo, nt30, "Markov"))

    # ---------- 8) 点位场（逐期序列，期序按开奖先后稳定） ----------
    scored = sorted(p for p in ds.point_sets if p in ds.draws_by_period)
    core_kt, core_nt, ext_kt, ext_nt = [], [], [], []
    for p in scored:
        ps = ds.point_sets[p]
        actual = ds.draws_by_period[p].numbers
        ext = set(ps)
        for m in ps:
            ext |= {ring_prev(m), ring_next(m)}
        core_kt.append(len(ps & actual)); core_nt.append(len(ps))
        ext_kt.append(len(ext & actual)); ext_nt.append(len(ext))
    rows.append(record_series("点位核心20", core_kt, core_nt, "点位场"))
    rows.append(record_series("点位扩展环", ext_kt, ext_nt, "点位场"))

    # FDR
    pvals = [r["p"] for r in rows]
    qvals = bh_fdr(pvals)
    for r, q in zip(rows, qvals):
        r["q"] = float(q)
        r["pass_fdr"] = bool(r["pass_ci"] and q < 0.05)

    # ---------- 半分稳定复检：直接切分已留存的逐期序列 ----------
    # 审计修正：原实现按 name 匹配后「重新算一遍」前半/后半，
    # 无法匹配的检验走 else: continue 被静默跳过（曾漏掉一半族的复检）。
    candidates = [r for r in rows if r["pass_ci"]]
    half = []
    for cand in candidates:
        kt, nt = cand["_series"]
        m = np.asarray(nt, dtype=float) > 0
        kt, nt = kt[m], nt[m]
        if len(kt) < 8:
            continue
        cut = len(kt) // 2
        seg = []
        for a, b in ((kt[:cut], nt[:cut]), (kt[cut:], nt[cut:])):
            r, lo, hi, _se, _nv, _np_, _k, _n = _ratio_stats(a, b)
            seg.append({"rate": r, "n": int(b.sum()), "n_periods": int(len(a)),
                        "ci": [lo, hi], "pass": bool(lo > BASE and int(b.sum()) >= 100)})
        half.append({
            "name": cand["name"],
            "first_half": seg[0],
            "second_half": seg[1],
            "stable": bool(seg[0]["ci"][0] > BASE and seg[1]["ci"][0] > BASE),
        })

    # 排序：按 CI 下界降序；_series 为内部对象，不出 JSON
    for r in rows:
        r.pop("_series", None)
    rows_sorted = sorted(rows, key=lambda r: (-r["ci"][0], -r["rate"]))
    return {
        "baseline": BASE,
        "T": T,
        "n_tests": len(rows),
        "rows": rows_sorted,
        "candidates_ci": [r for r in rows_sorted if r["pass_ci"]],
        "candidates_fdr": [r for r in rows_sorted if r["pass_fdr"]],
        "half_validation": half,
        "note": "pass_ci: 按期聚类的95%CI下界>0.25 且 n>=200 且期数>=30；"
                "pass_fdr: 另需 BH-q<0.05；"
                "stable: 时间前后半段 CI下界均>0.25",
    }


def format_report(result: dict) -> str:
    L = []
    sep = "=" * 78
    L.append(sep)
    L.append("  条件命中率全扫描诊断报告（相对基线 0.25）")
    L.append(f"  历史期数 T={result['T']} ｜ 检验条数={result['n_tests']}")
    L.append(sep)
    L.append("")
    L.append("[结论摘要]")
    L.append("-" * 78)
    ci_n = len(result["candidates_ci"])
    fdr_n = len(result["candidates_fdr"])
    stable = [h for h in result["half_validation"] if h.get("stable")]
    if fdr_n == 0 and ci_n == 0:
        L.append("  ★ 未发现任何「按期聚类95%CI下界 > 0.25」的分层信号。")
        L.append("  ★ 在现有特征族下，没有可跟进的稳定 edge；不建议调权/加模。")
    elif fdr_n == 0:
        L.append(f"  · CI下界>0.25 的候选 {ci_n} 个，但经 BH-FDR 后全部不显著（多重比较）。")
        L.append("  · 视为假阳性风险，不建议写入选号逻辑。")
    else:
        L.append(f"  · FDR 通过 {fdr_n} 个；半分稳定 {len(stable)} 个。")
        for r in result["candidates_fdr"][:10]:
            L.append(f"    - {r['name']}: rate={r['rate']:.4f} CI[{r['ci'][0]:.4f},{r['ci'][1]:.4f}] q={r['q']:.3f}")
    L.append("")
    L.append("[全表 Top20 · 按 CI下界排序]")
    L.append("-" * 78)
    L.append(f"  {'信号':28s} {'族':8s} {'rate':>7s} {'CI_lo':>7s} {'CI_hi':>7s} {'n':>9s} {'q':>7s} 旗标")
    for r in result["rows"][:20]:
        flag = []
        if r["pass_ci"]:
            flag.append("CI")
        if r["pass_fdr"]:
            flag.append("FDR")
        fs = ",".join(flag) if flag else "-"
        L.append(
            f"  {r['name'][:28]:28s} {r['family'][:8]:8s} {r['rate']:7.4f} "
            f"{r['ci'][0]:7.4f} {r['ci'][1]:7.4f} {r['n']:9d} {r['q']:7.3f}  {fs}"
        )
    L.append("")
    L.append("[按族：最强一条]")
    L.append("-" * 78)
    by_fam: dict[str, dict] = {}
    for r in result["rows"]:
        by_fam.setdefault(r["family"], r)  # rows already sorted by ci_lo desc
    for fam, r in by_fam.items():
        L.append(
            f"  {fam:8s} → {r['name']}: {r['rate']:.4f} "
            f"CI[{r['ci'][0]:.4f},{r['ci'][1]:.4f}] n={r['n']}"
        )
    if result["half_validation"]:
        L.append("")
        L.append("[半分验证 · 仅 CI 候选]")
        L.append("-" * 78)
        for h in result["half_validation"]:
            a, b = h["first_half"], h["second_half"]
            L.append(
                f"  {h['name']}: 前半 {a['rate']:.4f} CI[{a['ci'][0]:.4f},{a['ci'][1]:.4f}] | "
                f"后半 {b['rate']:.4f} CI[{b['ci'][0]:.4f},{b['ci'][1]:.4f}] | "
                f"{'稳定' if h['stable'] else '不稳定/未过线'}"
            )
    L.append("")
    L.append("[行动建议]")
    L.append("-" * 78)
    if fdr_n == 0:
        L.append("  1. 停止以「提升命中率」为目标的权重/特征迭代。")
        L.append("  2. 维持现有模型作结构化描述与复盘工具即可。")
        L.append("  3. 若继续研究，优先审计点位来源质量，而非再加数学模块。")
    else:
        L.append("  1. 仅将 FDR+半分双过的信号做成极简规则，独立样本外再验一次。")
        L.append("  2. 未双过的信号一律不进选号。")
    L.append("")
    L.append(result["note"])
    L.append(sep)
    return "\n".join(L)


def main(argv=None) -> int:
    _ensure_utf8()
    ap = argparse.ArgumentParser(description="条件命中率全扫描（只读）")
    ap.add_argument("--points", default=CONFIG.get("points_file", "daily_points.txt"))
    ap.add_argument("--history", default=CONFIG.get("history_file", "kl8_history_final.txt"))
    args = ap.parse_args(argv)

    print("=" * 60)
    print("  条件命中率全扫描 · 开始")
    print("=" * 60)
    ds = load_dataset(args.points, args.history)
    print(f"  历史 {ds.T} 期，点位已开奖期 "
          f"{sum(1 for p in ds.point_sets if p in ds.draws_by_period)} 个")
    result = scan_all(ds)
    text = format_report(result)
    print(text)

    ensure_output_dirs()
    out_dir = Path(CONFIG["output_dir"]) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "edge_scan_latest.txt"
    json_path = out_dir / "edge_scan_latest.json"
    txt_path.write_text(text + "\n", encoding="utf-8")
    # JSON: 去掉过大无关
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[已写入] {txt_path}")
    print(f"[已写入] {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
