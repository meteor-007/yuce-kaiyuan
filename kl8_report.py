# -*- coding: utf-8 -*-
"""H1-H7 评分合成 + 六节报告生成 + CLI 入口"""
import sys
import datetime
import numpy as np

# 确保在 Windows 控制台或管道重定向下输出 UTF-8，防止中文及特殊字符乱码
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from kl8_core import (CONFIG, N_NUM, Feat80, TimeSeries, load_dataset, minmax,
                      next_period_after, next_target_date, pad, zscore,
                      truncated, report_paths)
import kl8_methods as M

H_DEF = [
    ("H1热度", ["fib", "markov", "vol"], 0.20),
    ("H2遗漏", ["gap_score"], 0.16),
    ("H3集群", ["cluster", "cp", "run_evidence"], 0.20),
    ("H4周期", ["complex"], 0.14),
    ("H5波动", ["vol", "svd"], 0.10),
    ("H6定位", ["field_center", "zone", "phi_seg", "resonance"], 0.10),
    ("H7对抗", ["contra", "entropy"], 0.10),
]
SECTOR_CHARS = "▁▂▃▄▅▆▇█"

# 顺势榜（动量类维度）与对冲榜（补冷类维度）：避免单一综合分把相反方向相互抵消
TREND_DIMS = {"H1热度": 0.28, "H3集群": 0.28, "H5波动": 0.16, "H6定位": 0.20, "H4周期": 0.08}
HEDGE_DIMS = {"H2遗漏": 0.55, "H7对抗": 0.35, "H4周期": 0.10}


def agg(feat, cols):
    out = np.zeros(N_NUM)
    wsum = np.zeros(N_NUM)
    for c in cols:
        if c not in feat.cols:
            continue
        raw = feat.cols[c]
        v = feat.norm[c].copy()
        nanm = np.isnan(raw)
        v[nanm] = 0.5
        w = np.where(nanm, 0.5, 1.0)
        out += v * w
        wsum += w
    return out / np.maximum(wsum, 1e-9)


def compose_scores(feat):
    Hs = {}
    for name, cols, _ in H_DEF:
        Hs[name] = agg(feat, cols)
    comp = np.zeros(N_NUM)
    for name, _, w in H_DEF:
        comp += w * Hs[name]
    return Hs, comp


def _repeat_stats(ds):
    """近10/近50 相邻期 Jaccard（重号节奏，仅描述）。"""
    def jacc(a, b):
        return len(a & b) / 20.0
    draws = ds.draws
    if len(draws) < 3:
        return {"j10": 0.0, "j50": 0.0, "j_state": "样本不足"}
    pairs = [jacc(draws[i].numbers, draws[i - 1].numbers) for i in range(1, len(draws))]
    j10 = float(np.mean(pairs[-10:]))
    j50 = float(np.mean(pairs[-50:])) if len(pairs) >= 50 else float(np.mean(pairs))
    if j10 > j50 + 0.03:
        state = "重号偏热"
    elif j10 < j50 - 0.03:
        state = "重号偏冷"
    else:
        state = "正常"
    return {"j10": j10, "j50": j50, "j_state": state}


def build_context_cols(ds, ts, feat, ctx):
    """补充派生列：gap_score、run_evidence、field_center、zone、phi_seg。"""
    T = ds.T
    feat.add("gap_score", np.log1p(np.minimum(ts.gap[T - 1], CONFIG["gap_cap"])) / np.log1p(CONFIG["gap_cap"]))
    ri = ctx["triple_info"]
    feat.add("run_evidence", ri["run2"] + 2 * ri["run3"])
    core = set(ri["core"])
    ext = set(e["num"] for e in ctx["extended"])
    fc = np.array([1.0 if n in core else (0.6 if n in ext else 0.2) for n in range(1, 81)])
    feat.add("field_center", fc)
    _, _, dev = M.sector_energy(ds, ts)
    devz = zscore(dev)
    feat.add("zone", np.array([devz[(n - 1) // 5] for n in range(1, 81)]))
    bias = ctx["fib_info"]["phi_bias_seg"]
    feat.add("phi_seg", np.array([1.0 if (0 if n < 31 else (1 if n < 55 else 2)) == bias else 0.0
                                  for n in range(1, 81)]))
    res_vec = np.zeros(80)
    for t in ctx.get("triples", []):
        for m in t["members"]:
            res_vec[m - 1] = max(res_vec[m - 1], t.get("res", 1.0))
    feat.add("resonance", minmax(res_vec))


def run_pipeline(ds, point_period=None, cfg=CONFIG):
    ts = TimeSeries(ds)
    feat = Feat80()
    ctx = {"feat": feat}
    if point_period is not None and point_period in ds.point_sets:
        ds = _with_point_period(ds, point_period)
    M.m_cluster(ds, ts, ctx)
    feat.add("cluster", ctx["triple_info"]["cluster_score"])
    for fn in (M.m_markov, M.m_fib, M.m_linalg, M.m_complex, M.m_graph,
               M.m_volatility, M.m_entropy):
        name, vec, info = fn(ds, ts, ctx)
        feat.add(name, vec)
        ctx[name + "_info"] = info
    build_context_cols(ds, ts, feat, ctx)
    name, vec, _ = M.m_calc_zone(ds, ts, ctx)
    feat.add(name, vec)
    # calc_zones / calc_zone_info / calc_zone_picks 已由 m_calc_zone 写入 ctx
    ctx["crowding"] = ctx["graph_info"]["crowding"]
    Hs_pre, _ = compose_scores(feat)
    ctx["composite_h16"] = sum(w * Hs_pre[n] for n, _, w in H_DEF[:6])
    name, vec, info = M.m_adversarial(ds, ts, ctx)
    feat.add(name, vec)
    ctx["adv_info"] = info
    Hs, comp = compose_scores(feat)
    ctx["H"] = Hs
    ctx["composite"] = comp
    ctx["comp_trend"] = sum(w * Hs[n] for n, w in TREND_DIMS.items())
    ctx["comp_hedge"] = sum(w * Hs[n] for n, w in HEDGE_DIMS.items())
    ctx["repeat"] = _repeat_stats(ds)
    ctx["ts"] = ts
    ctx["ds"] = ds
    return feat, ctx


def _with_point_period(ds, period):
    """回测用：只在真实存在的点位期上切换分析期；禁止伪造未来点位。"""
    from kl8_core import Dataset
    nd = Dataset(draws=ds.draws, draws_by_period=ds.draws_by_period,
                 point_sets=dict(ds.point_sets), point_dates=dict(ds.point_dates),
                 target_period=period, warnings=ds.warnings)
    nd.pseudo_point_periods = set(getattr(ds, "pseudo_point_periods", set()))
    if period not in nd.point_sets:
        nd.point_sets.pop(period, None)
        nd.point_dates.pop(period, None)
        nd.pseudo_point_periods.discard(period)
    return nd


def top_idx(vec, k, exclude=()):
    order = np.argsort(-vec)
    out = []
    for i in order:
        if (i + 1) in exclude:
            continue
        out.append(i + 1)
        if len(out) >= k:
            break
    return out


def seconds(comp):
    return (comp * 100)


def _group_hit_lines(groups, actual):
    """主攻三元组对照开奖：≥2 码算命中组，3 码标全中。"""
    lines = []
    n_hit = 0
    for i, t in enumerate(groups, 1):
        mem = list(t.get("members", []))
        h = sorted(set(mem) & actual)
        mem_s = "-".join("%02d" % m for m in mem)
        if len(h) >= 2:
            n_hit += 1
            tag = " 【3连全中】" if len(h) == 3 else ""
            lines.append(f"    {i}. {mem_s} → 命中{len(h)}码 "
                         + " ".join("%02d" % x for x in h) + tag)
        else:
            lines.append(f"    {i}. {mem_s} → 未中"
                         + (f"（仅中 {' '.join('%02d' % x for x in h)}）" if h else ""))
    return n_hit, lines


def compute_picks(ctx):
    """按报告第6节同一契约：主攻=当日场分跟进的三元组（组数随日变化）。"""
    ti = ctx["triple_info"]
    lam = ctx["adv_info"]["lambda"]
    if lam >= 1.05:
        stance = "顺势"
    elif lam <= 0.95:
        stance = "对冲"
    else:
        stance = "中性"
    groups = M.select_main_groups(ti["triples"])
    return {
        "stance": "三元集群",
        "groups": groups,
        "trend5": top_idx(ctx["comp_trend"], 5),
        "hedge5": top_idx(ctx["comp_hedge"], 5),
        "rank_name": "三元集群主攻组",
        "lambda_stance": stance,
    }


def copyable_triples(ctx):
    """主攻三元组纯净可复制行（一组一行，无序号/表头）。"""
    groups = compute_picks(ctx).get("groups") or []
    return ["-".join("%02d" % m for m in g["members"]) for g in groups]


def build_target_settlement(actual_draw, ctx, point_set=None):
    """目标期已开奖时：用当期模型推荐对照实开号码做结算。"""
    if actual_draw is None:
        return []
    picks = compute_picks(ctx)
    act = actual_draw.numbers
    actual_sorted = sorted(act)
    lines = []
    lines.append(f"  【本期已开奖结算 (第{actual_draw.period}期 · {actual_draw.date})】:")
    lines.append("    · 真实开奖(20): " + " ".join("%02d" % n for n in actual_sorted))
    if point_set:
        hit_core = sorted(set(point_set) & act)
        lines.append(f"    · 核心点位命中: {len(hit_core)}/20 ({len(hit_core)/20*100:.1f}%) → "
                     + (" ".join("%02d" % n for n in hit_core) if hit_core else "未中"))
    groups = picks.get("groups") or []
    n_hit, g_lines = _group_hit_lines(groups, act)
    lines.append(f"    · 主攻三元组（{len(groups)} 组，≥2码算命中）: "
                 f"{n_hit}/{len(groups)} 组命中")
    lines.extend(g_lines)
    lines.append(f"    · 类型【{picks['stance']}】｜只看三元组，不以单号五码结算")
    return lines


def build_previous_period_review(ds, ctx):
    """构建上期全景详细复盘与模型预测回溯核算。
    <p>
    【业务背景与架构定位】:
    全面展示前一期的开奖全景、形态特征（连号、重号、大小、奇偶、和值）、
    核心点位与扩展集合命中率、三元组高爆发命中清单、
    以及前一期模型推荐（主攻三元组）实战命中表现。
    
    @param ds  Dataset 实例
    @param ctx 上下文容器
    @return list[str] 格式化的复盘文本行列表
    """
    lines = []
    T = ds.T
    if T < 1:
        return ["  无历史开奖记录，跳过复盘"]
    
    prev = ds.draws[T - 1]
    pp = ds.draws[T - 2] if T >= 2 else None
    actual_sorted = sorted(prev.numbers)
    repeats = sorted(prev.numbers & pp.numbers) if pp else []
    runs = [r for r in M.detect_runs(actual_sorted) if len(r) >= 2]
    n_small = sum(1 for x in actual_sorted if x <= 40)
    n_odd = sum(1 for x in actual_sorted if x % 2 == 1)
    
    lines.append(f"  【上期开奖全景 (第{prev.period}期 · {prev.date})】:")
    lines.append("    · 真实开奖(20): " + " ".join("%02d" % n for n in actual_sorted))
    lines.append(f"    · 形态统计: 大小比(1-40:41-80) {n_small}:{20-n_small} ｜ 奇偶比 {n_odd}:{20-n_odd} ｜ "
                 f"和值 {sum(actual_sorted)} ｜ 跨度 {actual_sorted[-1] - actual_sorted[0]}")
    lines.append(f"    · 重号追踪: 与前期({pp.period if pp else '无'})重叠 {len(repeats)} 个 → "
                 + (" ".join("%02d" % n for n in repeats) if repeats else "无"))
    runs_str = "、".join("-".join("%02d" % n for n in r) for r in runs) if runs else "无"
    lines.append(f"    · 连号集群: 实测开出 {len(runs)} 组连号 → {runs_str}")

    # 点位场实测
    if prev.period in ds.point_sets:
        pset = ds.point_sets[prev.period]
        hit_core = sorted(pset & prev.numbers)
        miss_core = sorted(pset - prev.numbers)
        ext_set = set(pset)
        for m in pset:
            ext_set.add(((m - 2) % N_NUM) + 1)
            ext_set.add((m % N_NUM) + 1)
        hit_ext = sorted(ext_set & prev.numbers)
        
        lines.append("")
        lines.append(f"  【上期点位场实测核算】:")
        lines.append(f"    · 核心点位(20) 命中: {len(hit_core)}/20 ({len(hit_core)/20*100:.1f}%)")
        lines.append(f"        命中号码: " + (" ".join("%02d" % n for n in hit_core) if hit_core else "无"))
        lines.append(f"        未中点位: " + " ".join("%02d" % n for n in miss_core))
        lines.append(f"    · 扩展集合({len(ext_set)}) 命中: {len(hit_ext)}/{len(ext_set)} ({len(hit_ext)/len(ext_set)*100:.1f}%) "
                     + f"→ 命中: " + " ".join("%02d" % n for n in hit_ext))

        # 回溯上期的模型推荐战绩
        try:
            tds = truncated(ds, prev.period)
            _, ctx_prev = run_pipeline(tds, point_period=prev.period)
            ti_p = ctx_prev["triple_info"]

            # 三元组爆发
            tri_hits = []
            for t in ti_p["triples"]:
                h_m = sorted(set(t["members"]) & prev.numbers)
                if len(h_m) >= 2:
                    tri_hits.append((t, h_m))
            if tri_hits:
                lines.append("    · 三元组集群命中 (≥2连/高共振场):")
                for t, h_m in tri_hits[:4]:
                    mem_s = "-".join("%02d" % x for x in t["members"])
                    lines.append(f"        - {mem_s} (中心p={t['center']:02d}): 命中 {len(h_m)} 码 → "
                                 + " ".join("%02d" % x for x in h_m)
                                 + (" 【3连全中!】" if len(h_m) == 3 else ""))

            # 与 compute_picks 同一契约：只复盘主攻三元组
            picks_p = compute_picks(ctx_prev)
            groups_p = picks_p.get("groups") or []
            stance_p = picks_p["stance"]
            n_hit, g_lines = _group_hit_lines(groups_p, prev.numbers)

            lines.append("")
            lines.append(f"  【上期模型推荐实测战绩 (【{stance_p}】主攻{len(groups_p)}组)】:")
            lines.append(f"    · 主攻三元组命中: {n_hit}/{len(groups_p)} 组（≥2码算命中）")
            lines.extend(g_lines)

        except Exception as e:
            lines.append(f"    [注] 模型历史推演回溯提示: {e}")
    else:
        lines.append(f"  上期({prev.period})无点位记录，已展示开奖号码与形态全景")

    return lines


def _edge_scan_note():
    """读取条件扫描诊断产物，返回 (检验条数, 结论行列表)。无产物时给出重跑指引。"""
    import json
    from kl8_core import resolve_existing
    from pathlib import Path
    candidates = ["output/diagnostics/edge_scan_latest.json", "edge_scan_latest.json"]
    path = resolve_existing(*candidates)
    if not path:
        return None, ["  · 命中率结论: 条件扫描产物缺失，请先运行: python kl8_edge_scan.py"]
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        n = int(d.get("n_tests", 0))
        cands = d.get("candidates_fdr") or d.get("candidates_ci") or []
        if cands:
            lines = [f"  · 命中率结论: 全历史 {n} 项条件扫描中 {len(cands)} 项通过 FDR 复检"]
            for c in cands[:5]:
                lines.append(f"    - {c['name']}: rate={c['rate']:.4f} n={c['n']}")
        else:
            lines = [f"  · 命中率结论: 全历史 {n} 项条件扫描均未出现 CI下界>0.25 的信号。"]
        return n, lines
    except Exception as e:
        return None, [f"  · 命中率结论: 诊断产物读取失败（{type(e).__name__}: {e}），"
                      f"请重跑 python kl8_edge_scan.py"]

# ============================================================
# 近期命中率面板：按日落盘的滚动实测统计
# ============================================================

HIT_PANEL_DAYS = 40


def recent_periods(ds_orig, n_days):
    """返回最近 n_days 个「点位+开奖」齐备的真实期号。

    口径与交付一致：必须是当日录入了核心点位且已开奖的期——
    这些期才是系统真实产出了推荐清单的期，回测才不掺水。
    """
    if ds_orig is None:
        return []
    return [p for p in sorted(ds_orig.point_sets) if p in ds_orig.draws_by_period][-n_days:]


def hyper_triple_probs(k=3, draw_n=20, pool=N_NUM):
    """环形三元组 k 码在「80 选 20」下的精确超几何命中概率。

    审计修正：此前用独立二项近似 P(≥2)=3p²-2p³≈0.156/0.516 之类是错的口径混用，
    正确值必须按超几何逐点数计算（组内号码同进同出，非独立）。
    @return (P恰好全中, P至少2码)
    """
    from math import comb
    tot = comb(pool, draw_n)
    p_all = comb(pool - k, draw_n - k) / tot
    p_ge2 = sum(comb(k, j) * comb(pool - k, draw_n - j) for j in range(2, k + 1)) / tot
    return float(p_all), float(p_ge2)


def fast_replay(ds_orig, period):
    """轻量复算：只重建「交付清单」真正依赖的两个模块（三元集群 + 八区微积分）。

    审计结论：综合评分链路里的 markov/svd/complex/graph/entropy 等重算子对
    交付清单（主攻三元组、八区16码）没有任何出口，逐期复算命中率不必再付这份代价。
    @return dict 命中明细；历史不足时 None
    """
    from kl8_core import TimeSeries, ring_prev, ring_next
    from kl8_methods import build_triples, build_calc_zones, select_main_groups
    tds = truncated(ds_orig, period)
    if tds.T < 40:
        return None
    ts = TimeSeries(tds)
    triples, _ext, _cs, info = build_triples(tds, ts, {})
    actual = ds_orig.draws_by_period[period].numbers
    groups = select_main_groups(triples)
    gnums = set()
    for g in groups:
        gnums |= set(g["members"])
    core = set(info["core"])
    ext = set()
    for m in core:
        ext |= {m, ring_prev(m), ring_next(m)}
    zones, p16, _s, _i = build_calc_zones(tds, ts, {})
    return {
        "period": period, "date": ds_orig.draws_by_period[period].date,
        "core_hit": len(core & actual), "core_n": len(core),
        "ext_hit": len(ext & actual), "ext_n": len(ext),
        "g_n": len(groups),
        "g_hit2": sum(1 for g in groups if len(set(g["members"]) & actual) >= 2),
        "g_full3": sum(1 for g in groups if len(set(g["members"]) & actual) == 3),
        "g_dedup": len(gnums & actual), "g_dedup_n": len(gnums),
        "z16": len(set(p16) & actual), "z16_n": len(p16),
    }


def recent_hits(ds_orig, n_days=HIT_PANEL_DAYS):
    """walk-forward 滚动复算近期各交付口径命中率（严格只用当期之前数据）。

    @return (per_period_rows, aggregate_dict) 或 (None, None) 样本不足
    """
    periods = recent_periods(ds_orig, n_days)
    if len(periods) < 5:
        return None, None
    rows = []
    for p in periods:
        r = fast_replay(ds_orig, p)
        if r:
            rows.append(r)
    if len(rows) < 5:
        return None, None
    n = len(rows)

    def mean(key):
        return float(sum(r[key] for r in rows) / n)

    # 主口径：当日主攻组去重号命中（n 随日变化，按各自 n 折算成命中率）
    k_total = sum(r["g_dedup_n"] for r in rows)
    h_total = sum(r["g_dedup"] for r in rows)
    rate = h_total / k_total if k_total else 0.0
    # CI：组内号码同进同出（非独立），以「期」为聚类单元的闭式比率估计。
    # （已实测与按期 bootstrap 一致，宽度差 <0.002；闭式无随机种子依赖，结果可复现）
    from kl8_core import clustered_rate_ci
    ci = clustered_rate_ci([r["g_dedup"] for r in rows],
                           [r["g_dedup_n"] for r in rows])
    ext_k = sum(r["ext_hit"] for r in rows)
    ext_n = sum(r["ext_n"] for r in rows)
    # 三元组精确超几何基线（P恰好全中, P至少2码）
    p_all3, p_ge2 = hyper_triple_probs(3)
    agg = {
        "n": n,
        "first": rows[0]["period"], "last": rows[-1]["period"],
        "main_rate": rate, "main_ci": [ci["lo"], ci["hi"]],
        "main_n": k_total, "main_h": h_total,
        "main_per_period": float(h_total / n), "main_k_avg": float(k_total / n),
        "p_ge2": p_ge2, "p_all3": p_all3,
        "group_hit2": mean("g_hit2"), "group_n": mean("g_n"), "group_full3": mean("g_full3"),
        "core": mean("core_hit"), "core_n": mean("core_n"),
        "ext_rate": ext_k / max(ext_n, 1), "ext_n_avg": float(ext_n / n),
        "z16": mean("z16"), "z16_n": float(sum(r["z16_n"] for r in rows) / n),
        "rows": rows,
    }
    return rows, agg


def _hp_row(label, val, base, unit="", ci=None, dec=2):
    lift = (val / base) if base else 0.0
    mark = "↑" if lift > 1.02 else ("↓" if lift < 0.98 else "≈")
    cis = f" 95%CI[{ci[0]*100:.2f}%,{ci[1]*100:.2f}%]" if ci else ""
    fmt = f"{{val:.{dec}f}}"
    return (f"  {pad(label, 26)}{fmt.format(val=val):>9}{unit}  随机基线 {base:.{dec}f}  "
            f"lift={lift:5.3f}{mark}{cis}")


def hit_panel_judgement(main_rate, ci):
    """按主口径命中率与 95%CI 生成判读文案，避免硬编码与面板数值矛盾。"""
    lo, hi = ci
    if lo > 0.25:
        return (f"  判读: 本窗口主口径 {main_rate * 100:.2f}% 的 95%CI 下界 {lo * 100:.2f}% 高于 25% 基线；"
                f"但多窗口/多重比较下仍不构成可跟进 edge，须按模板第 6 节复核。")
    if hi < 0.25:
        return (f"  判读: 本窗口主口径 {main_rate * 100:.2f}% 的 95%CI 上界 {hi * 100:.2f}% 低于 25% 基线；"
                f"低于随机，同样不构成可跟进 edge。")
    return (f"  判读: 本窗口主口径 {main_rate * 100:.2f}% 的 95%CI "
            f"[{lo * 100:.2f}%, {hi * 100:.2f}%] 覆盖 25% 基线，差异不显著，不构成可跟进 edge。")


def build_hit_panel(ctx, ds_orig=None, n_days=HIT_PANEL_DAYS):
    """近期命中率面板（[8] 节）。结果按滚动天数缓存，避免同一报告内重复复算。"""
    key = f"_hit_panel_cache_{int(n_days)}"
    if key not in ctx:
        try:
            ctx[key] = recent_hits(ds_orig, n_days)
        except Exception as e:
            ctx[key] = (None, {"err": f"{type(e).__name__}: {e}"})
    rows, agg = ctx[key]
    line = "-" * 78
    L = []
    L.append(f"[8] 近期命中率面板（{n_days} 日滚动 · walk-forward 严格复算，无未来数据）")
    L.append(line)
    if not agg or "err" in agg:
        L.append(f"  样本不足或计算失败："
                 + (f"真实「点位+开奖」齐备期少于 5 个（当前 {len(recent_periods(ds_orig, n_days))}）"
                    if not agg else agg["err"]))
        return L
    L.append(f"  统计区间: 第{agg['first']}期 → 第{agg['last']}期（{agg['n']} 期）")
    L.append("")
    L.append("  【主口径 = 系统每日实际交付清单】")
    L.append(_hp_row("主攻组去重号·命中率", agg["main_rate"] * 100, 25.0, "%",
                     agg["main_ci"], dec=2))
    L.append(_hp_row("主攻组去重号·命中个数", agg["main_per_period"],
                     agg["main_k_avg"] * 0.25, "个/期"))
    L.append(_hp_row("主攻三元组·≥2码组数", agg["group_hit2"],
                     agg["group_n"] * agg["p_ge2"], "组/期"))
    L.append(_hp_row("主攻三元组·3连全中", agg["group_full3"],
                     agg["group_n"] * agg["p_all3"], "组/期"))
    L.append(f"    （三元组基线按精确超几何: 单组 P(≥2码)={agg['p_ge2']*100:.2f}%、"
             f"P(3连全中)={agg['p_all3']*100:.2f}%，非独立二项近似）")
    L.append("")
    L.append("  【对照口径】")
    L.append(_hp_row("核心点位20码命中", agg["core"], agg["core_n"] * 0.25, "个/期"))
    L.append(_hp_row("扩展环命中(折算20码)", agg["ext_rate"] * 20, 5.0, "个/期"))
    L.append(_hp_row("八区16码命中", agg["z16"], agg["z16_n"] * 0.25, "个/期"))
    L.append("")
    L.append("  逐期明细（近 12 期）:")
    tail = rows[-12:]
    for r in tail:
        L.append(f"    {r['period']} {r['date']}  点位{r['core_hit']:2d}/{r['core_n']}  "
                 f"主攻{r['g_hit2']}/{r['g_n']}组(全中{r['g_full3']})  "
                 f"去重{r['g_dedup']:2d}/{r['g_dedup_n']:2d}  16码{r['z16']:2d}/{r['z16_n']}")
    if len(rows) > len(tail):
        L.append(f"    …（更早 {len(rows)-len(tail)} 期已计入均值，完整明细见详细版附录 F）")
    L.append(hit_panel_judgement(agg["main_rate"], agg["main_ci"]))
    return L


def build_report(ds, feat, ctx, ds_orig=None, actual_target=None, hit_days=HIT_PANEL_DAYS):
    L = []
    H = ctx["H"]
    comp = ctx["composite"]
    ti = ctx["triple_info"]
    ts = ctx["ts"]
    T = ds.T
    target = ds.target_period
    pdate = ds.point_dates.get(target, "?")
    last = ds.draws[T - 1]
    top5 = top_idx(comp, CONFIG["top_n"])
    comp100 = seconds(comp)

    sep = "=" * 78
    line = "-" * 78

    mode = "复盘结算" if actual_target is not None else "待开预测"
    L.append(sep)
    L.append("  快乐8「点位+遗漏」深度分析报告")
    L.append(f"  期号: {target}   日期: {pdate}   模式: {mode}")
    L.append(f"  历史数据截至: {last.period} ({last.date})")
    L.append("  核心点位(20): " + " ".join("%02d" % n for n in ti["core"]))
    L.append(f"  扩展集合: {len(ctx['extended'])} 个号码（环形[前一位,点位,后一位]）")
    L.append(sep)

    # [0] 点位场概览与上期深度复盘 / 本期结算
    L.append("[0] 点位场概览与上期深度复盘")
    L.append(line)
    settle = build_target_settlement(
        actual_target, ctx, point_set=ds.point_sets.get(target) or set(ti["core"])
    )
    if settle:
        L.extend(settle)
        L.append("")
    L.extend(build_previous_period_review(ds, ctx))
    L.append("")
    L.append("  【本期点位场环境承接】:")
    L.append(f"    · 连号温度: 近30期平均相邻对 {ti['adj_pairs_mean']:.2f} "
             f"(理论期望 {ti.get('adj_exp', 4.75):.2f}，z={ti.get('cluster_z', 0.0):+.2f}) → "
             + ti.get("cluster_state", "连号状态未知"))
    L.append(f"    · 扩展集合覆盖度: 20核心点位 → {len(ctx['extended'])} 号（重叠即集群密度）")
    L.append("")

    # [1] N维评分仪表盘
    L.append("[1] N维评分仪表盘（Top 5）")
    L.append(line)
    head = pad("排名", 6) + pad("号码", 6) + "".join(pad(n, 9) for n, _, _ in H_DEF) + pad("综合", 8) + "归因"
    L.append("  " + head)
    for r, n in enumerate(top5, 1):
        attrib = sorted(H_DEF, key=lambda hd: -hd[2] * H[hd[0]][n - 1])[:2]
        at = "+".join(h[0][:2] for h in attrib)
        row = pad(str(r), 6) + pad("%02d" % n, 6)
        row += "".join(pad("%.1f" % (H[name][n - 1] * 100), 9) for name, _, _ in H_DEF)
        row += pad("%.1f" % comp100[n - 1], 8) + at
        L.append("  " + row)
    trend5 = top_idx(ctx["comp_trend"], 5)
    hedge5 = top_idx(ctx["comp_hedge"], 5)
    L.append("  顺势榜(热度+集群) Top5: " + " ".join("%02d" % n for n in trend5))
    L.append("  对冲榜(遗漏)     Top5: " + " ".join("%02d" % n for n in hedge5))
    L.append("  说明: 综合分=H1–H7 加权。达标线=滚动回测95%CI下界高于随机基线；未过线仅作对照")
    L.append("")

    # [2] Top5 逐号点评
    L.append("[2] 本期最值得关注的 Top 5（热度/遗漏/集群）")
    L.append(line)
    for n in top5:
        m = n - 1
        h10 = int(ts.hits_window(n, T, 10))
        h3 = int(ts.hits_window(n, T, 3))
        h30 = int(ts.hits_window(n, T, 30))
        gap = int(ts.gap[T - 1, m])
        tri = [t for t in ti["triples"] if n in t["members"]]
        tbest = max(tri, key=lambda t: t["score"]) if tri else None
        core = set(ti["core"])
        ext = set(e["num"] for e in ctx["extended"])
        if n in core:
            loc = "核心点位"
        elif n in ext:
            loc = "扩展邻位"
        else:
            loc = "场外"
        L.append(f"  ● {n:02d} 号（综合 {comp100[m]:.1f}）")
        L.append(f"      热度: 近10中{h10}、近3中{h3}、近30中{h30}；热度分 {H['H1热度'][m]*100:.1f}")
        L.append(f"      遗漏: 当前连失{gap}期（理论期望4期）；回补压力分 {H['H2遗漏'][m]*100:.1f}")
        if tbest:
            lab = "/".join(tbest["labels"]) if tbest["labels"] else "常规"
            L.append(f"      集群: 属三元组 {'-'.join('%02d'%x for x in tbest['members'])}"
                     f"(p={tbest['center']:02d}) 场分{tbest['score']:.2f}、真实连号佐证{tbest['run']:.2f} [{lab}]，"
                     f"被{len(tri)}个点位场覆盖")
        else:
            L.append("      集群: 不在任何核心点位场三元组内")
        L.append(f"      定位: {loc}")
    L.append("")

    L.append("[3] 宏观趋势（主导周期 / 场域偏向 / 熵值）")
    L.append(line)
    domp = ctx["complex_info"]["dominant_periods"]
    if domp:
        L.append("  主导周期: " + "、".join(f"{p}期(出现{c}次)" for p, c in domp))
    fi = ctx["fib_info"]
    seg_name = ["1-30", "31-54", "55-80"][fi["phi_bias_seg"]]
    L.append(f"  场域偏向(黄金分割段): {seg_name} 段实测占比偏{'+' if fi['phi_bias_dev']>0 else ''}"
             f"{fi['phi_bias_dev']*100:.1f}%")
    theta, conc = M.energy_centroid(ds, ts)
    L.append(f"  能量质心: 第 {theta:.0f} 号附近（集中度 {conc:.3f}）")
    ent = ctx["entropy_info"]["entropy_anomaly"]
    L.append(f"  熵值: z={ent:.2f}" + ("（未达异常阈值2）" if abs(ent) <= 2 else "（异常）"))
    vs = ctx["vol_info"]
    L.append(f"  重号节奏: 近期Jaccard {vs['j10']:.3f} vs 长期 {vs['j50']:.3f} → {vs['j_state']}")
    L.append("")

    # [4] 风险预警（固定结论，不再引用已失效的七维回测缓存）
    L.append("[4] 风险预警与应对")
    L.append(line)
    low = [n for n in top_idx(-comp, 20) if H["H2遗漏"][n - 1] > 0.8]
    if low:
        L.append("  · 深遗漏观察: 综合排名后20中 " + "、".join("%02d" % n for n in low)
                 + " 遗漏分偏高（描述用，扫描未证明回补优势）")
    else:
        L.append("  · 深遗漏观察: 后20名中无极端遗漏号")
    _n_edge, edge_lines = _edge_scan_note()
    L.extend(edge_lines)
    L.append("    详见 output/diagnostics/edge_scan_latest.txt（python kl8_edge_scan.py 可重跑）")
    L.append("  · 使用边界: 以下推荐是点位场的结构化对照，不是可盈利信号。")
    L.append("")

    # [5] 决策树
    L.append("[5] 《趋势简报》决策树")
    L.append(line)
    lam = ctx["adv_info"]["lambda"]
    stance = "顺势" if lam >= 1.05 else ("对冲" if lam <= 0.95 else "中性")
    L.append(f"  ① 态势判定: λ={lam:.3f} → 本期基调【{stance}】")
    follow = set(top_idx(comp, CONFIG["follow_top"])) & set(ctx["markov_info"]["p01_top30"])
    L.append(f"  ② 跟随∩Markov筛选: {len(follow)} 个 → "
             + " ".join("%02d" % n for n in sorted(follow, key=lambda x: -comp[x - 1])[:16]))
    L.append("  ③ 连号集群三元组（场分全表，供对照）:")
    for line_t in M.triples_table(ti["triples"], 8).split("\n"):
        L.append("     " + line_t)
    L.append("")

    # [6] 终极推荐：只看主攻三元组（与五码无关）
    L.append("[6] 终极推荐清单 · 主攻三元组")
    L.append(line)
    picks = compute_picks(ctx)
    groups = picks.get("groups") or []
    n_g = len(groups)
    L.append(f"  选号规则: 场分跟进截断（跳过低温弃子）；组数随日变化，今日 {n_g} 组")
    L.append(f"  主攻三元组（{n_g} 组）:")
    for i, t in enumerate(groups, 1):
        lab = "/".join(t["labels"]) if t["labels"] else "常规"
        L.append(f"    {i}. {'-'.join('%02d'%m for m in t['members'])} "
                 f"(中心{t['center']:02d}) 场分{t['score']:.2f} [{lab}]")
    L.append("  类型: 三元集群")
    L.append(f"  使用建议: 只看上面 {n_g} 组整组连号；其他维度仅对照，不改这组推荐")
    if actual_target is not None:
        n_hit, g_lines = _group_hit_lines(groups, actual_target.numbers)
        L.append(f"  ★ 本期结算: 主攻 {n_hit}/{n_g} 组命中（≥2码）")
        for gl in g_lines:
            L.append("  " + gl.strip())
    L.append("")

    # [7] 集群三元组·全息动力学与金字塔推荐架构
    L.append("[7] 集群三元组·全息动力学与金字塔推荐架构（T1核心 + T2空间正交防御）")
    L.append(line)
    L.append("  架构原理: 点位引力拓扑共振 ｜ 50期同出协方差亲和度 ｜ 空间正交防撞车矩阵")
    L.append("  金字塔分层推荐清单:")
    for g in groups:
        mem_s = "%02d-%02d-%02d" % g["members"]
        tier_s = g.get("tier", "推荐")
        labs_s = "/".join(g.get("labels", []))
        c = g["center"]
        L.append(f"    ★ [{tier_s}] {mem_s} (中心p={c:02d}) 综合场分={g['score']:.3f} | "
                 f"亲和={g.get('aff', 0.0):.2f} 共振={g.get('res', 1.0):.1f} 覆盖={g['cov']:.2f} 遗漏={g['gap_s']:.2f} [{labs_s}]")
    L.append("")
    ordered_nums = []
    for g in groups:
        for m in g["members"]:
            if m not in ordered_nums:
                ordered_nums.append(m)
    L.append(f"  主攻去重精选码 ({len(ordered_nums)}个): " + " ".join("%02d" % n for n in ordered_nums))
    L.append("  实战使用建议: 优先重注 [T1核心] 连号组合，搭配 [T2防守] 覆盖其它宏观区间，实现防空门对冲。")
    L.append("")
    L.append("  ★ 候选三元组 Top8（对照用，非主攻清单）:")
    for t in ti["triples"][:8]:
        L.append(f"    {'-'.join('%02d'%m for m in t['members'])}")
    L.append("")

    # [8] 近期命中率面板（滚动实测，供第6步决策）
    L.extend(build_hit_panel(ctx, ds_orig=ds_orig, n_days=hit_days))
    L.append("")

    # 附录
    L.append("[附录] 扩展点位集合逐号统计（近10命中/近3命中/连失/覆盖点位场数）")
    L.append(line)
    rows = []
    for e in ctx["extended"]:
        rows.append("%02d:10中%d/3中%d/连失%d/覆盖%d" % (e["num"], e["h10"], e["h3"], e["gap"], e["cov"]))
    for i in range(0, len(rows), 3):
        L.append("  " + "   ".join(pad(r, 22) for r in rows[i:i + 3]))
    L.append("")
    L.append(line)
    L.append("  免责声明: 快乐8每号单期中出概率恒为 20/80=0.25，以上模型为结构化描述工具，")
    L.append("  任何方法均无法改变随机本质，请理性参考、量力而行。")
    L.append("")
    L.append(line)
    L.append("  本期主攻三元组·纯净可复制清单（与 [6] 同契约）")
    L.append(line)
    for g in groups:
        L.append("-".join("%02d" % m for m in g["members"]))
    L.append("")
    L.append(sep)
    return "\n".join(L)


def build_detail(ctx, feat):
    """详细版附加内容：全三元组子分、Top20各维原始分、扇区能量、周期相位、名单类"""
    L = []
    H = ctx["H"]
    comp = ctx["composite"]
    ts = ctx["ts"]
    ds = ctx["ds"]
    T = ds.T
    ti = ctx["triple_info"]
    line = "-" * 78
    sep = "=" * 78

    L.append(sep)
    L.append("  详细版附录 A：连号集群三元组全量表（20个，含成员明细与四子分原始值）")
    L.append(sep)
    L.append(" 排名 三元组(p)  子分:覆盖/遗漏/爆发/佐证  加权分 标签")
    for i, t in enumerate(ti["triples"], 1):
        mem = t["members"]
        L.append(f"  {i:>3}  {'-'.join('%02d'%m for m in mem)} (p={t['center']:02d})  "
                 f"分={t['score']:.3f}  覆盖={t['cov']:.3f} 遗漏={t['gap_s']:.3f} "
                 f"爆发={t['burst']:.2f} 佐证={t['run']:.3f}  "
                 f"[{'/'.join(t['labels']) if t['labels'] else '常规'}]")
        L.append("       " + " | ".join(
            f"{m:02d}:10中{int(t['h10'][k])} 3中{int(t['h3'][k])} 连失{int(t['gap'][k])}"
            for k, m in enumerate(mem)))
    L.append("")
    L.append("  —— 扩展集合逐号明细（覆盖点位场数=重叠度）——")
    for e in ctx["extended"]:
        L.append(f"   {e['num']:02d}: 近10中{e['h10']} 近3中{e['h3']} 连失{e['gap']} 覆盖{e['cov']}")

    L.append("")
    L.append(sep)
    L.append("  详细版附录 B：Top20 号码三维与连失")
    L.append(sep)
    top20 = top_idx(comp, 20)
    L.append("  号  综合   H1热度 H2遗漏 H3集群 | 连失")
    for n in top20:
        m = n - 1
        row = (f"  {n:02d} {comp[m]*100:5.1f}  " +
               " ".join(f"{H[h][m]*100:6.1f}" for h, _, _ in H_DEF) +
               f"  | {int(ts.gap[T-1, m]):2d}")
        L.append(row)

    L.append("")
    L.append(sep)
    L.append("  详细版附录 C：扇区能量")
    L.append(sep)
    energy, obs, dev = M.sector_energy(ds, ts)
    bar = "".join(SECTOR_CHARS[min(7, max(0, int((d + 1) / 2 * 8)))] for d in dev)
    L.append(f"  16扇区能量偏差(每5号一区, 期望25%): {bar}   （▁低→█高）")
    for i in range(0, 16, 8):
        L.append("   " + "  ".join(f"{5*(i+k)+1:02d}-{5*(i+k)+5:02d}:{dev[i+k]*100:+.0f}%"
                                   for k in range(8) if i + k < 16))
    theta, conc = M.energy_centroid(ds, ts)
    L.append(f"  能量质心: 第{theta:.0f}号附近, 集中度={conc:.3f}")
    rp = ctx["repeat"]
    L.append(f"  重号节奏: J10={rp['j10']:.3f} J50={rp['j50']:.3f} → {rp['j_state']}")

    L.append("")
    L.append(sep)
    L.append("  详细版附录 D：三元组全量全息动力学明细（共振度、亲和度与动力学标签）")
    L.append(sep)
    ti = ctx.get("triple_info") or {}
    triples = ti.get("triples") or ctx.get("triples") or []
    for i, t in enumerate(triples, 1):
        mem_s = "%02d-%02d-%02d" % t["members"]
        labs_s = "/".join(t.get("labels", []))
        tier_s = f"[{t.get('tier', '候选')}] "
        L.append(f"  {i:>2}. {tier_s}{mem_s} (中心p={t['center']:02d}) 场分={t['score']:.3f} | "
                 f"共振={t.get('res', 1.0):.1f} 亲和={t.get('aff', 0.0):.2f} 覆盖={t['cov']:.2f} 遗漏={t['gap_s']:.2f} [{labs_s}]")
    L.append("")
    L.append(sep)
    L.append("  详细版附录 E：全80号综合分排序（动量/补冷方向见第1节双榜）")
    L.append(sep)
    order = np.argsort(-comp)
    row = "  "
    for i, idx in enumerate(order, 1):
        row += f"{idx+1:02d}={comp[idx]*100:4.1f}  "
        if i % 8 == 0:
            L.append(row)
            row = "  "
    if row.strip():
        L.append(row)
    L.append(f"  综合分区间: min={comp.min()*100:.1f} max={comp.max()*100:.1f} "
             f"均值={comp.mean()*100:.1f}")

    L.append("")
    L.append(sep)
    L.append("  详细版附录 F：近期命中率逐期完整明细（[8] 面板全量 · walk-forward）")
    L.append(sep)
    _rows, agg = None, None
    for k, v in ctx.items():
        if isinstance(k, str) and k.startswith("_hit_panel_cache"):
            _rows, agg = v
            break
    if agg and "err" not in agg and _rows:
        L.append("  期号      日期          点位   扩展环  主攻组  ≥2码组  3连全中  去重命中  八区16码")
        for r in _rows:
            L.append(f"  {r['period']}  {r['date']}  {r['core_hit']:2d}/{r['core_n']:2d}  "
                     f"{r['ext_hit']:2d}/{r['ext_n']:2d}   {r['g_n']:4d}   {r['g_hit2']:4d}   "
                     f"{r['g_full3']:4d}     {r['g_dedup']:2d}/{r['g_dedup_n']:2d}    "
                     f"{r['z16']:2d}/{r['z16_n']}")
        L.append(f"  合计: {agg['n']} 期 ｜ 主攻组去重 {agg['main_h']}/{agg['main_n']} "
                 f"= {agg['main_rate']*100:.2f}%（随机基线 25.00%）｜ "
                 f"3连全中 {sum(r['g_full3'] for r in _rows)} 组"
                 f"（基线 {agg['group_n']*agg['p_all3']*agg['n']:.1f} 组）")
    else:
        L.append("  （无逐期明细数据）")
    L.append(sep)
    return "\n".join(L)


def main(argv=None):
    """CLI 主入口函数。
    <p>
    【业务背景与架构定位】:
    解析命令行参数，支持指定点位文件、开奖历史、目标期号或直接通过 --raw-points 传入 20 个临时核心点位；
    自动执行 pipeline 计算并生成标准版与详细版研判分析报告。
    
    @param argv 命令行参数列表，默认 None (读取 sys.argv)
    @return 状态码 (0 表示成功，非 0 表示参数错误或异常)
    """
    import argparse
    ap = argparse.ArgumentParser(description="快乐8点位+遗漏深度分析")
    ap.add_argument("--points", default=CONFIG.get("points_file", "daily_points.txt"), help="点位文件")
    ap.add_argument("--history", default=CONFIG.get("history_file", "kl8_history_final.txt"), help="开奖历史文件")
    ap.add_argument("--period", type=int, default=None, help="指定分析期号（默认点位文件最新一期）")
    ap.add_argument("--raw-points", type=str, default=None, help="临时指定20个核心点位（以空格或连字符分隔，如 '05 12 14 ...'），无需改写文件")
    ap.add_argument("--out", default=None, help="报告输出路径")
    ap.add_argument("--detail", action="store_true", help="附加详细版附录")
    ap.add_argument("--hit-days", type=int, default=HIT_PANEL_DAYS,
                    help="命中率面板滚动天数（默认40）")
    args = ap.parse_args(argv)

    # 步骤 1: 加载底层数据集
    ds = load_dataset(args.points, args.history)

    # 步骤 2: 若传入了临时核心点位，则覆盖/注入当前分析任务中
    if args.raw_points:
        raw_s = args.raw_points.replace("-", " ").replace(",", " ")
        try:
            parsed_nums = [int(x) for x in raw_s.split() if x.strip()]
            assert len(parsed_nums) == 20 and len(set(parsed_nums)) == 20
            assert all(1 <= n <= N_NUM for n in parsed_nums)
        except Exception:
            print(f"[错误] --raw-points 必须包含 1-80 范围内的 20 个不重复有效号码，当前解析到 {len(parsed_nums) if 'parsed_nums' in locals() else 0} 个")
            return 2
        raw_period = args.period if args.period is not None else (
            next_period_after(ds.draws[-1]) if ds.draws else 999999)
        raw_date = next_target_date(ds) or datetime.date.today().strftime("%Y-%m-%d")
        ds.point_sets[raw_period] = frozenset(parsed_nums)
        ds.point_dates[raw_period] = raw_date
        ds.target_period = raw_period
        target = raw_period
        # 注入后旧目标告警已失效，过滤掉避免与最终目标期矛盾
        ds.warnings = [w for w in ds.warnings
                       if "未检测到未开奖预测期" not in w
                       and "未匹配到任何有效点位数据" not in w]
    else:
        target = args.period if args.period is not None else ds.target_period
        if args.period is not None and args.period not in ds.point_sets:
            print(f"[错误] 期号 {args.period} 在点位文件中不存在")
            return 2

    # 告警在点位注入确定目标期后统一输出
    for w in ds.warnings:
        print("[数据告警]", w)

    # 步骤 3: 截断历史防未来数据泄漏并执行计算流
    actual_target = ds.draws_by_period.get(target)  # 已开奖则用于结算
    ds = truncated(ds, target)
    feat, ctx = run_pipeline(ds, point_period=None)
    rep_std = build_report(ctx["ds"], feat, ctx, ds_orig=ds, actual_target=actual_target,
                           hit_days=args.hit_days)
    rep_full = rep_std + "\n\n" + build_detail(ctx, feat) if args.detail else rep_std
    print(rep_full)

    # 步骤 4: 持久化到 output/reports/（标准版不含附录；详细版另存，避免两份同内容）
    base = ds.point_dates.get(ds.target_period, "date").replace("-", "")
    default_std, default_detail = report_paths(base)
    out = args.out or default_std
    with open(out, "w", encoding="utf-8") as f:
        f.write(rep_std + "\n")
    print(f"\n[已写入] {out}")
    if args.detail:
        dout = default_detail if args.out is None else (
            args.out.replace(".txt", "_详细.txt") if args.out.endswith(".txt")
            else args.out + "_详细.txt"
        )
        with open(dout, "w", encoding="utf-8") as f:
            f.write(rep_full + "\n")
        print(f"[已写入] {dout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())