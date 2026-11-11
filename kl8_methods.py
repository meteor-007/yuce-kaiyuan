# -*- coding: utf-8 -*-
"""十大分析方法 + 连号集群三元组（重点模块）。
每个方法函数签名: m_xxx(ds, ts, ctx) -> (col_name, vec80, info_dict)
"""
import numpy as np
from collections import Counter

from kl8_core import (CONFIG, N_NUM, ring_prev, ring_next, minmax)

TRIPLE_WEIGHTS = {"cov": 0.20, "gap": 0.20, "burst": 0.15, "run": 0.15, "aff": 0.15, "res": 0.15}


# ============================================================
# ★重点：连号集群三元组（点位场+遗漏+同出协方差+空间共振）
# ============================================================

def detect_runs(nums_sorted):
    """排序号码 → 连续段列表"""
    if not nums_sorted:
        return []
    runs = []
    cur = [nums_sorted[0]]
    for x in nums_sorted[1:]:
        if x == cur[-1] + 1:
            cur.append(x)
        else:
            runs.append(cur)
            cur = [x]
    runs.append(cur)
    return runs


def run_counts(ds, ts, W=30):
    """近W期每号参与的2连/≥3连次数、以及每期相邻对数"""
    T = ds.T
    run2 = np.zeros(N_NUM)
    run3 = np.zeros(N_NUM)
    adj_pairs = []
    lo = max(0, T - W)
    for i in range(lo, T):
        nums = sorted(ds.draws[i].numbers)
        pairs = 0
        for r in detect_runs(nums):
            L = len(r)
            if L >= 2:
                pairs += L - 1
                for n in r:
                    run2[n - 1] += 1
            if L >= 3:
                for n in r:
                    run3[n - 1] += 1
        adj_pairs.append(pairs)
    return run2, run3, adj_pairs


def compute_cooccurrence_affinity(ts, T, mem, window=50):
    """计算三元组在近 window 期内的两两与三码同出亲和度（过滤貌合神离伪连号）。
    Affinity = (co_12 + co_23 + co_13 + 2 * co_123) / max(union_draws, 1)
    """
    if ts is None or not hasattr(ts, "hit") or T < 5:
        return 0.0
    w = min(window, T)
    sub = ts.hit[T - w:T, [m - 1 for m in mem]] > 0.5
    co12 = int((sub[:, 0] & sub[:, 1]).sum())
    co23 = int((sub[:, 1] & sub[:, 2]).sum())
    co13 = int((sub[:, 0] & sub[:, 2]).sum())
    co123 = int((sub[:, 0] & sub[:, 1] & sub[:, 2]).sum())
    union_n = int((sub.sum(axis=1) > 0).sum())
    if union_n <= 0:
        return 0.0
    return float(co12 + co23 + co13 + 2 * co123) / float(union_n)


def compute_point_proximity_resonance(p, core_set):
    """计算点位在核心点位拓扑结构中的紧邻共振度。
    基准分：点位自身在 core_set 为 1.0；
    相邻加成：三元组内部若包含其他核心点位，每多一个 +1.0；
    次邻加成：若在距离 2 的环形位置有核心点位，每个 +0.4。
    """
    if not core_set:
        return 1.0
    mem = (ring_prev(p), p, ring_next(p))
    core_in_mem = sum(1.0 for m in mem if m in core_set)
    res = float(core_in_mem + 0.4 * sum(1.0 for d in (-2, 2) if (((p - 1 + d) % N_NUM) + 1) in core_set))
    return res


def build_triples(ds, ts, ctx):
    """对目标期核心点位构造环形三元组并进行深度多维物理与动力学评分。
    返回 (triples, extended_stats, cluster_score80, info)
    triples: 按分数降序的 list[dict]
    """
    cfg = CONFIG
    T = ds.T
    target = ds.target_period
    core = sorted(ds.point_sets.get(target, ()))
    core_set = set(core)
    W, W3 = cfg["window_short"], cfg["window_burst"]
    Wr = cfg["window_run"]
    cap = cfg["gap_cap"]

    prev = ds.draws[T - 1] if T >= 1 else None
    prev_set = set(prev.numbers) if prev else set()

    run2, run3, adj_pairs = run_counts(ds, ts, Wr)
    exp_pairs = 79 * (20 * 19) / (80 * 79) * Wr  # 理论期望对数
    run_norm = max(np.percentile(run2 + 2 * run3, 90), 1.0)
    
    adj_mean = float(np.mean(adj_pairs)) if adj_pairs else 0.0
    adj_exp = exp_pairs / Wr if Wr else 4.75
    adj_sd = float(np.std(adj_pairs, ddof=1)) if len(adj_pairs) > 1 else 0.0
    adj_se = adj_sd / np.sqrt(len(adj_pairs)) if adj_sd > 0 and adj_pairs else 0.0
    cluster_z = float((adj_mean - adj_exp) / adj_se) if adj_se > 0 else 0.0
    if cluster_z > 1.96:
        cluster_state = "连号偏热（显著）"
    elif cluster_z < -1.96:
        cluster_state = "连号偏冷（显著）"
    else:
        cluster_state = "连号正常（偏离未达显著）"

    triples = []
    coverage = Counter()
    for p in core:
        mem = (ring_prev(p), p, ring_next(p))
        for m in mem:
            coverage[m] += 1
        h10 = np.array([ts.hits_window(m, T, W) for m in mem], dtype=float)
        h3 = np.array([ts.hits_window(m, T, W3) for m in mem], dtype=float)
        gap = np.array([min(ts.gap[T - 1, m - 1], cap) for m in mem], dtype=float)
        w = np.array([0.8, 1.2, 0.8])
        cov = float((w * h10).sum() / (30.0))
        gap_s = float(np.log1p(gap).sum() / 3.0 / np.log1p(cap))
        burst = float(h3.sum() / 3.0)
        mi = np.array(mem) - 1
        run_e = float((run2[mi] + 2 * run3[mi]).sum() / run_norm)
        
        # 深度挖掘算子：同出亲和度与点位拓扑共振度
        aff = compute_cooccurrence_affinity(ts, T, mem, window=50)
        res = compute_point_proximity_resonance(p, core_set)
        has_repeat = any(m in prev_set for m in mem)

        triples.append({
            "center": p, "members": mem,
            "h10": h10, "h3": h3, "gap": gap,
            "cov": cov, "gap_s": gap_s, "burst": burst, "run": run_e,
            "aff": aff, "res": res, "has_repeat": has_repeat,
        })

    # 各子分跨三元组 min-max 后合成
    def mm(key):
        v = np.array([t[key] for t in triples])
        return minmax(v)
    
    if triples:
        cov_n, gap_n, burst_n, run_n = mm("cov"), mm("gap_s"), mm("burst"), mm("run")
        aff_n, res_n = mm("aff"), mm("res")
        for i, t in enumerate(triples):
            t["score"] = (TRIPLE_WEIGHTS["cov"] * cov_n[i] + TRIPLE_WEIGHTS["gap"] * gap_n[i]
                          + TRIPLE_WEIGHTS["burst"] * burst_n[i] + TRIPLE_WEIGHTS["run"] * run_n[i]
                          + TRIPLE_WEIGHTS["aff"] * aff_n[i] + TRIPLE_WEIGHTS["res"] * res_n[i])
            
            # 精细化动力学分型
            labels = []
            if t["res"] >= 2.0 and t["aff"] >= 0.35:
                labels.append("双核超导")
            elif t["res"] >= 2.0:
                labels.append("点位共振")
            if t["burst"] >= 1.0 and t["run"] >= 0.5:
                labels.append("极速连号")
            if t["has_repeat"] and t["aff"] >= 0.30:
                labels.append("连庄滑行")
            if t["gap_s"] >= 0.6:
                labels.append("深漏回补")
            if t["burst"] >= 2 / 3 and t["run"] >= 0.5:
                labels.append("三连候选")
            if t["cov"] <= 0.1 and t["gap_s"] <= 0.2 and t["res"] <= 1.0:
                labels.append("低温弃子")
            if not labels:
                labels.append("结构防御")
            t["labels"] = labels
        triples.sort(key=lambda t: -t["score"])

    # 扩展集合逐号统计（近10命中/近3命中/连失/覆盖点位场数）
    ext = []
    for n in sorted(coverage):
        ext.append({
            "num": n, "cov": coverage[n],
            "h10": int(ts.hits_window(n, T, W)),
            "h3": int(ts.hits_window(n, T, W3)),
            "gap": int(ts.gap[T - 1, n - 1]),
        })

    # 号码级 ClusterScore = 所属三元组最高分
    cs = np.zeros(N_NUM)
    for t in triples:
        for m in t["members"]:
            cs[m - 1] = max(cs[m - 1], t["score"])

    if not core:
        info = {
            "core": [],
            "adj_pairs_mean": float(np.mean(adj_pairs)) if adj_pairs else 0.0,
            "adj_exp": adj_exp, "adj_se": adj_se,
            "cluster_z": cluster_z, "cluster_state": cluster_state,
            "run2": run2, "run3": run3, "extended": ext,
            "triples": triples,
        }
        return triples, ext, cs, info

    info = {
        "core": core,
        "adj_pairs_mean": adj_mean,
        "adj_exp": adj_exp, "adj_se": adj_se,
        "cluster_z": cluster_z, "cluster_state": cluster_state,
        "run2": run2, "run3": run3, "extended": ext,
        "triples": triples,
    }
    return triples, ext, cs, info


def triple_line(t, rank):
    mem_s = "%02d-%02d-%02d" % t["members"]
    lab = "/".join(t["labels"]) if t.get("labels") else "-"
    tier_s = f"[{t.get('tier', '候选')}] " if t.get("tier") else ""
    row = (f"{rank:>2}  {tier_s}{mem_s} (p={t['center']:02d})  场分={t['score']:.2f}  "
           f"覆盖={t['cov']:.2f} 遗漏={t['gap_s']:.2f} 亲和={t.get('aff', 0.0):.2f} 共振={t.get('res', 1.0):.1f}  {lab}")
    return row


def triples_table(triples, topn=10):
    lines = ["排名  推荐分级 点位场(三元组)   综合场分  覆盖   遗漏   亲和   共振   动力学标签"]
    for i, t in enumerate(triples[:topn], 1):
        lines.append(triple_line(t, i))
    return "\n".join(lines)


def select_main_groups(triples, floor=0.50, max_drop=0.15, max_groups=4):
    """金字塔分层推荐引擎（T1 极高共振核心 + T2 空间正交防御）：
    1. T1 核心组：场分最高且满足共振/亲和的领头组（1~2 组）；
    2. T2 防守组：在未被 T1 覆盖的其他宏观空间区间（低区 01-30、中区 31-55、高区 56-80）中挑选得分最高者（1~2 组），实现空间正交互补防空门。
    """
    ranked = [t for t in triples if "低温弃子" not in t.get("labels", [])]
    if not ranked:
        ranked = list(triples)
    if not ranked:
        return []
    
    top_score = float(ranked[0]["score"])
    cut = max(float(floor), top_score - float(max_drop))
    candidates = [t for t in ranked if float(t["score"]) >= cut] or ranked[:1]

    def macro_zone(center):
        return 1 if center <= 30 else (2 if center <= 55 else 3)

    # 步骤 1: 遴选 T1 核心组 (1 ~ 2 组)
    t1_groups = []
    t1_groups.append(candidates[0])
    candidates[0]["tier"] = "T1核心"
    
    if len(candidates) > 1 and float(candidates[1]["score"]) >= top_score - 0.08:
        t1_groups.append(candidates[1])
        candidates[1]["tier"] = "T1核心"

    covered_zones = {macro_zone(t["center"]) for t in t1_groups}
    
    # 步骤 2: 空间正交防御组 T2 (1 ~ 2 组)
    t2_groups = []
    remaining = [t for t in ranked if t not in t1_groups]
    for unrep_zone in (1, 2, 3):
        if unrep_zone not in covered_zones:
            zone_triples = [t for t in remaining if macro_zone(t["center"]) == unrep_zone]
            if zone_triples and float(zone_triples[0]["score"]) >= 0.40:
                best_zone_t = zone_triples[0]
                best_zone_t["tier"] = "T2防守"
                t2_groups.append(best_zone_t)
                remaining.remove(best_zone_t)
                if len(t1_groups) + len(t2_groups) >= max_groups:
                    break

    selected = t1_groups + t2_groups
    return selected


def triple_picks(triples, fill=(), n_groups=None):
    """从三元集群出号。
    主攻组 = 当天场分金字塔分层推荐（T1核心 + T2防御）；若显式传 n_groups 则按前 N 组（仅调试用）。
    五码/两码只从主攻组内取：中心优先，再按近10命中、连失排序去重。
    """
    ranked = [t for t in triples if "低温弃子" not in t.get("labels", [])]
    if not ranked:
        ranked = list(triples)
    if n_groups is not None:
        groups = ranked[: max(1, int(n_groups))]
    else:
        groups = select_main_groups(triples)
    ordered = []
    for t in groups:
        mem = list(t["members"])
        idx = {m: i for i, m in enumerate(mem)}
        center = t["center"]
        rest = [m for m in mem if m != center]
        h10 = t.get("h10")
        gap = t.get("gap")
        if h10 is not None and gap is not None:
            rest.sort(key=lambda m: (-float(h10[idx[m]]), -float(gap[idx[m]])))
        for m in [center] + rest:
            if int(m) not in ordered:
                ordered.append(int(m))
    for n in list(fill) + list(range(1, 81)):
        if len(ordered) >= 7:
            break
        if int(n) not in ordered:
            ordered.append(int(n))
    five = ordered[:5]
    two = ordered[5:7]
    return {
        "five": five,
        "two": two,
        "seven": five + two,
        "groups": groups,
        "skipped_cold": (len(triples) - len(ranked)) if ranked is not triples else 0,
    }


# 方法接口：供评分层调用
def m_cluster(ds, ts, ctx):
    triples, ext, cs, info = build_triples(ds, ts, ctx)
    ctx["triples"] = triples
    info["cluster_score"] = cs
    ctx["triple_info"] = info
    ctx["extended"] = ext
    return "cluster", cs, {"cluster_z": info["cluster_z"],
                           "cluster_state": info["cluster_state"],
                           "adj_pairs_mean": info["adj_pairs_mean"]}


# ============================================================
# 1) 马尔可夫
# ============================================================

def m_markov(ds, ts, ctx):
    W = min(CONFIG["window_markov"], ds.T)
    T = ds.T
    X = ts.hit[T - W:T, :]
    prev, cur = X[:-1], X[1:]
    n0 = (prev == 0).sum(axis=0)
    n1 = (prev == 1).sum(axis=0)
    p01 = ((prev == 0) & (cur == 1)).sum(axis=0)
    p11 = ((prev == 1) & (cur == 1)).sum(axis=0)
    P01 = (p01 + 1) / (n0 + 2)
    P11 = (p11 + 1) / (n1 + 2)
    score = 0.6 * P01 + 0.4 * P11
    score[(n0 < 5) | (n1 < 5)] = np.nan
    top30 = list(np.argsort(-np.nan_to_num(P01, nan=0))[:CONFIG["markov_follow_top"]] + 1)
    info = {"P01": P01, "P11": P11, "p01_top30": top30,
            "low_conf": int(np.isnan(score).sum())}
    return "markov", score, info


def m_fib(ds, ts, ctx):
    T = ds.T
    weights = (3, 5, 8, 13)
    score = np.zeros(N_NUM)
    wsum = 0.0
    for w in weights:
        if T < w:
            continue
        score += w * ts.hit[T - w:T].mean(axis=0)
        wsum += w
    if wsum > 0:
        score /= wsum
    segs = [(0, 30), (30, 54), (54, 80)]
    exp = np.array([(b - a) / 80.0 for a, b in segs])
    obs = np.zeros(3)
    W = min(30, T)
    for i in range(T - W, T):
        for n in ds.draws[i].numbers:
            if n <= 30:
                obs[0] += 1
            elif n <= 54:
                obs[1] += 1
            else:
                obs[2] += 1
    obs = obs / max(obs.sum(), 1)
    dev = obs - exp
    bias = int(np.argmax(np.abs(dev)))
    return "fib", score, {"phi_bias_seg": bias, "phi_bias_dev": float(dev[bias])}


def m_linalg(ds, ts, ctx):
    W = min(CONFIG["window_cooccur"], ds.T)
    X = ts.hit[-W:]
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        svd = np.abs(vt[0])
    except np.linalg.LinAlgError:
        svd = np.full(N_NUM, np.nan)
    # 条件概率：上期已开时，本期共现率
    prev, cur = X[:-1], X[1:]
    n1 = prev.sum(axis=0)
    cp = ((prev == 1) & (cur == 1)).sum(axis=0)
    cp = (cp + 1) / (n1 + 2)
    if "feat" in ctx:
        ctx["feat"].add("cp", cp)
    return "svd", svd, {"cp": cp}


def m_complex(ds, ts, ctx):
    W = min(CONFIG["window_complex"], ds.T)
    X = ts.hit[-W:]
    period = np.full(N_NUM, np.nan)
    spec = np.zeros(N_NUM)
    for n in range(N_NUM):
        x = X[:, n] - X[:, n].mean()
        if np.allclose(x, 0):
            continue
        mag = np.abs(np.fft.rfft(x))
        mag[0] = 0
        k = int(np.argmax(mag))
        if k <= 0:
            continue
        period[n] = W / k
        spec[n] = mag[k]
    bins = {}
    for p in period[np.isfinite(period)]:
        key = int(round(p))
        if 2 <= key <= 30:
            bins[key] = bins.get(key, 0) + 1
    dom = sorted(bins.items(), key=lambda kv: -kv[1])[:3]
    return "complex", spec, {"period": period, "dominant_periods": dom}


def m_graph(ds, ts, ctx):
    """共现图度中心性 → 拥挤度（供对抗模块折价使用）。

    社群划分（标签传播）经审计其结果不进入任何评分维度与报告正文，
    已按「移除过于复杂且无业务出口的实现」原则删除。
    """
    W = min(CONFIG["window_graph"], ds.T)
    X = ts.hit[-W:]
    co = X.T @ X
    np.fill_diagonal(co, 0)
    thr = np.percentile(co[co > 0], 75) if np.any(co > 0) else 1
    adj = co >= thr
    deg = adj.sum(axis=1).astype(float)
    crowding = minmax(deg)
    return "graph", crowding, {"crowding": crowding, "n_edges": int(adj.sum() // 2)}


def m_volatility(ds, ts, ctx):
    W = min(CONFIG["window_vol"], ds.T)
    X = ts.hit[-W:]
    vol = X.std(axis=0)
    rate = X.mean(axis=0)
    # 三分位代替重型聚类，标签稳定
    q1, q2 = np.quantile(vol, [1 / 3, 2 / 3])
    labels = np.where(vol <= q1, 0, np.where(vol <= q2, 1, 2))
    names = {0: "冷稳组", 1: "间歇组", 2: "高频震荡组"}
    crate = {c: float(rate[labels == c].mean()) if np.any(labels == c) else 0.0 for c in (0, 1, 2)}
    draws = ds.draws
    def jacc(a, b):
        return len(a & b) / 20.0
    pairs = [jacc(draws[i].numbers, draws[i - 1].numbers) for i in range(1, len(draws))]
    j10 = float(np.mean(pairs[-10:])) if pairs else 0.0
    j50 = float(np.mean(pairs[-50:])) if len(pairs) >= 50 else (float(np.mean(pairs)) if pairs else 0.0)
    if j10 > j50 + 0.03:
        state = "重号偏热"
    elif j10 < j50 - 0.03:
        state = "重号偏冷"
    else:
        state = "正常"
    return "vol", vol, {
        "labels": labels, "cluster_names": names, "cluster_rate": crate,
        "j10": j10, "j50": j50, "j_state": state,
    }


def m_entropy(ds, ts, ctx):
    W = min(CONFIG["window_main"], ds.T)
    X = ts.hit[-W:]
    # 每期出号在 16 扇区上的香农熵
    ents = []
    for i in range(ds.T - W, ds.T):
        c = np.zeros(16)
        for n in ds.draws[i].numbers:
            c[(n - 1) // 5] += 1
        p = c / c.sum()
        p = p[p > 0]
        ents.append(float(-(p * np.log(p)).sum()))
    ents = np.array(ents)
    mu, sd = ents.mean(), ents.std()
    z = 0.0 if sd < 1e-12 else float((ents[-1] - mu) / sd)
    # 号码级：序列越可预测（低熵）分越高，取负熵
    p_hat = np.clip(X.mean(axis=0), 1e-6, 1 - 1e-6)
    ent_n = -(p_hat * np.log(p_hat) + (1 - p_hat) * np.log(1 - p_hat))
    score = -ent_n
    return "entropy", score, {"entropy_anomaly": z}


def m_adversarial(ds, ts, ctx):
    T = ds.T
    base = ctx.get("composite_h16")
    if base is None:
        base = ctx["feat"].get_norm("fib") if "fib" in ctx["feat"].cols else np.zeros(N_NUM)
    gains = []
    W = min(50, T - 1)
    for t in range(T - W, T):
        window = ts.hit[max(0, t - 10):t]
        if len(window) == 0:
            continue
        heat = window.mean(axis=0)
        top = np.argsort(-heat)[:20]
        gains.append(float(ts.hit[t, top].sum()))
    gains_mean = float(np.mean(gains)) if gains else 5.0
    lam = gains_mean / 5.0
    crowd = ctx.get("crowding", np.zeros(N_NUM))
    # 拥挤热号折价，冷号抬升
    contra = (1 - minmax(base)) * (0.5 + 0.5 * crowd)
    if lam >= 1.05:
        contra = 1 - contra
    return "contra", contra, {"lambda": float(lam), "gains_mean": gains_mean}


def sector_energy(ds, ts, W=20):
    T = ds.T
    energy = np.zeros(16)
    for i in range(T - W, T):
        for m in ds.draws[i].numbers:
            energy[(m - 1) // 5] += 1
    obs = energy / energy.sum()
    exp = 5.0 / 80.0
    dev = (obs - exp) / exp
    return energy, obs, dev


def energy_centroid(ds, ts, W=20):
    T = ds.T
    s = 0j
    for i in range(T - W, T):
        for m in ds.draws[i].numbers:
            s += np.exp(2j * np.pi * (m - 1) / 80)
    theta = (np.angle(s) % (2 * np.pi)) * 80 / (2 * np.pi) + 1
    return float(theta), abs(s) / (20 * W)


# ============================================================
# 微积分 + 八区分区预测（80→8区×10号，每区最优2码）
# 定稿：区内z分回补（due/spring/cool）。点位加成与多路投票经分段回测均不优于纯公式。
# ============================================================

CALC_ZONE_WEIGHTS = {
    "due": 0.40,
    "spring": 0.35,
    "cool": 0.25,
}


def calc_zone_bounds(zone_id, n_zones=None, zone_size=None):
    """区号 0..7 → (lo, hi) 含端 1-based 号码。"""
    n_zones = int(n_zones or CONFIG["calc_zone_n"])
    zone_size = int(zone_size or CONFIG["calc_zone_size"])
    lo = zone_id * zone_size + 1
    hi = min((zone_id + 1) * zone_size, N_NUM)
    return lo, hi


def _zone_zscore(arr):
    """区内 z-score；方差为 0 时返回 0 向量。"""
    a = np.asarray(arr, dtype=float)
    sd = float(a.std())
    if sd < 1e-12:
        return np.zeros_like(a)
    return (a - a.mean()) / sd


def build_calc_zones(ds, ts, ctx=None):
    """离散微积分分区选号。

    量：∫短/中/长窗密度，v=∫s−∫l，a 二阶差，G=遗漏/帽
    子分：due=G·(1−∫s)，spring=relu(v)·G，cool=1−∫s
    区内 z-score 后按固定权重合成，取 Top-2。不混入点位加成。
    @return (zones, picks16, score80, info)
    """
    n_zones = int(CONFIG["calc_zone_n"])
    zone_size = int(CONFIG["calc_zone_size"])
    top_k = int(CONFIG["calc_zone_top_k"])
    W = min(int(CONFIG["window_calc"]), ds.T)
    Ws = min(int(CONFIG["window_calc_short"]), W)
    Wm = min(int(CONFIG["window_calc_mid"]), W)
    T = ds.T
    gap_cap = float(CONFIG["gap_cap"])
    w0 = CALC_ZONE_WEIGHTS

    core = set(ds.point_sets.get(ds.target_period, ()))

    integ = np.zeros(N_NUM)
    vel = np.zeros(N_NUM)
    acc = np.zeros(N_NUM)
    due = np.zeros(N_NUM)
    spring = np.zeros(N_NUM)
    cool = np.zeros(N_NUM)
    gaps = np.zeros(N_NUM)
    k_scores = np.zeros(N_NUM)
    p_scores = np.zeros(N_NUM)

    for i in range(N_NUM):
        x = ts.hit[T - W:T, i].astype(float) if T else np.zeros(0)
        if len(x) == 0:
            continue
        s_s = float(x[-Ws:].mean()) if Ws else 0.0
        s_m = float(x[-Wm:].mean()) if Wm else s_s
        s_l = float(x.mean())
        v = s_s - s_l
        a = (s_s - s_m) - (s_m - s_l)
        g = float(ts.gap[T - 1, i]) if T > 0 else 0.0
        g_n = min(g, gap_cap) / gap_cap
        integ[i], vel[i], acc[i], gaps[i] = s_s, v, a, g
        due[i] = g_n * (1.0 - s_s)
        spring[i] = max(v, 0.0) * g_n
        cool[i] = 1.0 - s_s

        # 纯开奖微积分双态动力学：
        # 1. 动能分 (Kinetic): 顺势速度v + 加速度a + 密度s_s (连失>4期实施惩罚，锁定短线冲顶爆发态)
        k_scores[i] = 0.50 * s_s + 0.30 * v + 0.20 * a - (0.40 if g > 4.0 else 0.0)
        # 2. 势能分 (Potential): 针对适度遗漏号的弹性恢复力与拐点反弹势能
        p_scores[i] = (min(g, 15.0) / 15.0) * (1.0 - s_s) + 0.25 * max(v, 0.0)

    score80 = np.zeros(N_NUM)
    meta = [None] * N_NUM
    zones = []
    picks16 = []

    for z in range(n_zones):
        lo, hi = calc_zone_bounds(z, n_zones, zone_size)
        members = list(range(lo, hi + 1))
        sl = slice(lo - 1, hi)
        zden = float(integ[sl].mean()) if hi >= lo else 0.25

        rows = []
        for j, n in enumerate(members):
            i = n - 1
            # 综合展示分为动能与势能的均衡
            sc = float(0.5 * k_scores[i] + 0.5 * p_scores[i])
            score80[i] = sc
            row = {
                "num": n,
                "integral": float(integ[i]),
                "velocity": float(vel[i]),
                "accel": float(acc[i]),
                "gap_rec": float(due[i]),
                "spring": float(spring[i]),
                "cool": float(cool[i]),
                "gap": float(gaps[i]),
                "k_score": float(k_scores[i]),
                "p_score": float(p_scores[i]),
                "score": sc,
            }
            rows.append(row)
            meta[i] = row

        # 双态选号：1码动能顺势码 (Kinetic Top) + 1码势能反弹码 (Potential Top)
        k_sorted = sorted(rows, key=lambda r: (-r["k_score"], -r["score"], r["num"]))
        p_sorted = sorted(rows, key=lambda r: (-r["p_score"], -r["score"], r["num"]))

        top_k_item = k_sorted[0]
        top_p_item = p_sorted[0]
        if top_p_item["num"] == top_k_item["num"] and len(p_sorted) > 1:
            top_p_item = p_sorted[1]

        top = [top_k_item, top_p_item]
        top_nums = [top_k_item["num"], top_p_item["num"]]

        zone_hits = float(ts.hit[T - W:T, lo - 1:hi].sum()) if T else 0.0
        expect = W * len(members) * 0.25
        energy_dev = (zone_hits / expect - 1.0) if expect else 0.0
        label = ("区热" if zden > 0.26 else ("区冷" if zden < 0.24 else "区平"))

        zones.append({
            "zone_id": z + 1,
            "lo": lo,
            "hi": hi,
            "members": members,
            "top2": top_nums,
            "top2_detail": top,
            "ranked": sorted(rows, key=lambda r: (-r["score"], r["num"])),
            "energy_hits": zone_hits,
            "energy_dev": energy_dev,
            "zone_density": zden,
            "label": label,
            "weights": {"k_weight": 0.5, "p_weight": 0.5},
        })
        picks16.extend(top_nums)

    for z in zones:
        idxs = [n - 1 for n in z["members"]]
        vals = score80[idxs]
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax > vmin:
            score80[idxs] = (vals - vmin) / (vmax - vmin)
        else:
            score80[idxs] = 0.5

    info = {
        "n_zones": n_zones,
        "zone_size": zone_size,
        "top_k": top_k,
        "window": W,
        "weights": dict(w0),
        "picks16": picks16,
        "zones": zones,
        "meta": meta,
        "scoring": "pure_draw_dual_calculus (Kinetic + Potential)",
    }
    return zones, picks16, score80, info


def calc_zone_picks(zones=None, info=None):
    """结构化出号：每区最优2码 → 共16码。"""
    if zones is None and info is not None:
        zones = info.get("zones") or []
    zones = zones or []
    by_zone = []
    sixteen = []
    for z in zones:
        top2 = list(z.get("top2") or [])
        by_zone.append({
            "zone_id": z["zone_id"],
            "range": f"{z['lo']:02d}-{z['hi']:02d}",
            "top2": top2,
            "label": z.get("label", ""),
            "energy_dev": z.get("energy_dev", 0.0),
        })
        sixteen.extend(top2)
    return {"by_zone": by_zone, "sixteen": sixteen, "n": len(sixteen)}


def calc_zones_table(zones, show_all=False):
    """报告用文本表。"""
    lines = []
    lines.append("分区  号码范围   区态   密度   动能码(顺势速) 势能码(弹性反)        微积分动力学明细(k分/p分/v/a/due)")
    for z in zones:
        t0, t1 = (z["top2_detail"] + [None, None])[:2]

        def _fmt(t, role):
            if t is None:
                return "——"
            return (f"{role}:{t['num']:02d}(k{t['k_score']:+.2f} p{t['p_score']:.2f}"
                    f" v{t['velocity']:+.2f} a{t['accel']:+.2f} g{int(t['gap'])})")

        picks_s = f"{t0['num']:02d}(动)  {t1['num']:02d}(势)" if t0 and t1 else "——"
        dens = z.get("zone_density", 0.0)
        lines.append(
            f" Z{z['zone_id']}   {z['lo']:02d}-{z['hi']:02d}   "
            f"{z['label']:<4} {dens:.2f}  "
            f"{picks_s:<14}  {_fmt(t0, '动')} | {_fmt(t1, '势')}"
        )
        if show_all:
            rest = " ".join(
                f"{r['num']:02d}:{r['score']:.2f}" for r in z["ranked"][2:6]
            )
            if rest:
                lines.append(f"       次优对照: {rest}")
    return "\n".join(lines)


def m_calc_zone(ds, ts, ctx):
    """方法入口：微积分八区分区 → feat列 calc_zone + ctx 结构化结果。"""
    zones, picks16, score80, info = build_calc_zones(ds, ts, ctx)
    ctx["calc_zones"] = zones
    ctx["calc_zone_info"] = info
    ctx["calc_zone_picks"] = calc_zone_picks(zones)
    return "calc_zone", score80, {
        "picks16": picks16,
        "n_zones": info["n_zones"],
        "top_k": info["top_k"],
        "window": info["window"],
        "scoring": info.get("scoring", ""),
    }


# ============================================================
# END_OF_METHODS（后续方法在此行之前插入）
# ============================================================
