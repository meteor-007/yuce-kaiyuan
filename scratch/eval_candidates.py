# -*- coding: utf-8 -*-
"""候选方法 walk-forward 评估（实验脚本，不改动生产代码）。

A. Dirichlet-multinomial 收缩（均匀先验 alpha=1/10/100，窗口 100；附 30/200/全历史敏感性）
B. 每号特征 逻辑回归 / 梯度提升（频率窗 10/30/100、遗漏、上期指示、环形邻接），
   严格 walk-forward：每个预测期只用该期之前的开奖，分块重训。
C. 点位场约束下的三元组选择检验：主攻组 vs 同场随机三元组；另加同候选池随机对照。
D. 覆盖设计（对子覆盖三元轮盘）覆盖 vs 命中率模拟，不宣称 edge。

评估窗口：与现有回测一致的最后 299 个有点位开奖期（2026001..2026299）。
输出：output/diagnostics/candidates_eval_<date>.json / .txt
"""
import os
import sys
import json
import math
import time
import hashlib
from itertools import combinations

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from kl8_core import load_dataset, truncated, TimeSeries  # noqa: E402
from kl8_methods import build_triples, triple_picks  # noqa: E402

OUT_DIR = os.path.join(ROOT, "output", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)
N_EVAL = 299
RNG = np.random.default_rng(20261110)
PERM_B = 20000
BOOT_B = 10000


# ---------------------------------------------------------------- stats utils
def summarize(name, method, hits, base, notes):
    arr = np.asarray(hits, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    if n > 1:
        lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
        t, p = stats.ttest_1samp(arr, base)
    else:
        lo = hi = mean
        p = 1.0
    return {
        "name": name, "method": method, "n": int(n),
        "mean": mean, "base": float(base),
        "lift": (mean / base) if base else 0.0,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "p": float(p), "beats_baseline": bool(lo > base),
        "notes": notes,
    }


def paired_perm_p(diffs, rng, B=PERM_B):
    """符号翻转置换检验（双侧）。"""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(perm >= obs)) / (B + 1))


def boot_mean_ci(arr, rng, B=BOOT_B):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    means = a[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def rate_summary(name, method, rates, base_rates, notes, rng, paired=True):
    """率型实验：mean=逐期观测率均值，base=逐期零假设率均值，CI=bootstrap(观测率)。"""
    rates = np.asarray(rates, dtype=float)
    base_rates = np.asarray(base_rates, dtype=float)
    mean = float(rates.mean())
    base = float(base_rates.mean())
    lo, hi = boot_mean_ci(rates, rng)
    diffs = rates - base_rates
    p = paired_perm_p(diffs, rng) if paired else 1.0
    dmean = float(diffs.mean())
    dse = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    dlo, dhi = (dmean - 1.96 * dse, dmean + 1.96 * dse) if len(diffs) > 1 else (dmean, dmean)
    full_notes = (f"{notes} | 配对差={dmean:.4f}, 差95%CI[{dlo:.4f},{dhi:.4f}], "
                  f"置换p={p:.4f}")
    return {
        "name": name, "method": method, "n": int(len(rates)),
        "mean": mean, "base": base,
        "lift": (mean / base) if base else 0.0,
        "ci_lo": lo, "ci_hi": hi,
        "p": float(p), "beats_baseline": bool(lo > base),
        "notes": full_notes,
    }


# ---------------------------------------------------------------- experiment A
def exp_a(ds, H, eval_periods, idx, W, alpha):
    h5, h20 = [], []
    for p in eval_periods:
        t = idx[p]
        if t < W:
            continue
        counts = H[t - W:t].sum(axis=0)
        post = (counts + alpha) / (20.0 * W + 80.0 * alpha)
        order = np.argsort(-post)
        actual = ds.draws[t].numbers
        h5.append(len(set(order[:5] + 1) & actual))
        h20.append(len(set(order[:20] + 1) & actual))
    return h5, h20


def exp_a_expanding(ds, H, eval_periods, idx, alpha):
    """全历史（扩张窗）Dirichlet 后验——贝叶斯频率估计的规范形式。"""
    h5, h20 = [], []
    for p in eval_periods:
        t = idx[p]
        if t < 1:
            continue
        counts = H[:t].sum(axis=0)
        post = (counts + alpha) / (20.0 * t + 80.0 * alpha)
        order = np.argsort(-post)
        actual = ds.draws[t].numbers
        h5.append(len(set(order[:5] + 1) & actual))
        h20.append(len(set(order[:20] + 1) & actual))
    return h5, h20


def run_exp_a(ds, H, eval_periods, idx, results):
    primary_h5 = primary_h20 = None
    identical = True
    sens = {}
    for W in (30, 100, 200):
        per_alpha = {}
        for alpha in (1, 10, 100):
            h5, h20 = exp_a(ds, H, eval_periods, idx, W, alpha)
            per_alpha[alpha] = (tuple(h5), tuple(h20))
            if W == 100 and alpha == 1:
                primary_h5, primary_h20 = h5, h20
        ref5, ref20 = per_alpha[1]
        identical = identical and all(v[0] == ref5 and v[1] == ref20
                                      for v in per_alpha.values())
        h5, h20 = exp_a(ds, H, eval_periods, idx, W, 10)
        m5 = float(np.mean(h5)); m20 = float(np.mean(h20))
        t5, p5 = stats.ttest_1samp(h5, 1.25)
        t20, p20 = stats.ttest_1samp(h20, 5.0)
        sens[W] = {"top5_mean": m5, "top5_lift": m5 / 1.25, "top5_p": float(p5),
                   "top20_mean": m20, "top20_lift": m20 / 5.0, "top20_p": float(p20)}
    h5e, h20e = exp_a_expanding(ds, H, eval_periods, idx, 10)
    sens["expanding"] = {"top5_mean": float(np.mean(h5e)), "top5_lift": float(np.mean(h5e) / 1.25),
                         "top20_mean": float(np.mean(h20e)), "top20_lift": float(np.mean(h20e) / 5.0)}
    note_base = ("窗口=100。均匀对称先验下后验均值 (n_i+α)/(20W+80α) 对 n_i 单调，"
                 f"α∈{{1,10,100}} 排序完全相同（实测={identical}），收缩不改变选号；"
                 f"敏感性: W=30 lift5={sens[30]['top5_lift']:.3f}, "
                 f"W=200 lift5={sens[200]['top5_lift']:.3f}, "
                 f"全历史 lift5={sens['expanding']['top5_lift']:.3f}。"
                 "本质是热号频率策略，与既有H1热度高度重合。")
    results.append(summarize("A_dirichlet_top5", "Dirichlet-multinomial 后验均值排序 (W=100, α=1/10/100)",
                             primary_h5, 1.25, note_base))
    results.append(summarize("A_dirichlet_top20", "Dirichlet-multinomial 后验均值排序 (W=100, α=1/10/100)",
                             primary_h20, 5.0, note_base))


# ---------------------------------------------------------------- experiment B
def make_features(ds, H):
    T = H.shape[0]
    ts = TimeSeries(ds)
    i = np.arange(80)
    p1, n1 = (i - 1) % 80, (i + 1) % 80
    p2, n2 = (i - 2) % 80, (i + 2) % 80
    F = np.zeros((T, 80, 7), dtype=np.float32)
    for t in range(1, T):
        F[t, :, 0] = H[max(0, t - 10):t].mean(axis=0)
        F[t, :, 1] = H[max(0, t - 30):t].mean(axis=0)
        F[t, :, 2] = H[max(0, t - 100):t].mean(axis=0)
        F[t, :, 3] = np.log1p(np.minimum(ts.gap[t - 1], 50.0))
        last = H[t - 1]
        F[t, :, 4] = last
        F[t, :, 5] = 0.5 * (last[p1] + last[n1])
        F[t, :, 6] = 0.5 * (last[p2] + last[n2])
    return F


def walkforward_model(ds, H, F, eval_t, kind="lr", block=20, s0=200, seed=0):
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
    hits5, hits20 = [], []
    model = scaler = None
    nfit = 0
    for pos, t in enumerate(eval_t):
        if model is None or pos % block == 0:
            X = F[s0:t].reshape(-1, F.shape[2])
            y = H[s0:t].reshape(-1)
            scaler = StandardScaler().fit(X)
            Xs = scaler.transform(X)
            if kind == "lr":
                model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            else:
                model = HistGradientBoostingClassifier(
                    max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                    min_samples_leaf=500, l2_regularization=1.0,
                    early_stopping=False, random_state=seed)
            model.fit(Xs, y)
            nfit += 1
        Xt = scaler.transform(F[t])
        if kind == "lr":
            score = model.decision_function(Xt)
        else:
            score = model.predict_proba(Xt)[:, 1]
        order = np.argsort(-score)
        actual = ds.draws[t].numbers
        hits5.append(len(set(order[:5] + 1) & actual))
        hits20.append(len(set(order[:20] + 1) & actual))
    return hits5, hits20, nfit


def run_exp_b(ds, H, eval_periods, idx, results):
    F = make_features(ds, H)
    eval_t = [idx[p] for p in eval_periods]
    notes_common = ("特征: 近10/30/100期频率、log1p(遗漏 cap50)、上期指示、"
                    "环形±1/±2邻接上期出号比例；StandardScaler+LR(C=1)。"
                    "严格 walk-forward: 每20期用此前全部开奖重训，预测块内各期仅用重训点之前数据。")
    t0 = time.time()
    h5, h20, nfit = walkforward_model(ds, H, F, eval_t, kind="lr", block=20, s0=200)
    notes = notes_common + f" 重训{nfit}次，用时{time.time()-t0:.1f}s。"
    results.append(summarize("B_logreg_top5", "每号特征 LogisticRegression walk-forward", h5, 1.25, notes))
    results.append(summarize("B_logreg_top20", "每号特征 LogisticRegression walk-forward", h20, 5.0, notes))
    t0 = time.time()
    try:
        g5, g20, nfit2 = walkforward_model(ds, H, F, eval_t, kind="gbdt", block=20, s0=200)
        notes_g = ("HistGradientBoosting(max_iter=150, lr=0.05, leaves=15, min_leaf=500, l2=1.0)，"
                   "同一严格 walk-forward 切分与特征。" + f" 重训{nfit2}次，用时{time.time()-t0:.1f}s。")
        results.append(summarize("B_gbdt_top5", "每号特征 HistGradientBoosting walk-forward", g5, 1.25, notes_g))
        results.append(summarize("B_gbdt_top20", "每号特征 HistGradientBoosting walk-forward", g20, 5.0, notes_g))
    except Exception as e:  # pragma: no cover
        results.append(summarize("B_gbdt_top5", "HistGradientBoosting", [0.0], 1.25,
                                 f"GBDT 不可用/失败: {e}"))
    return F


# ---------------------------------------------------------------- experiment C
def run_exp_c(ds, eval_periods, results, rng):
    rows = []
    t0 = time.time()
    for p in eval_periods:
        tds = truncated(ds, p)
        ts = TimeSeries(tds)
        triples, _, _, _ = build_triples(tds, ts, {})
        pk = triple_picks(triples)
        groups = pk["groups"]
        actual = ds.draws_by_period[p].numbers
        field = set(ds.point_sets[p])
        h = len(field & actual)
        g = 0.0
        if h >= 2:
            g = (math.comb(h, 2) * (20 - h) + math.comb(h, 3)) / math.comb(20, 3)
        wins = sum(1 for gr in groups if len(set(gr["members"]) & actual) >= 2)
        m = len(groups)
        rated = [x for x in triples if "低温弃子" not in x.get("labels", [])] or list(triples)
        q = sum(1 for x in rated if len(set(x["members"]) & actual) >= 2) / max(len(rated), 1)
        rows.append({
            "period": int(p), "m": int(m), "wins": int(wins),
            "rate": (wins / m) if m else float("nan"),
            "h": int(h), "g_field": float(g), "q_pool": float(q),
            "pool_n": int(len(rated)),
        })
    rates = np.array([r["rate"] for r in rows], dtype=float)
    gs = np.array([r["g_field"] for r in rows], dtype=float)
    qs = np.array([r["q_pool"] for r in rows], dtype=float)
    total_groups = int(sum(r["m"] for r in rows))
    total_wins = int(sum(r["wins"] for r in rows))
    pooled = total_wins / total_groups if total_groups else 0.0
    note_c1 = (f"主攻三元组（T1+T2，共{total_groups}组/299期，均值{total_groups/len(rows):.2f}组/期）"
               f"≥2码实测合并率={pooled:.4f}（{total_wins}/{total_groups}）。"
               "基线=同场20码中随机抽3码的精确超几何P(≥2|h)=[C(h,2)(20-h)+C(h,3)]/C(20,3)，逐期按实际h计算。"
               "注意：模型三元组含场外环形邻号，与纯场随机三元组构成不完全同分布，此对照偏严格/偏保守；"
               f"建triples用时{time.time()-t0:.1f}s。")
    results.append(rate_summary("C_field_random_triples", "主攻三元组 vs 同20点场随机三元组(精确超几何零假设)",
                                rates, gs, note_c1, rng))
    note_c2 = ("公平选择检验：基线=同一期模型全部合规候选三元组（环形邻域，剔除低温弃子）中的≥2码比例 q；"
               "主攻组是从该候选池按场分选出，检验选择步骤是否优于同池随机选择。")
    results.append(rate_summary("C_candidate_pool_random", "主攻组 vs 同候选池随机三元组",
                                rates, qs, note_c2, rng))
    return rows


# ---------------------------------------------------------------- experiment D
def build_cover(v=20, k=3, restarts=40, seed=7):
    """贪心+随机重启构造 v 个点的对子覆盖三元组集合。"""
    tris = list(combinations(range(v), k))
    pairs = list(combinations(range(v), 2))
    pidx = {pr: i for i, pr in enumerate(pairs)}
    masks = []
    for tri in tris:
        m = 0
        for a, b in combinations(tri, 2):
            m |= 1 << pidx[(a, b)]
        masks.append(m)
    full = (1 << len(pairs)) - 1
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        unc = full
        chosen = []
        while unc:
            best_c = 0
            ties = []
            for i, mk in enumerate(masks):
                c = (mk & unc).bit_count()
                if c > best_c:
                    best_c = c
                    ties = [i]
                elif c == best_c and c > 0:
                    ties.append(i)
            i = int(rng.choice(ties)) if len(ties) > 1 else ties[0]
            chosen.append(i)
            unc &= ~masks[i]
        if best is None or len(chosen) < len(best):
            best = chosen
    cover_ok = True
    u = 0
    for i in best:
        u |= masks[i]
    cover_ok = (u == full)
    return [tris[i] for i in best], cover_ok


def run_exp_d(ds, eval_periods, results, rng):
    design, cover_ok = build_cover(20, 3, restarts=40, seed=7)
    m_tickets = len(design)
    wheel_rates, field_rates, any_wheel, any_random, per_num = [], [], [], [], []
    for p in eval_periods:
        field = sorted(ds.point_sets[p])
        actual = ds.draws_by_period[p].numbers
        hit = np.array([1.0 if n in actual else 0.0 for n in field])
        h = int(hit.sum())
        g = 0.0
        if h >= 2:
            g = (math.comb(h, 2) * (20 - h) + math.comb(h, 3)) / math.comb(20, 3)
        win = 0
        for tri in design:
            if hit[list(tri)].sum() >= 2:
                win += 1
        wheel_rates.append(win / m_tickets)
        field_rates.append(g)
        any_wheel.append(1.0 if win > 0 else 0.0)
        any_random.append(float(1.0 - (1.0 - g) ** m_tickets))
        per_num.append(h / 20.0)
    note_ticket = (f"轮盘=20码场上的贪心对子覆盖设计，{m_tickets} 注三元（覆盖完整={cover_ok}，"
                   f"理论下界ceil(190/3)=64）。逐注≥2码率 vs 同场随机三元组精确超几何率 g(h)。"
                   "覆盖设计只改变注与注之间的相关结构，不改变任何号码25%的边际命中率；"
                   "期望返奖率不变，成本为单注的67倍量级。")
    results.append(rate_summary("D_wheel_ticket_hit", "对子覆盖轮盘逐注≥2码率 vs 随机三元组",
                                wheel_rates, field_rates, note_ticket, rng))
    note_cov = ("覆盖保证：h≥2 时轮盘必有至少1注≥2码（构造保证）；基线=同数量随机注至少1注≥2码的概率 "
                "1-(1-g)^m（独立性近似）。这是组合覆盖性质，不是预测 edge——"
                "随机买同样多的注在绝大多数期同样能中，且成本相同。")
    results.append(rate_summary("D_wheel_coverage", "轮盘覆盖(至少1注≥2码) vs 同数量随机注",
                                any_wheel, any_random, note_cov, rng))
    per_num = np.asarray(per_num)
    results.append(summarize("D_field_per_number_rate", "点位场20码单号命中率（轮盘不改变边际）",
                             per_num * 20.0, 5.0,
                             "按每期场命中数 h/20 折算的单号命中率；任何固定20码集合的期望恒为25%，"
                             "覆盖设计无法改变该边际。"))
    return design, m_tickets, cover_ok


# ---------------------------------------------------------------- main
def main():
    t_start = time.time()
    ds = load_dataset()
    H = ds.draw_matrix()
    idx = {d.period: d.idx for d in ds.draws}
    all_periods = sorted(p for p in ds.point_sets if p in ds.draws_by_period)
    eval_periods = all_periods[-N_EVAL:]
    data_hash = hashlib.sha256()
    with open(os.path.join(ROOT, "kl8_history_final.txt"), "rb") as f:
        data_hash.update(f.read())
    meta = {
        "history_rows": int(ds.T),
        "point_rows_matched": int(len(all_periods)),
        "eval_periods": [int(eval_periods[0]), int(eval_periods[-1])],
        "n_eval": len(eval_periods),
        "history_sha256": data_hash.hexdigest()[:16],
        "numpy": np.__version__,
    }
    try:
        import sklearn
        meta["sklearn"] = sklearn.__version__
    except Exception:
        meta["sklearn"] = "unavailable"
    print(f"[data] T={ds.T}, point periods={len(all_periods)}, eval={meta['eval_periods']}")

    results = []
    print("[A] Dirichlet-multinomial ...", flush=True)
    run_exp_a(ds, H, eval_periods, idx, results)
    print("[B] per-number ML walk-forward ...", flush=True)
    run_exp_b(ds, H, eval_periods, idx, results)
    print("[C] point-field constrained triples ...", flush=True)
    rows_c = run_exp_c(ds, eval_periods, results, RNG)
    print("[D] covering design wheel ...", flush=True)
    design, m_tickets, cover_ok = run_exp_d(ds, eval_periods, results, RNG)

    payload = {
        "meta": meta,
        "experiments": results,
        "extras": {
            "C_rows": rows_c,
            "D_design": {
                "n_tickets": int(m_tickets),
                "cover_complete": bool(cover_ok),
                "design": [[int(x) for x in tri] for tri in design],
            },
            "elapsed_s": round(time.time() - t_start, 1),
            "multiple_testing_note": (
                f"共 {len(results)} 项检验；若以 Bonferroni α=0.05 计，单项需 p<{0.05/max(len(results),1):.4f}。"
                "全部候选均为同一数据集上的探索性比较，CI 与 p 未做多重比较校正；"
                "任何单一 5% 显著不应视为可跟进 edge。"),
        },
    }
    out_json = os.path.join(OUT_DIR, "candidates_eval_20261110.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    lines = []
    lines.append("候选方法 walk-forward 评估（最后299个点位期）")
    lines.append("=" * 78)
    for r in results:
        lines.append(f"{r['name']:28s} n={r['n']:4d} mean={r['mean']:.4f} base={r['base']:.4f} "
                     f"lift={r['lift']:.3f} CI[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                     f"p={r['p']:.4g} beats={r['beats_baseline']}")
        lines.append(f"    {r['notes']}")
    lines.append(f"轮盘注数={m_tickets}, 覆盖完整={cover_ok}, 总用时={payload['extras']['elapsed_s']}s")
    out_txt = os.path.join(OUT_DIR, "candidates_eval_20261110.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[saved] {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())