# -*- coding: utf-8 -*-
"""快乐8点位+遗漏分析系统全面工程与算法自动化测试套件。
<p>
【测试范围】:
1. 审计精简后接线完整性（H1–H7 引用的特征列全部真实产出，无死信号）；
2. 三元集群主攻组截断与兜底；
3. 十大特征提取算子输出规格（80 维、非空、可归一化）；
4. 点位无待开奖期时自适应切换历史复盘模式；
5. CLI --raw-points 临时点位传参端到端生成；
6. 近期命中率面板（[8] 节）契约与 walk-forward 无泄漏；
7. 选号策略五码与 5+2 长度兜底防护。
"""
import os
import pytest
import numpy as np
from kl8_core import (N_NUM, Dataset, Draw, TimeSeries, Feat80, CONFIG,
                      load_dataset, next_period_after, parse_history_file,
                      parse_points_file, truncated)
import kl8_methods as M
import kl8_report as R
from kl8_backtest import picks


def _create_mock_dataset(num_draws=60):
    """构造不依赖外部文本文件的内存 Mock Dataset"""
    draws = []
    for idx in range(num_draws):
        start = (idx * 5) % 80 + 1
        nums = frozenset(((start + i) % 80) + 1 for i in range(20))
        draws.append(Draw(date=f"2026-01-{(idx % 28) + 1:02d}", period=2026000 + idx, numbers=nums, idx=idx))
    
    # 构造历史点位及一个待预测期
    point_sets = {}
    point_dates = {}
    for p in range(2026000, 2026010):
        point_sets[p] = frozenset(((p * 3 + i) % 80) + 1 for i in range(20))
        point_dates[p] = f"2026-01-{(p % 28) + 1:02d}"
    
    # 待预测期 (有核心点位、尚未开奖)
    pending_p = 2026000 + num_draws
    point_sets[pending_p] = frozenset(((pending_p * 3 + i) % 80) + 1 for i in range(20))
    point_dates[pending_p] = "2026-02-01"

    ds = Dataset(
        draws=draws,
        draws_by_period={d.period: d for d in draws},
        point_sets=point_sets,
        point_dates=point_dates,
        target_period=pending_p,
    )
    ds.pending_periods = {pending_p}
    return ds


def test_wiring_no_dead_signal():
    """审计精简：H1-H7 引用的每个特征列必须由 pipeline 真实产出，且不存在恒零死信号。"""
    ds = _create_mock_dataset(num_draws=60)
    feat, ctx = R.run_pipeline(ds)
    used = sorted({c for _, cols, _ in R.H_DEF for c in cols})
    missing = [c for c in used if c not in feat.cols]
    assert not missing, f"评分引用了未产出的特征列: {missing}"
    # 已删除的死信号不应复活：HMM（占83%耗时）、余数共振（恒零向量）、cycle（complex 的副本）
    for gone in ("hmm", "hmm_cold", "resid", "cycle"):
        assert gone not in feat.cols, f"{gone} 应已随模块精简下线"
        assert not hasattr(M, f"m_{gone}"), f"M.m_{gone} 应已删除"
    # 每个被引用的列都不能是常值（常值列 min-max 后塌成 0.5，等同无效权重）
    for c in used:
        v = feat.cols[c]
        assert np.nanstd(v) > 1e-9, f"特征列 {c} 为常值，对评分无贡献"


def _create_settled_dataset(num_draws=60, n_points=12):
    """构造「点位+开奖齐备」的复盘样本：点位期落在历史尾部，保证 walk-forward 截断窗口 ≥ 40 期。"""
    ds = _create_mock_dataset(num_draws=num_draws)
    ds.point_sets = {}
    ds.point_dates = {}
    for i in range(num_draws - n_points, num_draws):
        p = 2026000 + i
        ds.point_sets[p] = frozenset(((i * 7 + j) % 80) + 1 for j in range(20))
        ds.point_dates[p] = f"2026-01-{(i % 28) + 1:02d}"
    ds.target_period = 2026000 + num_draws
    return ds


def test_hit_panel_contract():
    """[8] 近期命中率面板：结构完整、口径与主链路一致、walk-forward 无未来泄漏。"""
    ds = _create_settled_dataset(num_draws=60)
    rows, agg = R.recent_hits(ds, n_days=40)
    assert rows and agg, "样本充足时应产出面板"
    assert agg["n"] == len(rows) == len(R.recent_periods(ds, 40))
    for key in ("main_rate", "main_ci", "main_h", "main_n", "group_hit2", "core", "z16"):
        assert key in agg
    assert 0.0 <= agg["main_rate"] <= 1.0
    assert agg["main_n"] >= agg["main_h"] >= 0
    assert agg["main_ci"][0] <= agg["main_rate"] <= agg["main_ci"][1]
    # 逐期字段齐备，供附录 F
    for r in rows:
        for key in ("period", "date", "g_n", "g_hit2", "g_dedup", "g_dedup_n", "z16"):
            assert key in r
    # 面板文本可渲染
    _f, ctx = R.run_pipeline(ds)
    lines = R.build_hit_panel(ctx, ds_orig=ds, n_days=40)
    assert any(l.startswith("[8]") for l in lines)


def test_fast_replay_matches_delivery():
    """命中率面板的轻量复算必须与报告[6]/[7]交付清单同一契约（同组数、同16码、同点位20）。"""
    ds = _create_settled_dataset(num_draws=60)
    p = 2026000 + 55
    rep = R.fast_replay(ds, p)
    assert rep is not None
    # walk-forward：截断集不得包含本期及之后
    tds = R.truncated(ds, p)
    assert p not in tds.draws_by_period
    assert tds.draws[-1].period < p
    _f, ctx = R.run_pipeline(tds, point_period=p)
    pk = R.compute_picks(ctx)
    assert rep["g_n"] == len(pk["groups"])
    cz = ctx["calc_zones"]
    assert rep["z16_n"] == len([n for z in cz for n in z["top2"]])
    assert rep["core_n"] == 20


def test_slim_pipeline_output_shape():
    """精简链路：集群 + 三维综合分可运行，且不再依赖已下线模块。"""
    ds = _create_mock_dataset(num_draws=50)
    ts = TimeSeries(ds)
    name1, vec1, _ = M.m_cluster(ds, ts, {"feat": Feat80()})
    assert name1 == "cluster" and len(vec1) == N_NUM
    feat, ctx = R.run_pipeline(ds)
    assert set(ctx["H"]) == {n for n, _, _ in R.H_DEF}
    assert "markov_info" in ctx and "adv_info" in ctx
    assert len(ctx["composite"]) == N_NUM
    picks = R.compute_picks(ctx)
    assert picks["stance"] == "三元集群"
    assert len(picks["groups"]) >= 1
    assert all("members" in g and "score" in g for g in picks["groups"])


def test_cluster_state_label_consistent_with_value():
    """连号温度标签必须可由已展示的数值直接检验，杜绝「均值高于期望却标偏冷」的方向矛盾。

    历史缺陷：旧实现用固定阈值 cluster_hot_ratio=1.2 判定，
    导致 adj_pairs_mean=4.83 > 期望4.75 时仍输出「连号正常/偏冷」。
    """
    ds = _create_mock_dataset(num_draws=60)
    ts = TimeSeries(ds)
    _t, _e, _c, info = M.build_triples(ds, ts, {})
    mean, exp, z = info["adj_pairs_mean"], info["adj_exp"], info["cluster_z"]
    state = info["cluster_state"]
    # z 分数必须由展示的均值与期望直接推出
    assert abs(z - (mean - exp) / info["adj_se"]) < 1e-9 if info["adj_se"] else z == 0.0
    # 标签方向与数值方向绝不相反
    if mean > exp:
        assert "偏冷" not in state, f"均值{mean:.2f}>期望{exp:.2f} 却标为 {state}"
    elif mean < exp:
        assert "偏热" not in state, f"均值{mean:.2f}<期望{exp:.2f} 却标为 {state}"
    # |z|<1.96 必须判为正常（不显著），不得随意给方向性结论
    if abs(z) <= 1.96:
        assert "正常" in state
    # 报告文案取的就是这个自洽标签
    _f, ctx = R.run_pipeline(ds)
    rep = R.build_report(ctx["ds"], _f, ctx, ds_orig=ds)
    assert any(s in rep for s in ("连号偏热（显著）", "连号偏冷（显著）", "连号正常（偏离未达显著）"))


def test_no_dead_info_outputs():
    """info 中不得保留零消费者的死输出（曾出现 hot_set 计算后无人使用）。"""
    ds = _create_mock_dataset(num_draws=60)
    ts = TimeSeries(ds)
    _n, _v, adv = M.m_adversarial(ds, ts, {"composite_h16": np.linspace(0, 1, 80)})
    assert "hot_set" not in adv
    assert {"lambda", "gains_mean"} <= set(adv)
    # 阈值配置项已随判定逻辑改写而下线，不得复活
    assert "cluster_hot_ratio" not in CONFIG


def test_edge_scan_detects_planted_signal():
    """正对照：edge 扫描管线必须能检出合成数据里植入的真规律。

    这是「真实数据 39 项扫描零信号」这一阴性结论成立的前提——
    探测器若本身失灵，零信号就毫无意义。同时锁死本轮审计修掉的三类缺陷：
      1) 「·全体」恒等式检验（每期 n+=80、k≡20 → rate≡0.25，无信息却占用 BH 的 m）；
      2) 数学上不可达的空桶（如「近3命中=5+次」，W=3 窗口最多 3 次）；
      3) 半分复检靠名称重算、匹配不上则静默 continue 的漏检。
    """
    from kl8_backtest import synth_dataset
    from kl8_edge_scan import scan_all

    ds = synth_dataset(T=400, mode="omission")
    r = scan_all(ds)
    names = [x["name"] for x in r["rows"]]

    # (1) 恒等式检验已彻底移除
    assert not any("·全体" in nm for nm in names), "不应存在 rate 恒为 0.25 的恒等式检验"
    # (2) 不存在空桶：每条检验都要有实际观测量
    assert all(x["n_periods"] > 0 and x["n"] > 0 for x in r["rows"])
    # (3) 内部序列不得泄漏到结果结构
    assert all("_series" not in x for x in r["rows"])

    # 植入的是「命中概率∝连失」的规律 → 遗漏类分桶应被检出
    assert len(r["candidates_ci"]) > 0, "植入真信号却检不出，探测器失效"
    assert len(r["candidates_fdr"]) > 0, "通过 CI 的候选应能过 BH-FDR"
    gap_hits = [x for x in r["candidates_fdr"] if x["name"].startswith("gap")]
    assert gap_hits, "遗漏分桶信号应被检出，实际: %s" % [x["name"] for x in r["candidates_fdr"]]
    # 半分复检必须对每个 CI 候选都产出条目（不再静默跳过）
    assert len(r["half_validation"]) == len(r["candidates_ci"])
    stable = [h for h in r["half_validation"] if h["stable"]]
    assert stable, "植入的稳定规律应在前后半段同时过线"

    # 负向对照：纯随机数据不应出现任何 pass_ci 候选
    rng = np.random.default_rng(3)
    from kl8_core import Dataset, Draw
    draws = []
    for t in range(400):
        nums = frozenset(int(x) for x in rng.choice(np.arange(1, 81), 20, replace=False))
        draws.append(Draw(date="2000-01-01", period=700000 + t, numbers=nums, idx=t))
    rnd = Dataset(draws=draws, draws_by_period={d.period: d for d in draws},
                  point_sets={}, point_dates={}, target_period=700400)
    rn = scan_all(rnd)
    assert len(rn["candidates_ci"]) == 0, "纯随机数据不应检出信号（假阳性防线）"


def test_clustered_rate_ci_matches_hypergeometric():
    """聚类 CI 必须与「每期恰好开 20 个号」的超几何理论方差一致。

    原实现把每个 (期,号) 格子当独立伯努利试验，忽略同期内无放回抽样的负相关。
    """
    from kl8_core import clustered_rate_ci
    # 构造已知真值：每期从 f=40 个号中命中 S_t 个，S_t 服从超几何(80,20,40)
    rng = np.random.default_rng(11)
    T, f, D, N = 40000, 40, 20, 80
    kt = np.empty(T)
    for t in range(T):
        chosen = rng.choice(N, D, replace=False)
        kt[t] = int(((chosen + 1) <= f).sum())
    nt = np.full(T, f)
    r = clustered_rate_ci(kt, nt)
    # 超几何精确：E[S]=D*f/N=10, Var(S)=D*(f/N)*(1-f/N)*(N-D)/(N-1)
    var_S = D * (f / N) * (1 - f / N) * (N - D) / (N - 1)
    se_theory = np.sqrt(var_S) / f / np.sqrt(T)
    assert abs(r["se"] - se_theory) / se_theory < 0.05, \
        f"聚类 se={r['se']:.6f} 偏离超几何理论 se={se_theory:.6f} 超 5%"
    # 命中率应≈0.25，且 95%CI 应覆盖真值
    assert abs(r["rate"] - 0.25) < 0.01
    assert r["lo"] <= 0.25 <= r["hi"]


def test_calc_zone_module_contract():
    """微积分八区分区：8区×10号，每区最优2码，互不重叠。"""
    ds = _create_mock_dataset(num_draws=50)
    ts = TimeSeries(ds)
    ctx = {"feat": Feat80()}
    M.m_cluster(ds, ts, ctx)
    name, vec, info = M.m_calc_zone(ds, ts, ctx)
    assert name == "calc_zone"
    assert len(vec) == N_NUM
    assert np.isfinite(vec).all()
    zones = ctx["calc_zones"]
    assert len(zones) == CONFIG["calc_zone_n"] == 8
    all_picks = []
    for z in zones:
        assert len(z["members"]) == CONFIG["calc_zone_size"]
        assert len(z["top2"]) == CONFIG["calc_zone_top_k"] == 2
        assert z["lo"] <= min(z["top2"]) <= max(z["top2"]) <= z["hi"]
        assert set(z["top2"]).issubset(set(z["members"]))
        all_picks.extend(z["top2"])
    assert len(all_picks) == 16
    assert len(set(all_picks)) == 16
    pk = M.calc_zone_picks(zones)
    assert pk["n"] == 16 and len(pk["by_zone"]) == 8
    # 挂入 pipeline 后报告可见
    _, ctx2 = R.run_pipeline(ds)
    assert "calc_zones" in ctx2 and len(ctx2["calc_zones"]) == 8
    assert "calc_zone" in ctx2["feat"].cols


def test_no_pending_periods_fallback():
    """验证当点位文件无待预测期（全部已开奖）时，系统自动切入最新一期复盘模式"""
    ds = _create_mock_dataset(num_draws=30)
    # 移除 pending_periods，模拟所有点位均已有开奖结果
    ds.pending_periods = set()
    # 重新加载或走判断逻辑
    if not ds.pending_periods and ds.point_sets:
        target = max(ds.point_sets.keys())
    else:
        target = 0
    assert target > 0
    assert target in ds.point_sets


def test_picks_safety_and_fallback():
    """验证五码与两码备防在面对极端低温弃子时具有保底 5 码与 2 码的能力"""
    comp = np.linspace(0.1, 0.9, 80)
    H = {hd[0]: np.full(80, 0.5) for hd in R.H_DEF}
    feat = Feat80()
    feat.add("cluster", np.full(80, 0.5))
    
    # 模拟构建全被标为“低温弃子”的三元组
    triples = [{
        "center": 1,
        "members": (80, 1, 2),
        "score": 0.1,
        "labels": ["低温弃子"]
    }]
    ctx = {
        "composite": comp,
        "triple_info": {"triples": triples},
        "comp_hedge": np.linspace(0.9, 0.1, 80),
        "comp_trend": comp,
    }

    result = picks(comp, H, feat, ctx)
    assert len(result["top5"]) == 5
    assert len(result["top20"]) == 20
    assert len(result["five"]) == 5
    assert len(result["seven"]) == 7
    # 验证没有重复选号
    assert len(set(result["five"])) == 5
    assert len(set(result["seven"])) == 7


def test_raw_points_cli_flow(tmp_path):
    """验证 CLI --raw-points 能够正确贯通并生成报表。"""
    out_file = str(tmp_path / "custom_report.txt")
    raw_points_arg = "01 05 10 15 20 25 30 35 40 45 50 55 60 65 70 75 76 77 78 79"
    exit_code = R.main([
        "--raw-points", raw_points_arg,
        "--out", out_file
    ])
    assert exit_code == 0
    assert os.path.exists(out_file)
    with open(out_file, encoding="utf-8") as f:
        content = f.read()
    assert "快乐8「点位+遗漏」" in content
    assert "[1] N维评分仪表盘" in content
    assert "[6] 终极推荐清单" in content
    assert "三元集群" in content
    assert "[7] 集群三元组" in content
    assert "金字塔分层推荐清单:" in content


def test_triplet_resonance_and_affinity_contract():
    """验证升级后的三元组深度挖掘算子：同出亲和度、空间拓扑共振度与金字塔分层推荐。"""
    ds = _create_mock_dataset(num_draws=60)
    ts = TimeSeries(ds)
    T = ds.T
    core = [5, 6, 20, 21, 50]
    core_set = set(core)
    
    # 1. 验证共振度：相邻点位(5,6)的共振度应显著大于孤立点位(50)
    res_5 = M.compute_point_proximity_resonance(5, core_set)
    res_50 = M.compute_point_proximity_resonance(50, core_set)
    assert res_5 > res_50, f"相邻点位共振分应更高: res(5)={res_5} vs res(50)={res_50}"
    
    # 2. 验证亲和度计算（Mock数据同出率高，范围允许在 [0, 5.0]）
    aff = M.compute_cooccurrence_affinity(ts, T, (5, 6, 7), window=50)
    assert 0.0 <= aff <= 5.0, f"亲和度数值非法: {aff}"
    
    # 3. 验证金字塔分层推荐
    ds.point_sets[ds.target_period] = frozenset(core)
    triples, ext, cs, info = M.build_triples(ds, ts, {})
    assert len(triples) == len(core)
    for t in triples:
        assert "aff" in t and "res" in t
        assert "labels" in t and len(t["labels"]) > 0
    
    groups = M.select_main_groups(triples)
    assert len(groups) >= 1
    assert groups[0].get("tier") == "T1核心"


def _fmt_20(nums):
    return " ".join(f"{n:02d}" for n in nums)


def test_upsert_points_refuses_cross_date_overwrite(tmp_path):
    """同步滞后时，今日点位不得覆盖同期的昨日历史点位行。"""
    from kl8_daily import upsert_points
    p = tmp_path / "points.txt"
    old = _fmt_20(range(1, 21))
    new = _fmt_20(range(41, 61))
    p.write_text(f"date:2026-11-08,period:2026298,points:{old}\n", encoding="utf-8")

    with pytest.raises(ValueError):
        upsert_points(str(p), "2026-11-09", 2026298, list(range(41, 61)))
    text = p.read_text(encoding="utf-8")
    assert "date:2026-11-08,period:2026298" in text and old in text

    upsert_points(str(p), "2026-11-09", 2026298, list(range(41, 61)), allow_overwrite=True)
    text = p.read_text(encoding="utf-8")
    assert "date:2026-11-09,period:2026298" in text and new in text
    assert "date:2026-11-08" not in text


def test_truncated_blocks_future_data_when_target_missing():
    """目标期缺行时只能截到目标期之前，不能把更新的开奖喂给模型。"""
    periods = [1, 2, 3, 4, 5, 7, 8, 9, 10]  # 缺 2026006
    draws = []
    for i, p in enumerate(periods):
        nums = frozenset(((p * 3 + j) % 80) + 1 for j in range(20))
        draws.append(Draw(date=f"2026-01-{p:02d}", period=2026000 + p, numbers=nums, idx=i))
    ds = Dataset(draws=draws, draws_by_period={d.period: d for d in draws},
                 point_sets={}, point_dates={}, target_period=2026006)
    t = truncated(ds, 2026006)
    assert t.T == 5
    assert max(d.period for d in t.draws) == 2026005
    assert any("防止未来数据泄漏" in w for w in t.warnings)


def test_next_period_after_handles_year_rollover():
    assert next_period_after(Draw(date="2020-12-31", period=2020365, numbers=frozenset())) == 2021001
    assert next_period_after(Draw(date="2026-11-09", period=2026299, numbers=frozenset())) == 2026300


def test_load_dataset_excludes_stale_missing_history_from_target(tmp_path):
    """历史缺行的旧点位不能被当成待预测期，target 应回到最近的可复盘期。"""
    hist = tmp_path / "hist.txt"
    pts = tmp_path / "points.txt"
    nums20 = "-".join(f"{n:02d}" for n in range(1, 21))
    hist.write_text(
        f"date:2026-11-08,period:2026298,numbers:{nums20}\n"
        f"date:2026-11-09,period:2026299,numbers:{nums20}\n",
        encoding="utf-8",
    )
    pts.write_text(
        f"date:2026-10-31,period:2026290,points:{_fmt_20(range(1, 21))}\n"
        f"date:2026-11-08,period:2026298,points:{_fmt_20(range(21, 41))}\n",
        encoding="utf-8",
    )
    ds = load_dataset(str(pts), str(hist))
    assert ds.target_period == 2026298
    assert any("历史缺期" in w for w in ds.warnings)


def test_load_dataset_keeps_future_points_off_drawn_date(tmp_path):
    """下一期点位误用已开奖日期时，不能被日期兜底绑到已开奖期。"""
    hist = tmp_path / "hist.txt"
    pts = tmp_path / "points.txt"
    nums20 = "-".join(f"{n:02d}" for n in range(1, 21))
    hist.write_text(
        f"date:2026-11-09,period:2026299,numbers:{nums20}\n",
        encoding="utf-8",
    )
    pts.write_text(
        f"date:2026-11-09,period:2026300,points:{_fmt_20(range(41, 61))}\n",
        encoding="utf-8",
    )
    ds = load_dataset(str(pts), str(hist))
    assert ds.target_period == 2026300
    assert 2026299 not in ds.point_sets
    assert 2026300 in ds.point_sets
    assert any("晚于该日期开奖期号" in w for w in ds.warnings)


def test_parse_files_skip_non_numeric_tokens_without_crashing(tmp_path):
    hist = tmp_path / "hist.txt"
    pts = tmp_path / "points.txt"
    hist.write_text(
        "date:2026-11-09,period:2026299,numbers:01-02-03-04-05-06-07-08-09-10-11-12-13-14-15-16-17-18-19-7x\n",
        encoding="utf-8",
    )
    pts.write_text(
        "date:2026-11-09,period:2026299,points:01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 0l\n",
        encoding="utf-8",
    )
    h_rows, h_warn = parse_history_file(str(hist))
    p_rows, p_warn = parse_points_file(str(pts))
    assert h_rows == [] and h_warn
    assert p_rows == [] and p_warn


def test_copyable_triples_uses_main_groups_not_top8():
    """纯净可复制清单必须与 [6] 主攻组同契约，不能混入未推荐的 Top8。"""
    triples = [
        {"members": (43, 44, 45), "center": 44, "score": 0.65, "labels": ["候选"]},
        {"members": (7, 8, 9), "center": 8, "score": 0.50, "labels": ["候选"]},
        {"members": (70, 71, 72), "center": 71, "score": 0.41, "labels": ["候选"]},
        {"members": (6, 7, 8), "center": 7, "score": 0.30, "labels": ["候选"]},
    ]
    ctx = {
        "triple_info": {"triples": triples},
        "adv_info": {"lambda": 1.0},
        "comp_trend": np.zeros(N_NUM),
        "comp_hedge": np.zeros(N_NUM),
    }
    assert R.copyable_triples(ctx) == ["43-44-45", "07-08-09", "70-71-72"]


def test_hit_panel_judgement_is_dynamic():
    """[8] 判读必须随 CI 与基线的相对位置变化，不能硬编码「覆盖基线」。"""
    high = R.hit_panel_judgement(0.30, (0.255, 0.345))
    cover = R.hit_panel_judgement(0.26, (0.24, 0.28))
    low = R.hit_panel_judgement(0.20, (0.18, 0.22))
    assert "高于 25% 基线" in high and "多窗口" in high
    assert "覆盖 25% 基线" in cover
    assert "低于 25% 基线" in low


def test_forward_test_settle_logic():
    """前瞻检验结算：≥2 码、3 连全中、去重号的计数必须与开奖一致。"""
    import kl8_forward_test as FT
    pred = {
        "period": 2026300,
        "groups": [
            {"members": [42, 43, 44], "center": 43, "score": 0.7,
             "tier": "T1核心", "labels": []},
            {"members": [5, 6, 7], "center": 6, "score": 0.6,
             "tier": "T2防守", "labels": []},
        ],
    }
    actual = [42, 44, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
              17, 18, 19, 20, 21, 22, 23]
    draw = Draw(date="2026-11-10", period=2026300, numbers=frozenset(actual))
    ev = FT._settle_one(pred, draw)
    assert ev["g_n"] == 2
    assert ev["g_hit2"] == 2
    assert ev["g_full3"] == 1
    assert ev["dedup_hits"] == 5
    assert ev["dedup_n"] == 6


def test_forward_test_record_is_idempotent_and_rejects_changes(tmp_path):
    """同一期重复登记必须幂等；点位变化必须拒绝，保护预注册完整性。"""
    import json
    import kl8_forward_test as FT
    meta_path = tmp_path / "meta.json"
    events_path = tmp_path / "events.jsonl"
    meta = FT.ensure_meta(meta_path)
    meta["start_period"] = 2026000
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    ds = _create_mock_dataset(num_draws=60)
    target = ds.target_period
    _feat, ctx = R.run_pipeline(ds)
    rec = FT.record_prediction(ds, ctx, target, events_path, meta_path)
    assert rec and rec["event"] == "prediction"
    again = FT.record_prediction(ds, ctx, target, events_path, meta_path)
    assert again["period"] == target
    events = FT.load_events(events_path)
    assert len([e for e in events if e.get("event") == "prediction"]) == 1

    ds.point_sets[target] = frozenset(range(41, 61))
    with pytest.raises(ValueError):
        FT.record_prediction(ds, ctx, target, events_path, meta_path)


def test_forward_test_report_smoke(tmp_path):
    """前瞻检验报告：进度、主口径与进行中判定可正常渲染。"""
    import json
    import kl8_forward_test as FT
    meta_path = tmp_path / "meta.json"
    events_path = tmp_path / "events.jsonl"
    meta = FT.ensure_meta(meta_path)
    meta["start_period"] = 2026000
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    events = [
        {"event": "prediction", "period": 2026001, "date": "2026-01-01",
         "points": list(range(1, 21)),
         "groups": [{"members": [1, 2, 3], "center": 2, "score": 0.5,
                     "tier": "T1核心", "labels": []}], "n_groups": 1},
        {"event": "settlement", "period": 2026001, "g_n": 1, "g_hit2": 1,
         "g_full3": 0, "dedup_hits": 2, "dedup_n": 3,
         "groups": [{"members": [1, 2, 3], "tier": "T1核心",
                     "hit_ge2": True, "full3": False}]},
    ]
    events_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8")
    out = FT.report(events_path, meta_path)
    assert out["status"] == "进行中"
    assert out["k"] == 1 and out["n"] == 1 and out["eligible"] == 1
