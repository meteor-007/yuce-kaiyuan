# -*- coding: utf-8 -*-
"""快乐8点位+遗漏分析 —— 数据层与特征层"""
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

N_NUM = 80
POINTS_FILE = "daily_points.txt"
HISTORY_FILE = "kl8_history_final.txt"

# 产出目录：报告 / 回测缓存与根目录代码、数据源分离，避免根目录膨胀
OUTPUT_DIR = "output"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"
BACKTEST_DIR = f"{OUTPUT_DIR}/backtest"

CONFIG = {
    "points_file": POINTS_FILE,
    "history_file": HISTORY_FILE,
    "output_dir": OUTPUT_DIR,
    "reports_dir": REPORTS_DIR,
    "backtest_dir": BACKTEST_DIR,
    "backtest_result": f"{BACKTEST_DIR}/kl8_backtest_result.json",
    "backtest_stash": f"{BACKTEST_DIR}/kl8_H_stash.npz",
    "window_short": 10,
    "window_burst": 3,
    "window_run": 30,
    "window_main": 100,
    "window_markov": 200,
    "window_cooccur": 200,
    "window_complex": 100,
    "window_graph": 200,
    "window_vol": 20,
    "gap_cap": 12,
    "top_n": 5,
    "follow_top": 20,
    "markov_follow_top": 30,
    # 微积分·八区分区：80号→8区×10号，每区取优2
    "calc_zone_n": 8,
    "calc_zone_size": 10,
    "calc_zone_top_k": 2,
    "window_calc": 40,
    "window_calc_short": 10,
    "window_calc_mid": 20,
}


def ensure_output_dirs() -> None:
    """确保 output/reports、output/backtest 存在。"""
    Path(CONFIG["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["backtest_dir"]).mkdir(parents=True, exist_ok=True)


def report_paths(date_key: str) -> tuple[str, str]:
    """返回 (标准报告路径, 详细报告路径)。"""
    ensure_output_dirs()
    base = Path(CONFIG["reports_dir"])
    return (
        str(base / f"分析报告_{date_key}.txt"),
        str(base / f"分析报告_{date_key}_详细.txt"),
    )


def backtest_result_path() -> str:
    ensure_output_dirs()
    return CONFIG["backtest_result"]


def backtest_stash_path() -> str:
    ensure_output_dirs()
    return CONFIG["backtest_stash"]


def resolve_existing(*candidates: str) -> str | None:
    """按候选顺序返回首个已存在路径（兼容旧根目录产物）。"""
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None

_RING = np.arange(1, N_NUM + 1)


def ring_prev(x):
    return ((x - 2) % N_NUM) + 1


def ring_next(x):
    return (x % N_NUM) + 1


def ring_dist(a, b):
    d = (a - b) % N_NUM
    return min(d, N_NUM - d)


def ring_window(center, r):
    return frozenset(((center - 1 + k) % N_NUM) + 1 for k in range(-r, r + 1))


@dataclass
class Draw:
    date: str
    period: int
    numbers: frozenset
    idx: int = -1


@dataclass
class Dataset:
    draws: list = field(default_factory=list)            # 旧→新
    draws_by_period: dict = field(default_factory=dict)
    point_sets: dict = field(default_factory=dict)       # period -> frozenset20
    point_dates: dict = field(default_factory=dict)      # period -> date
    target_period: int = 0                               # 待预测期（有点位无开奖的最新一期）
    warnings: list = field(default_factory=list)
    pseudo_point_periods: set = field(default_factory=set)  # 回测用伪点位期

    @property
    def T(self):
        return len(self.draws)

    def draw_matrix(self):
        H = np.zeros((self.T, N_NUM))
        for i, d in enumerate(self.draws):
            for n in d.numbers:
                H[i, n - 1] = 1.0
        return H


def parse_points_file(path):
    rows = []
    warnings = []
    pat = re.compile(r"date:(\d{4}-\d{2}-\d{2}),period:(\d+),points:(.+)")
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                warnings.append(f"points文件第{ln}行格式无法解析，跳过")
                continue
            date, period, pts = m.groups()
            try:
                nums = [int(x) for x in pts.replace(",", " ").replace("-", " ").split()]
                assert len(nums) == 20 and len(set(nums)) == 20
                assert all(1 <= x <= N_NUM for x in nums)
            except (AssertionError, ValueError):
                warnings.append(f"points文件第{ln}行号码校验失败，跳过")
                continue
            rows.append((date, int(period), frozenset(nums)))
    return rows, warnings


def parse_history_file(path):
    rows = []
    warnings = []
    pat = re.compile(r"date:(\d{4}-\d{2}-\d{2}),period:(\d+),numbers:(.+)")
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            m = pat.match(line)
            if not m:
                warnings.append(f"history文件第{ln}行格式无法解析，跳过")
                continue
            date, period, nums_s = m.groups()
            try:
                nums = [int(x) for x in nums_s.replace(",", "-").split("-") if x.strip()]
                assert len(nums) == 20 and len(set(nums)) == 20
                assert all(1 <= x <= N_NUM for x in nums)
            except (AssertionError, ValueError):
                warnings.append(f"history文件第{ln}行号码校验失败，跳过")
                continue
            rows.append((date, int(period), frozenset(nums)))
    return rows, warnings


def next_period_after(draw):
    """下一期期号：期号为 YYYYNNN；12-31 的最后一期后归零到次年 001。"""
    if draw is None:
        return 0
    try:
        y, m, d = (int(x) for x in str(draw.date).split("-"))
    except Exception:
        return int(draw.period) + 1
    if (m, d) == (12, 31):
        return (y + 1) * 1000 + 1
    return int(draw.period) + 1


def next_target_date(ds):
    """目标期默认日期：最新开奖日期的次日；无历史时返回 None。"""
    import datetime as _dt
    if not ds.draws:
        return None
    try:
        last = _dt.date.fromisoformat(ds.draws[-1].date)
    except Exception:
        return None
    return (last + _dt.timedelta(days=1)).strftime("%Y-%m-%d")


def load_dataset(points_path=POINTS_FILE, history_path=HISTORY_FILE):
    """加载开奖历史与核心点位数据，构建核心 Dataset 数据集实体。
    <p>
    【业务背景与架构定位】:
    读取开奖历史与核心点位两份数据文件，执行日期/期号双主键对齐校验；
    识别待开奖预测期（有点位无开奖），若无待预测期则自适应切入最新历史期复盘模式，杜绝空指针与越界。
    
    @param points_path  点位文件路径，默认为 daily_points.txt
    @param history_path 开奖历史文件路径，默认为 kl8_history_final.txt
    @return Dataset 完整数据集对象
    """
    # 步骤 1: 解析点位与开奖历史原始文本
    point_rows, w1 = parse_points_file(points_path)
    draw_rows, w2 = parse_history_file(history_path)
    warnings = w1 + w2

    # 步骤 2: 开奖历史按日期与期号升序排序 (旧 -> 新)
    draws = [Draw(date=d, period=p, numbers=ns) for d, p, ns in draw_rows]
    draws.sort(key=lambda x: (x.date, x.period))
    for i, d in enumerate(draws):
        d.idx = i
    draws_by_period = {d.period: d for d in draws}
    date_occupied = {}
    for d in draws:
        date_occupied.setdefault(d.date, set()).add(d.period)

    # 步骤 3: 匹配点位期号与日期映射关系，提取待预测期
    latest_draw_period = draws[-1].period if draws else 0
    point_sets = {}
    point_dates = {}
    pending_periods = set()
    matched_dates = set()
    for date, period, nums in point_rows:
        if period in draws_by_period:
            point_sets[period] = nums
            point_dates[period] = date
            matched_dates.add(date)
            continue
        cands = date_occupied.get(date, set())
        if len(cands) == 1:
            real_p = next(iter(cands))
            if period < real_p:
                if date in matched_dates:
                    warnings.append(f"孤儿点位行 period:{period}（{date}）：该日期已被其他期号占用，丢弃")
                    continue
                # 期号写旧了：按日期兜底修正到当日真实期号
                warnings.append(f"点位期号{period}（{date}）在开奖中不存在，按日期兜底匹配到期号{real_p}")
                point_sets[real_p] = nums
                point_dates[real_p] = date
                matched_dates.add(date)
            elif period > real_p:
                # 下一期的点位误用了已开奖期的日期：保留期号，但明确告警日期异常
                point_sets[period] = nums
                point_dates[period] = date
                if period > latest_draw_period:
                    pending_periods.add(period)
                    warnings.append(
                        f"点位期号{period}（{date}）晚于该日期开奖期号{real_p}；"
                        f"已按待预测期处理，建议用 --date 指定目标开奖日"
                    )
                else:
                    warnings.append(
                        f"历史缺期：点位期号{period}（{date}）在开奖历史中缺失，"
                        f"已排除出待预测期；请补全历史记录"
                    )
            else:
                warnings.append(f"孤儿点位行 period:{period}（{date}）：该日期已被其他期号占用，丢弃")
        elif len(cands) == 0:
            point_sets[period] = nums
            point_dates[period] = date
            if period > latest_draw_period:
                # 该日期无开奖记录且期号晚于最新开奖 → 待预测期
                pending_periods.add(period)
            else:
                warnings.append(
                    f"历史缺期：点位期号{period}（{date}）在开奖历史中缺失，"
                    f"已排除出待预测期；请补全历史记录"
                )
        else:
            warnings.append(f"孤儿点位行 period:{period}（{date}）：该日期已被其他期号占用，丢弃")

    # 步骤 4: 确定目标预测期
    if pending_periods:
        target_period = max(pending_periods)
    else:
        # 只从「已开奖且有点位」的期中选复盘目标，避免历史缺行使目标长期停在旧期
        review_periods = [p for p in point_sets if p in draws_by_period]
        if review_periods:
            target_period = max(review_periods)
            warnings.append(f"未检测到未开奖预测期，自适应切入最新历史点位期 {target_period} 进行复盘研判")
        elif draws:
            target_period = next_period_after(draws[-1])
            warnings.append(
                f"未检测到未开奖预测期，也未找到可复盘的历史点位，"
                f"目标期取最新开奖期的下一期 {target_period}（无点位，无法产出主攻清单）"
            )
        else:
            target_period = 0
            warnings.append("未匹配到任何有效点位数据")

    ds = Dataset(draws=draws, draws_by_period=draws_by_period,
                 point_sets=point_sets, point_dates=point_dates,
                 target_period=target_period, warnings=warnings)
    ds.pending_periods = pending_periods
    return ds


def truncated(ds, target_period):
    """返回历史截断到 target_period 之前的 Dataset（防泄漏），点位集沿用全量。

    target 不在历史中时也必须按期号截断：若历史里存在比目标期更新的开奖，
    直接返回全量会把未来数据泄漏进推荐。
    """
    warnings = list(ds.warnings)
    if target_period in ds.draws_by_period:
        t = ds.draws_by_period[target_period].idx
        draws = ds.draws[:t]
    else:
        try:
            tp = int(target_period)
            draws = [d for d in ds.draws if int(d.period) < tp]
            newer = [d for d in ds.draws if int(d.period) >= tp]
        except (TypeError, ValueError):
            draws = list(ds.draws)
            newer = []
        if newer:
            warnings.append(
                f"目标期 {target_period} 不在开奖历史中，但存在 {len(newer)} 期更新的开奖"
                f"（最新 {newer[-1].period} / {newer[-1].date}）；已按期号截断，防止未来数据泄漏。"
            )
    nd = Dataset(draws=draws,
                 draws_by_period={d.period: d for d in draws},
                 point_sets=dict(ds.point_sets),
                 point_dates=dict(ds.point_dates),
                 target_period=target_period,
                 warnings=warnings)
    if target_period not in nd.point_sets:
        nd.pending_periods = {target_period}
    return nd


class TimeSeries:
    """每号命中时序与派生量"""

    def __init__(self, ds: Dataset):
        self.H = ds.draw_matrix()
        self.hit = self.H
        T = ds.T
        gap = np.zeros((T, N_NUM))
        if T > 0:
            gap[0] = np.where(self.hit[0] > 0, 0, 1)
            for i in range(1, T):
                gap[i] = np.where(self.hit[i] > 0, 0, gap[i - 1] + 1)
        self.gap = gap
        self.Hcum = self.H.cumsum(axis=0)

    def hits_in(self, n, lo, hi):
        """号码 n（1-80）在 [lo, hi) 期内的命中次数"""
        lo = max(lo, 0)
        hi = min(hi, self.H.shape[0])
        if hi <= lo:
            return 0
        return self.Hcum[hi - 1, n - 1] - (self.Hcum[lo - 1, n - 1] if lo > 0 else 0.0)

    def hits_window(self, n, T, w):
        return self.hits_in(n, T - w, T)


class Feat80:
    """80维特征列容器：raw + 跨号min-max归一化"""

    def __init__(self):
        self.cols = {}
        self.norm = {}

    def add(self, name, vec):
        vec = np.asarray(vec, dtype=float).copy()
        self.cols[name] = vec
        self.norm[name] = minmax(vec)
        return vec

    def get(self, name):
        return self.cols[name]

    def get_norm(self, name):
        return self.norm[name]


def clustered_rate_ci(kt, nt, p0: float = 0.25, z: float = 1.96):
    """以「期」为聚类单元的比率估计与 95%CI。

    【为什么不能用二项/Wilson 独立口径】
    快乐8每期恰好开出 20 个号（无放回超几何抽样），同一期内各号码格子负相关，
    把每个 (期, 号) 格子当独立伯努利试验会误估不确定度（方向不固定：
    纯静态区位组只剩组内负相关→naive 偏宽；实际主攻组还叠加逐期组数/质量的
    日间异质→naive 偏窄）。以「期」为聚类单元可同时捕获两者。
        R = ΣS_t / Σf_t,   Var(R) = Σ(S_t − R·f_t)² / [(T−1)·f̄²·T]
    验证锚点：f=40 的奇数族，每期命中均值恒为 10（范围约 4–17，非零方差），
    本式给出 se≈0.00106，与超几何理论 se=0.00106 一致。

    @param kt 每期命中数序列
    @param nt 每期组内号码数序列
    @return dict(rate, lo, hi, se, k, n, n_periods)
    """
    kt = np.asarray(kt, dtype=float)
    nt = np.asarray(nt, dtype=float)
    keep = nt > 0
    kt, nt = kt[keep], nt[keep]
    T_ = int(len(kt))
    k, n = float(kt.sum()), float(nt.sum())
    rate = k / n if n else 0.0
    if T_ > 1:
        e = kt - rate * nt
        var = float((e ** 2).sum() / ((T_ - 1) * (float(nt.mean()) ** 2) * T_))
        se = float(np.sqrt(max(var, 0.0)))
    else:
        se = float(np.sqrt(max(rate * (1 - rate), 1e-12) / n)) if n else 1.0
    return {"rate": float(rate), "lo": float(rate - z * se), "hi": float(rate + z * se),
            "se": se, "k": int(k), "n": int(n), "n_periods": T_}


def minmax(vec):
    v = np.asarray(vec, dtype=float)
    out = np.full_like(v, 0.5)
    mask = ~np.isnan(v)
    if mask.sum() == 0:
        return out
    vmin, vmax = v[mask].min(), v[mask].max()
    if vmax - vmin < 1e-12:
        out[mask] = 0.5
    else:
        out[mask] = (v[mask] - vmin) / (vmax - vmin)
    return out


def zscore(vec):
    v = np.asarray(vec, dtype=float)
    out = np.full_like(v, 0.0)
    mask = ~np.isnan(v)
    if mask.sum() < 2:
        return out
    mu, sd = v[mask].mean(), v[mask].std()
    if sd < 1e-12:
        out[mask] = 0.0
    else:
        out[mask] = (v[mask] - mu) / sd
    return out


def disp_width(s):
    w = 0
    for ch in s:
        w += 2 if ord(ch) > 127 else 1
    return w


def pad(s, width):
    return s + " " * max(0, width - disp_width(s))
