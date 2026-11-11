# -*- coding: utf-8 -*-
"""日常一键：同步开奖 →（可选）录入点位 → 昨日深度复盘 + 全模块预测 + 命中率面板。

控制台默认打印完整标准报告（[0]–[8] 全节 + 附录），
详细版附录（A–F：全三元组子分、Top20 原始矩阵、扇区能量、八区全量、近期逐期明细）落盘 output/reports/。
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from kl8_core import (CONFIG, N_NUM, load_dataset, next_period_after,
                      next_target_date, pad, report_paths, truncated)
from kl8_sync import sync as sync_history


POINTS_FILE = CONFIG.get("points_file", "daily_points.txt")
HISTORY_FILE = CONFIG.get("history_file", "kl8_history_final.txt")
BAR = "═" * 78


def _ensure_utf8_console():
    """Windows 控制台尽量切到 UTF-8，避免中文乱码。"""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass


def parse_points(raw: str) -> list[int]:
    s = raw.replace("-", " ").replace(",", " ")
    nums = [int(x) for x in s.split() if x.strip()]
    if len(nums) != 20 or len(set(nums)) != 20 or not all(1 <= x <= N_NUM for x in nums):
        raise ValueError(f"需要 1-80 不重复共 20 个号码，当前 {len(nums)} 个")
    return sorted(nums)


def upsert_points(path: str, date: str, period: int, nums: list[int],
                  allow_overwrite: bool = False) -> None:
    """按期号写入/更新点位文件（最新期在文件顶部）。

    同期号已有不同日期的历史行时拒绝静默覆盖：sync 滞后时今天的点位
    会被误算成昨天期号，整行覆盖昨日真实预测。显式 allow_overwrite 才允许。
    """
    line = f"date:{date},period:{period},points:" + " ".join(f"{n:02d}" for n in nums)
    p = Path(path)
    rows = []
    existing_dates = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            m = re.match(r"date:([^,]+),period:(\d+),points:(.+)", ln)
            if m and int(m.group(2)) == period:
                existing_dates.append(m.group(1))
                continue
            rows.append(ln)
    other_dates = sorted({d for d in existing_dates if d != date})
    if other_dates and not allow_overwrite:
        raise ValueError(
            f"点位期 {period} 已存在日期 {', '.join(other_dates)}，本次日期 {date} 不同；"
            f"为防止覆盖历史预测已拒绝写入。确认覆盖请加 --force。"
        )
    rows.insert(0, line)
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")


def print_banner(title: str):
    print()
    print(BAR)
    print(f"  {title}")
    print(BAR)


def step(no: int, total: int, title: str):
    print()
    print(f"▶ 步骤 {no}/{total}  {title}")
    print("-" * 78)


def module_readout(feat, ctx, ds):
    """逐模块明细读出：每个预测模块的关键中间产出（控制台美观输出用）。"""
    from collections import Counter
    import numpy as np
    from kl8_report import top_idx

    def row(no, name, detail):
        print(f"   {no} {pad(name, 18)} {detail}")

    ti = ctx["triple_info"]
    import kl8_report as R
    picks = R.compute_picks(ctx)
    groups = picks.get("groups") or []
    head = ti["triples"][0] if ti["triples"] else None
    row("①", "连号集群三元组",
        f"{len(ti['triples'])} 组｜主攻 {len(groups)} 组｜头组 "
        + (f"{'-'.join('%02d' % m for m in head['members'])}(场分{head['score']:.2f})" if head else "-")
        + f"｜连号温度 {ti['adj_pairs_mean']:.2f}/期望{ti.get('adj_exp', 4.75):.2f} "
        f"z={ti.get('cluster_z', 0.0):+.2f} {ti.get('cluster_state', '')}")

    mi = ctx["markov_info"]
    p01 = np.nan_to_num(mi["P01"])
    row("②", "马尔可夫转移",
        f"窗口{min(CONFIG['window_markov'], ds.T)}期｜P01 Top5: "
        + " ".join("%02d" % n for n in top_idx(p01, 5))
        + f"｜样本不足 {mi['low_conf']} 号")

    fi = ctx["fib_info"]
    seg = ["1-30", "31-54", "55-80"][fi["phi_bias_seg"]]
    fv = np.nan_to_num(feat.cols["fib"])
    row("③", "斐波那契/黄金分割",
        f"偏向段 {seg}（{fi['phi_bias_dev']*100:+.1f}%）｜Fib热度Top5: "
        + " ".join("%02d" % n for n in top_idx(fv, 5)))

    cp = np.nan_to_num(feat.cols["cp"])
    sv = np.nan_to_num(feat.cols["svd"])
    row("④", "SVD+共现矩阵",
        "共现最强: " + " ".join("%02d" % n for n in top_idx(cp, 5))
        + "｜首奇异方向: " + " ".join("%02d" % n for n in top_idx(sv, 5)))

    ci = ctx["complex_info"]
    row("⑤", "复变傅里叶周期",
        "主导周期: " + "、".join(f"{p}期×{c}" for p, c in ci["dominant_periods"]))

    gi = ctx["graph_info"]
    row("⑥", "图网络共现度",
        f"边数 {gi['n_edges']}｜最拥挤: "
        + " ".join("%02d" % n for n in top_idx(np.nan_to_num(gi["crowding"]), 5)))

    vi = ctx["vol_info"]
    cnt = Counter(int(x) for x in vi["labels"])
    row("⑦", "波动率聚类",
        f"冷稳{cnt.get(0,0)}/间歇{cnt.get(1,0)}/高频震荡{cnt.get(2,0)}｜"
        f"Jaccard 近{vi['j10']:.3f} vs 长{vi['j50']:.3f} → {vi['j_state']}")

    ei = ctx["entropy_info"]
    z = ei["entropy_anomaly"]
    row("⑧", "熵值均衡监控",
        f"最新期分布熵 z={z:+.2f}" + ("（异常，聚集/离散异动）" if abs(z) > 2 else "（未达异常阈值±2）"))

    czp = ctx.get("calc_zone_picks") or {}
    row("⑨", "微积分八区双态",
        f"{czp.get('n', 0)} 码: " + " ".join("%02d" % n for n in czp.get("sixteen", [])))

    ai = ctx["adv_info"]
    lam = ai["lambda"]
    stance = "顺势" if lam >= 1.05 else ("对冲" if lam <= 0.95 else "中性")
    row("⑩", "对抗博弈 λ",
        f"λ={lam:.3f}（追热Top20平均收益 {ai['gains_mean']:.2f}/期）→ 基调【{stance}】")

    comp = ctx["composite"]
    Hs = ctx["H"]
    print(f"   {'':2}{'H1–H7 合成':18} 综合Top5: "
          + " ".join(f"{n:02d}({comp[n-1]*100:.1f})" for n in top_idx(comp, 5)))


def extract_section(report: str, tag: str, end_tags: tuple[str, ...]) -> str:
    """按节标签抽取整节文本。"""
    lines = report.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(tag):
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if any(lines[j].startswith(t) for t in end_tags):
            end = j
            break
    return "\n".join(lines[start:end]).rstrip()


def build_digest(report: str) -> str:
    """精要模式：头部 + 昨日复盘 + 仪表盘 + 终极推荐 + 八区 + 命中率面板。"""
    chunks = ["\n".join(report.splitlines()[:7])]
    plan = [("[0]", ("[1]",)), ("[1]", ("[2]",)), ("[6]", ("[7]",)),
            ("[7]", ("[8]",)), ("[8]", ("[附录]", "=" * 78))]
    for tag, ends in plan:
        s = extract_section(report, tag, ends)
        if s:
            chunks.append(s)
    return "\n\n".join(chunks)


def main(argv=None) -> int:
    _ensure_utf8_console()
    ap = argparse.ArgumentParser(description="快乐8日常一键：同步+点位+昨日复盘+全模块预测")
    ap.add_argument("--points-file", default=POINTS_FILE, help="点位文件")
    ap.add_argument("--history", default=HISTORY_FILE, help="开奖历史文件")
    ap.add_argument("--raw-points", type=str, default=None, help="录入/覆盖本期 20 个点位")
    ap.add_argument("--period", type=int, default=None, help="点位对应期号（默认=最新开奖+1）")
    ap.add_argument("--date", type=str, default=None, help="点位日期 YYYY-MM-DD（默认=最新开奖日次日）")
    ap.add_argument("--force", action="store_true", help="允许覆盖同期但不同日期的历史点位（谨慎）")
    ap.add_argument("--no-sync", action="store_true", help="跳过开奖同步")
    ap.add_argument("--full-sync", action="store_true", help="全量同步（默认快速）")
    ap.add_argument("--detail", action="store_true", default=True, help="同时生成详细版附录（默认开）")
    ap.add_argument("--no-detail", action="store_true", help="不生成详细版附录")
    ap.add_argument("--brief", action="store_true", help="控制台只打精要摘要（默认打完整标准报告）")
    ap.add_argument("--full-print", action="store_true", help="控制台连详细版附录一并打印")
    ap.add_argument("--hit-days", type=int, default=40, help="命中率面板滚动天数（默认40）")
    args = ap.parse_args(argv)

    detail = bool(args.detail) and not bool(args.no_detail)
    t_all = time.perf_counter()

    print()
    print(BAR)
    print("  快乐8「点位 + 遗漏」· 日常分析流水线")
    print(f"  运行时间 {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(BAR)

    # ---------------- 步骤 1：同步 ----------------
    step(1, 4, "同步最新开奖数据")
    if not args.no_sync:
        rc = sync_history(args.history, dry_run=False, quick=not args.full_sync)
        if rc != 0:
            print("[中止] 同步失败，可加 --no-sync 使用本地数据继续")
            return rc
    else:
        print("（已跳过：--no-sync）")

    # ---------------- 步骤 2：点位 ----------------
    step(2, 4, "点位准备")
    ds = load_dataset(args.points_file, args.history)
    if args.raw_points:
        nums = parse_points(args.raw_points)
        period = args.period if args.period is not None else next_period_after(ds.draws[-1] if ds.draws else None)
        date = args.date or next_target_date(ds) or datetime.date.today().strftime("%Y-%m-%d")
        if period in ds.draws_by_period and not args.force:
            print(f"[错误] 目标期 {period}（{ds.draws_by_period[period].date}）已开奖；"
                  f"录入点位会覆盖该期历史预测。请确认期号/日期，或加 --force。")
            return 2
        try:
            upsert_points(args.points_file, date, period, nums, allow_overwrite=args.force)
        except ValueError as e:
            print(f"[错误] {e}")
            return 2
        ds = load_dataset(args.points_file, args.history)
    # 告警须在录入点位后统一输出：否则 --raw-points 的首次加载会把
    # 「尚无待预测期」这一录入前的状态误报给用户，与最终目标期自相矛盾。
    for w in ds.warnings:
        print("[数据告警]", w)
    target = ds.target_period
    pts = sorted(ds.point_sets.get(target, ()))
    if pts:
        pdate = ds.point_dates.get(target, "?")
        if args.raw_points:
            # 命令行录入：回显「本次录入」的号码本身（不是文件名），并校验落盘一致，
            # 避免标签写「本次录入」却把文件名当内容展示的口径错位。
            entered = sorted(parse_points(args.raw_points))
            src = f"本次录入（命令行 --raw-points，已写入 {args.points_file}）"
            print(f"[点位] 目标期 {target}（{pdate}）｜{src}")
            if entered != pts:
                missing = sorted(set(entered) - set(pts))
                extra = sorted(set(pts) - set(entered))
                print(f"[错误] 录入号码与落盘点位不一致：录入 {len(entered)} 个 / 落盘 {len(pts)} 个")
                if missing:
                    print("        落盘缺少: " + " ".join(f"{n:02d}" for n in missing))
                if extra:
                    print("        落盘多出: " + " ".join(f"{n:02d}" for n in extra))
                return 2
        else:
            print(f"[点位] 目标期 {target}（{pdate}）｜读取自点位文件 {args.points_file}，共 {len(pts)} 个")
        print("       " + " ".join(f"{n:02d}" for n in pts))
    else:
        print(f"[警告] 目标期 {target} 无点位，无法产出主攻清单；请用 --raw-points 录入")

    # ---------------- 步骤 3：全模块预测 ----------------
    step(3, 4, "执行全量预测模块（含昨日深度复盘）")
    import kl8_report as R
    import kl8_forward_test as FT

    # 预注册前瞻检验：先结算已开奖的登记期，再登记本期（仅待开预测）
    FT.settle_pending(ds)

    actual_target = ds.draws_by_period.get(target)
    tds = truncated(ds, target)
    print(f"  目标期 {target}（{ds.point_dates.get(target, '已开奖结算')}）｜"
          f"历史截至 {tds.draws[-1].period} ｜ 样本 {tds.T} 期")

    t0 = time.perf_counter()
    feat, ctx = R.run_pipeline(tds, point_period=None)
    ms_pipe = (time.perf_counter() - t0) * 1000

    if actual_target is None and target >= FT.START_PERIOD:
        try:
            FT.record_prediction(tds, ctx, target)
        except Exception as e:
            print(f"[前瞻检验][警告] {e}")

    print(f"  全模块执行完成（pipeline {ms_pipe:.0f} ms）→ 逐模块明细读出：")
    module_readout(feat, ctx, tds)
    print()
    print(f"  特征列 {len(feat.cols)} 维 → H1–H7 七维评分合成完毕")
    print("  " + FT.status_line())

    rep_std = R.build_report(ctx["ds"], feat, ctx, ds_orig=ds, actual_target=actual_target,
                             hit_days=args.hit_days)
    rep_full = rep_std + "\n\n" + R.build_detail(ctx, feat) if detail else rep_std

    base = ds.point_dates.get(target, datetime.date.today().strftime("%Y-%m-%d")).replace("-", "")
    out, dout = report_paths(base)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(rep_std + "\n", encoding="utf-8")
    print(f"[已写入] {out}")
    if detail:
        Path(dout).write_text(rep_full + "\n", encoding="utf-8")
        print(f"[已写入] {dout}（附录 A–F：全三元组子分/Top20矩阵/扇区能量/八区全量/80码排序/近期逐期明细）")

    # ---------------- 步骤 4：控制台展示 ----------------
    step(4, 4, "分析推荐结果展示")
    if args.full_print:
        print(rep_full)
    elif args.brief:
        print_banner("控制台精要汇总（完整内容见 output/reports/）")
        print(build_digest(rep_std))
    else:
        print(rep_std)

    print()
    print(BAR)
    print(f"  完成 ｜ 总用时 {(time.perf_counter()-t_all):.1f}s ｜ 目标期 {target}")
    print("  开奖公布后再次运行本脚本：自动同步→结算上期→滚动复盘推荐战绩。")
    print(BAR)
    print()
    # 最终交付：主攻三元组纯净可复制清单（控制台最后一段）
    for ln in R.copyable_triples(ctx):
        print(ln)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
