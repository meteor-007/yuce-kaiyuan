# -*- coding: utf-8 -*-
"""预注册 200 期前瞻检验：主攻三元组 ≥2 码率 vs 精确超几何 15.26%。

协议：docs/预注册_主攻三元组_20261110.md
登记：output/forward_test/pre_registered_20261110.jsonl（append-only）
元数据：output/forward_test/pre_registered_20261110_meta.json

规则冻结：
  kl8_report.compute_picks -> kl8_methods.select_main_groups / build_triples
  规则指纹覆盖 kl8_core.py / kl8_methods.py / kl8_report.py + TRIPLE_WEIGHTS/H_DEF。
  任何改动都会改变指纹，登记函数拒绝继续；需新开预注册。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from scipy import stats

from kl8_core import CONFIG, clustered_rate_ci, load_dataset
import kl8_methods as M
import kl8_report as R

REGISTRY_DIR = Path(CONFIG["output_dir"]) / "forward_test"
META_PATH = REGISTRY_DIR / "pre_registered_20261110_meta.json"
EVENTS_PATH = REGISTRY_DIR / "pre_registered_20261110.jsonl"

START_PERIOD = 2026300
TARGET_PERIODS = 200
P0_GE2 = 0.1526       # 3 码组 P(≥2 码) 精确超几何
P0_FULL3 = 0.0139     # 3 码组 3 连全中
ALPHA = 0.01


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rule_fingerprint():
    """规则指纹：选号相关源文件 + 规则常量的 SHA256 前 16 位。"""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("kl8_core.py", "kl8_methods.py", "kl8_report.py"):
        h.update((root / name).read_bytes())
    h.update(json.dumps(M.TRIPLE_WEIGHTS, sort_keys=True).encode("utf-8"))
    h.update(json.dumps(R.H_DEF, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()[:16]


def ensure_meta(meta_path=META_PATH):
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    meta = {
        "protocol": "docs/预注册_主攻三元组_20261110.md",
        "created_at": _now(),
        "start_period": START_PERIOD,
        "target_periods": TARGET_PERIODS,
        "p0_ge2": P0_GE2,
        "p0_full3": P0_FULL3,
        "alpha": ALPHA,
        "rule_fingerprint": rule_fingerprint(),
        "rule": "compute_picks(ctx)['groups']；规则冻结，禁止中途调参/改权重",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def load_events(events_path=EVENTS_PATH):
    events_path = Path(events_path)
    if not events_path.exists():
        return []
    out = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _append_event(event, events_path=EVENTS_PATH):
    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _group_record(g):
    return {
        "members": [int(x) for x in g["members"]],
        "center": int(g["center"]),
        "score": float(g["score"]),
        "tier": g.get("tier", ""),
        "labels": list(g.get("labels", [])),
    }


def record_prediction(ds, ctx, target, events_path=EVENTS_PATH, meta_path=META_PATH):
    """登记本期主攻组。幂等；点位不同或规则指纹变化时拒绝。"""
    meta = ensure_meta(meta_path)
    fp = rule_fingerprint()
    if fp != meta["rule_fingerprint"]:
        raise ValueError(
            f"规则指纹已变化（注册 {meta['rule_fingerprint']} -> 当前 {fp}）；"
            f"预注册检验不接受规则变更，请新开预注册。"
        )
    target = int(target)
    if target < int(meta["start_period"]):
        return None
    points = sorted(int(x) for x in ds.point_sets.get(target, ()))
    events = load_events(events_path)
    preds = [e for e in events
             if e.get("event") == "prediction" and e.get("period") == target]
    if preds:
        old = preds[-1]
        if old.get("points") != points:
            raise ValueError(
                f"期 {target} 已登记点位 {old.get('points')}，本次 {points} 不同；"
                f"预注册检验不允许修改已登记预测。"
            )
        return old
    groups = R.compute_picks(ctx).get("groups") or []
    record = {
        "event": "prediction",
        "period": target,
        "date": ds.point_dates.get(target),
        "recorded_at": _now(),
        "rule_fingerprint": fp,
        "points": points,
        "groups": [_group_record(g) for g in groups],
        "n_groups": len(groups),
    }
    _append_event(record, events_path)
    n_pred = len([e for e in events if e.get("event") == "prediction"]) + 1
    print(f"[前瞻检验] 第 {target} 期已登记（已登记 {n_pred}/{meta['target_periods']} 期）："
          + " ".join("-".join("%02d" % m for m in g["members"]) for g in record["groups"]))
    return record


def _settle_one(pred, draw):
    actual = sorted(int(x) for x in draw.numbers)
    actual_set = set(actual)
    details = []
    g_hit2 = 0
    g_full3 = 0
    dedup = set()
    for g in pred.get("groups", []):
        members = [int(x) for x in g.get("members", [])]
        hits = sorted(set(members) & actual_set)
        hit_ge2 = len(hits) >= 2
        full3 = len(hits) == 3
        g_hit2 += int(hit_ge2)
        g_full3 += int(full3)
        dedup.update(members)
        details.append({**g, "hits": hits, "n_hits": len(hits),
                        "hit_ge2": hit_ge2, "full3": full3})
    return {
        "event": "settlement",
        "period": int(pred["period"]),
        "settled_at": _now(),
        "draw_date": draw.date,
        "actual": actual,
        "groups": details,
        "g_n": len(pred.get("groups", [])),
        "g_hit2": g_hit2,
        "g_full3": g_full3,
        "dedup_hits": len(dedup & actual_set),
        "dedup_n": len(dedup),
    }


def settle_pending(ds, events_path=EVENTS_PATH, meta_path=META_PATH):
    """把已开奖的登记期结算；append-only，幂等。"""
    ensure_meta(meta_path)
    events = load_events(events_path)
    preds = {e["period"]: e for e in events if e.get("event") == "prediction"}
    settled = {e["period"] for e in events if e.get("event") == "settlement"}
    out = []
    for period in sorted(set(preds) - settled):
        draw = ds.draws_by_period.get(period)
        if draw is None:
            continue
        ev = _settle_one(preds[period], draw)
        _append_event(ev, events_path)
        out.append(ev)
        print(f"[前瞻检验] 第 {period} 期结算：主攻 {ev['g_hit2']}/{ev['g_n']} 组≥2码，"
              f"3连全中 {ev['g_full3']}，去重 {ev['dedup_hits']}/{ev['dedup_n']}")
    return out


def _rate(rows):
    n = sum(r.get("g_n", 0) for r in rows)
    k = sum(r.get("g_hit2", 0) for r in rows)
    return k, n, (k / n if n else 0.0)


def _tier_stats(rows):
    out = {}
    for r in rows:
        for g in r.get("groups", []):
            tier = g.get("tier") or "未分级"
            d = out.setdefault(tier, {"groups": 0, "ge2": 0, "full3": 0})
            d["groups"] += 1
            d["ge2"] += int(g.get("hit_ge2", False))
            d["full3"] += int(g.get("full3", False))
    return out


def report(events_path=EVENTS_PATH, meta_path=META_PATH):
    meta = ensure_meta(meta_path)
    events = load_events(events_path)
    preds = {e["period"]: e for e in events if e.get("event") == "prediction"}
    settles = {e["period"]: e for e in events if e.get("event") == "settlement"}
    rows = [settles[p] for p in sorted(preds) if p in settles]
    eligible = [r for r in rows if r.get("g_n", 0) > 0]
    k, n, rate = _rate(eligible)
    print("=" * 78)
    print("  预注册 200 期前瞻检验 · 主攻三元组 ≥2 码")
    print(f"  协议: {meta['protocol']} ｜ 规则指纹: {meta['rule_fingerprint']}")
    print(f"  进度: 已登记 {len(preds)}/{meta['target_periods']} 期 ｜ "
          f"已结算 {len(rows)} 期 ｜ 可评估 {len(eligible)} 期")
    print("=" * 78)
    if not eligible:
        print("  暂无可评估的已结算期。")
        print("=" * 78)
        return {"status": "pending", "n": 0, "preds": len(preds), "settled": len(rows)}
    ci = clustered_rate_ci([r["g_hit2"] for r in eligible], [r["g_n"] for r in eligible])
    p = stats.binomtest(k, n, P0_GE2, alternative="greater").pvalue if n else 1.0
    half1 = _rate(eligible[:len(eligible) // 2])
    half2 = _rate(eligible[len(eligible) // 2:])
    full3 = sum(r.get("g_full3", 0) for r in eligible)
    dedup_hits = sum(r.get("dedup_hits", 0) for r in eligible)
    dedup_n = sum(r.get("dedup_n", 0) for r in eligible)
    print(f"  主口径: 主攻组 ≥2 码 {k}/{n} = {rate * 100:.2f}% ｜ 基线 {P0_GE2 * 100:.2f}% ｜ "
          f"lift={rate / P0_GE2:.3f} ｜ 单侧精确 p={p:.4f}")
    print(f"  按期聚类 95%CI: [{ci['lo'] * 100:.2f}%, {ci['hi'] * 100:.2f}%]")
    print(f"  3 连全中: {full3}/{n} = {full3 / n * 100:.2f}% ｜ 基线 {P0_FULL3 * 100:.2f}%")
    print(f"  去重号: {dedup_hits}/{dedup_n} = {dedup_hits / dedup_n * 100:.2f}% ｜ 基线 25.00%"
          if dedup_n else "  去重号: 无")
    print(f"  前半 {half1[1]} 组: {half1[2] * 100:.2f}% ｜ 后半 {half2[1]} 组: {half2[2] * 100:.2f}%")
    for tier, s in sorted(_tier_stats(eligible).items()):
        print(f"  [{tier}] {s['ge2']}/{s['groups']} = {s['ge2'] / s['groups'] * 100:.2f}% ≥2 码"
              f" ｜ 3 连全中 {s['full3']}")
    verdict = "进行中"
    if len(eligible) >= meta["target_periods"]:
        ok = (p < meta["alpha"]) and (half1[2] > P0_GE2) and (half2[2] > P0_GE2) and (ci["lo"] > P0_GE2)
        verdict = "通过（拒绝 H0）" if ok else "未通过（关闭该方向）"
    print(f"  判定: {verdict}（成功门槛: p<{meta['alpha']} 且前后半均>{P0_GE2 * 100:.2f}% "
          f"且 CI 下界>{P0_GE2 * 100:.2f}%）")
    print("=" * 78)
    return {"status": verdict, "k": k, "n": n, "rate": rate, "p": p, "ci": ci,
            "half1": half1, "half2": half2, "full3": full3,
            "dedup_hits": dedup_hits, "dedup_n": dedup_n,
            "eligible": len(eligible), "preds": len(preds), "settled": len(rows)}


def status_line(events_path=EVENTS_PATH, meta_path=META_PATH):
    meta = ensure_meta(meta_path)
    events = load_events(events_path)
    preds = len([e for e in events if e.get("event") == "prediction"])
    settled = len([e for e in events if e.get("event") == "settlement"])
    return (f"[前瞻检验] 已登记 {preds}/{meta['target_periods']} 期，"
            f"已结算 {settled} 期；规则指纹 {meta['rule_fingerprint']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="预注册 200 期前瞻检验")
    ap.add_argument("--report", action="store_true", help="输出进度与统计（默认行为）")
    ap.add_argument("--settle", action="store_true", help="先结算已开奖的登记期")
    ap.add_argument("--events", default=str(EVENTS_PATH))
    ap.add_argument("--meta", default=str(META_PATH))
    args = ap.parse_args(argv)
    if args.settle:
        settle_pending(load_dataset(), args.events, args.meta)
    report(args.events, args.meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())