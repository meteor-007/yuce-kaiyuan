# -*- coding: utf-8 -*-
"""从中国福彩官网 API 同步快乐8开奖历史，校正本地 kl8_history_final.txt。"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

HISTORY_FILE = "kl8_history_final.txt"
CWL_API = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    "?name=kl8&issueCount=&issueStart=&issueEnd=&dayStart=&dayEnd="
    "&pageNo={page}&pageSize={size}&week=&systemType=PC"
)


def _fetch_page(page: int, size: int = 100) -> dict:
    url = CWL_API.format(page=page, size=size)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cwl.gov.cn/",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_official(max_pages: int = 50) -> dict[int, tuple[str, tuple[int, ...]]]:
    """返回 period -> (date, sorted_numbers)。"""
    official: dict[int, tuple[str, tuple[int, ...]]] = {}
    for page in range(1, max_pages + 1):
        data = _fetch_page(page, 100)
        rows = data.get("result") or []
        if not rows:
            break
        for item in rows:
            period = int(item["code"])
            date = item["date"].split("(")[0]
            nums = tuple(sorted(int(x) for x in item["red"].split(",") if x.strip()))
            if len(nums) != 20:
                continue
            official[period] = (date, nums)
        print(f"  拉取第 {page} 页: +{len(rows)} 条，累计 {len(official)}")
        if len(rows) < 100:
            break
    return official


def fetch_official_recent(pages: int = 2) -> dict[int, tuple[str, tuple[int, ...]]]:
    """日常快速同步：只拉最近若干页（默认 200 期）。"""
    return fetch_official(max_pages=max(1, pages))


def fetch_ydniu_recent(year: int | None = None) -> dict[int, tuple[str, tuple[int, ...]]]:
    """备用源：一定牛历史页（福彩 API 开奖后常有延迟，用作补洞）。"""
    import datetime as _dt
    y = year or _dt.date.today().year
    url = f"https://www.ydniu.com/open/kl8History-{y}.html"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", "replace")
    out: dict[int, tuple[str, tuple[int, ...]]] = {}
    # <td>2026274</td><td>2026-10-15 星期四</td><td ...>08 09 ...</td>
    pat = re.compile(
        r"<tr>\s*<td>(\d{7})</td>\s*<td>(\d{4}-\d{2}-\d{2})[^<]*</td>\s*<td[^>]*>(.*?)</td>",
        re.S | re.I,
    )
    for m in pat.finditer(html):
        period = int(m.group(1))
        date = m.group(2)
        nums = tuple(sorted(int(x) for x in re.findall(r"\d{1,2}", m.group(3))))
        if len(nums) != 20 or len(set(nums)) != 20:
            continue
        if not all(1 <= n <= 80 for n in nums):
            continue
        out[period] = (date, nums)
    return out


def merge_fallback(
    primary: dict[int, tuple[str, tuple[int, ...]]],
    fallback: dict[int, tuple[str, tuple[int, ...]]],
) -> tuple[dict[int, tuple[str, tuple[int, ...]]], list[int]]:
    """用备用源补齐 primary 中缺失的更新期；不覆盖已有号码。"""
    added = []
    merged = dict(primary)
    for p, (d, n) in fallback.items():
        if p not in merged:
            merged[p] = (d, n)
            added.append(p)
    return merged, sorted(added)


def load_local(path: Path) -> dict[int, tuple[str, tuple[int, ...], str]]:
    """period -> (date, sorted_nums, raw_line)。"""
    local = {}
    pat = re.compile(r"date:([^,]+),period:(\d+),numbers:(.+)")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pat.match(line)
        if not m:
            continue
        date, period_s, nums_s = m.groups()
        period = int(period_s)
        nums = tuple(sorted(int(x) for x in nums_s.replace(",", "-").split("-") if x.strip()))
        local[period] = (date, nums, line)
    return local


def fmt_line(date: str, period: int, nums: tuple[int, ...]) -> str:
    return f"date:{date},period:{period},numbers:" + "-".join(f"{n:02d}" for n in nums)


def sync(history_path: str = HISTORY_FILE, dry_run: bool = False, quick: bool = False) -> int:
    path = Path(history_path)
    print("=" * 60)
    print("  快乐8 开奖数据同步" + (" · 快速模式" if quick else " · 全量"))
    print("=" * 60)
    print("[1] 拉取福彩官网…")
    official = fetch_official_recent(2) if quick else fetch_official()
    if not official:
        print("[警告] 福彩官网无数据，尝试备用源…")
        official = {}

    local = load_local(path)
    latest_loc = max(local) if local else 0
    latest_cwl = max(official) if official else 0

    # 福彩 API 开奖后常滞后：一定牛补齐缺失的更新期（不覆盖已有号码）
    print("[1b] 检查一定牛备用源…")
    try:
        yd = fetch_ydniu_recent()
        yd_latest = max(yd) if yd else 0
        print(f"     一定牛最新期: {yd_latest}" + (f" ({yd[yd_latest][0]})" if yd else ""))
        if yd and (not official or yd_latest > latest_cwl):
            # 快速模式：只并入比 CWL/本地更新的期，避免备用源整页冲掉全量校正集
            if quick:
                floor = max(latest_cwl, latest_loc)
                yd = {p: v for p, v in yd.items() if p > floor}
            official, fb_added = merge_fallback(official, yd)
            if fb_added:
                print(f"     备用源补入 {len(fb_added)} 期: "
                      + " ".join(str(p) for p in sorted(fb_added, reverse=True)[:10]))
        elif not yd:
            print("     备用源无有效数据")
    except Exception as e:
        print(f"     备用源失败: {type(e).__name__}: {e}")

    if not official:
        print("[错误] 未能拉取到任何开奖数据")
        return 1

    latest_off = max(official)
    print(f"[2] 合并最新期: {latest_off} ({official[latest_off][0]})")
    print(f"    本地最新期: {latest_loc}" + (f" ({local[latest_loc][0]})" if local else ""))
    if latest_cwl and latest_off > latest_cwl:
        print(f"    注: 福彩 API 仍停在 {latest_cwl}，已用备用源补齐")

    added, fixed, date_fixed = [], [], []
    for period, (date, nums) in official.items():
        if period not in local:
            added.append(period)
            continue
        ld, ln, _ = local[period]
        if ln != nums:
            fixed.append(period)
        elif ld != date:
            date_fixed.append(period)

    # 全量模式才统计「本地多余」；快速模式只增量覆盖
    removed = [] if quick else sorted(set(local) - set(official))
    print(f"[3] 对比结果: 新增 {len(added)} | 号码校正 {len(fixed)} | 日期校正 {len(date_fixed)}"
          + (f" | 本地多余 {len(removed)}" if not quick else " | 快速增量"))
    if added:
        print("    新增期号(近10): " + " ".join(str(p) for p in sorted(added, reverse=True)[:10]))
    if fixed:
        for p in sorted(fixed):
            print(f"    校正 {p}:")
            print(f"      本地: {' '.join(f'{n:02d}' for n in local[p][1])}")
            print(f"      官方: {' '.join(f'{n:02d}' for n in official[p][1])}")
    if date_fixed:
        print("    日期校正: " + " ".join(str(p) for p in sorted(date_fixed)[:20]))

    # 合并：官方覆盖；快速模式保留全部本地未触及期
    merged: dict[int, tuple[str, tuple[int, ...]]] = {}
    for p, (d, n, _) in local.items():
        if quick or p not in official:
            merged[p] = (d, n)
    for p, (d, n) in official.items():
        merged[p] = (d, n)

    lines = [
        fmt_line(d, p, n)
        for p, (d, n) in sorted(merged.items(), key=lambda x: (-x[0],))
    ]
    changed = bool(added or fixed or date_fixed)
    if dry_run:
        print(f"[4] dry-run：将写入 {len(lines)} 行，未实际保存")
        return 0

    if changed or not path.exists():
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[4] 已写入 {path}（共 {len(lines)} 期）")
    else:
        print(f"[4] 已是最新，无需改写（共 {len(lines)} 期）")

    # 展示以合并后文件为准（本地可能已超前于福彩 API）
    final_latest = max(merged)
    fd, fn = merged[final_latest]
    print("=" * 60)
    print(f"  同步完成 · 最新开奖 {final_latest} · {fd}")
    print("  号码: " + " ".join(f"{n:02d}" for n in fn))
    if latest_cwl and final_latest > latest_cwl:
        print(f"  （福彩 API 仍为 {latest_cwl}，以本地/备用源为准）")
    print("=" * 60)
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="同步快乐8开奖历史")
    ap.add_argument("--history", default=HISTORY_FILE, help="历史文件路径")
    ap.add_argument("--dry-run", action="store_true", help="只对比不写文件")
    ap.add_argument("--quick", action="store_true", help="只拉最近约200期（日常推荐）")
    ap.add_argument("--full", action="store_true", help="全量拉取并校正（默认）")
    args = ap.parse_args(argv)
    quick = bool(args.quick) and not bool(args.full)
    # 无参数时默认全量（兼容首次校正）；日常脚本会显式传 --quick
    if not args.quick and not args.full:
        quick = False
    return sync(args.history, dry_run=args.dry_run, quick=quick)


if __name__ == "__main__":
    sys.exit(main())
