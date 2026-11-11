# yuce-kaiyuan · 快乐8「点位+遗漏」分析系统

一个本地运行的快乐8每日分析流水线：20 个核心点位 → 环形三元组（前一位/点位/后一位）→ 10 个分析模块 → H1–H7 评分 → 主攻三元组清单；附带严格 walk-forward 回测、全历史条件扫描和 200 期预注册前瞻检验。

## 重要声明

- 快乐8每期从 80 个号中无放回开 20 个，任一号码单期命中率恒为 **20/80 = 25%**。
- 本项目是**结构化描述与复盘工具**，不是可盈利预测器；全历史扫描与严格 walk-forward 未发现统计上可验证的 edge。
- 每日推荐不应表述为「显著提升」；请理性参考、量力而行。

## 快速开始

```bash
# 1) 同步最新开奖（福彩官网 + 备用源）
python kl8_sync.py --quick

# 2) 录入当期 20 个点位并执行全量分析
python kl8_daily.py --no-sync --raw-points "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20" --full-print

# 3) 测试
python -m pytest -q

# 4) 预注册前瞻检验进度
python kl8_forward_test.py --report
```

Windows 下也可以直接运行 `日常分析.bat`（把 20 个点位作为参数传入）。

## 主要文件

| 文件 | 用途 |
|---|---|
| `kl8_daily.py` | 每日流水线：同步 → 点位 → 全模块预测 → 报告 |
| `kl8_core.py` | 数据层：解析两份数据文件、构建数据集/命中矩阵/遗漏序列、防泄漏截断 |
| `kl8_methods.py` | 10 个分析模块（连号集群、Markov、斐波那契、SVD/共现、傅里叶、图网络、波动率、熵、八区、对抗博弈） |
| `kl8_report.py` | H1–H7 评分、标准报告 [0]–[8]、详细附录 A–F、命中率面板 |
| `kl8_forward_test.py` | 200 期预注册前瞻检验：append-only 登记、自动结算、统计判定 |
| `kl8_backtest.py` / `kl8_edge_scan.py` / `kl8_audit.py` | 回测、全历史条件扫描、模块审计 |
| `kl8_sync.py` | 官方开奖同步（福彩官网 + 一定牛备用源） |
| `kl8_excel_export.py` | 文本报告 → Excel |
| `test_kl8_comprehensive.py` | 回归测试（当前 27 项全部通过） |

## 数据与产出

- 输入：`daily_points.txt`（每日 20 个核心点位）、`kl8_history_final.txt`（开奖历史）。
- 产出：`output/reports/分析报告_YYYYMMDD.txt`（标准版/详细版）。
- 预注册检验：`output/forward_test/pre_registered_20261110.jsonl`（append-only）与同名 `_meta.json`。
- 协议与复盘：
  - `每日分析指令.txt`（可复用模板）
  - `docs/预注册_主攻三元组_20261110.md`
  - `docs/命中率深度复盘_20261110.md`

## 预注册前瞻检验

- 起点：第 2026300 期；目标：200 个可评估期。
- 规则冻结：`compute_picks → select_main_groups / build_triples`；规则指纹覆盖 `kl8_core.py` + `kl8_methods.py` + `kl8_report.py` + 规则常量。
- 主要终点：主攻组「≥2 码」率 vs 精确超几何 15.26%，单侧 α=0.01。
- 成功需同时满足：p<0.01、前后半均 >15.26%、按期聚类 95%CI 下界 >15.26%；否则关闭该方向。
- 查看进度：`python kl8_forward_test.py --report`。

## 测试

```bash
python -m pytest -q
# 27 passed
```