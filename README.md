# yuce-kaiyuan · 快乐8「点位+遗漏」分析系统

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-27%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

> 本地运行的快乐8每日分析流水线：20 个核心点位 → 环形三元组（前一位/点位/后一位）→ 10 个分析模块 → H1–H7 评分 → 主攻三元组清单；附带严格 walk-forward 回测、全历史条件扫描和 200 期预注册前瞻检验。

## 重要声明

- 快乐8每期从 80 个号中无放回开 20 个，任一号码单期命中率恒为 **20/80 = 25%**。
- 本项目是**结构化描述与复盘工具**，不是可盈利预测器；全历史扫描与严格 walk-forward 未发现统计上可验证的 edge。
- 每日推荐不应表述为「显著提升」；请理性参考、量力而行。

## 功能

- **每日一键流水线**：同步开奖 → 录入 20 个点位 → 昨日深度复盘 → 全模块预测 → 标准/详细报告落盘。
- **10 个分析模块**：连号集群三元组、马尔可夫 P01/P11、斐波那契/黄金分割、SVD+共现条件概率、复变傅里叶周期、图网络共现拥挤度、波动率聚类、香农熵、微积分八区双态、对抗博弈 λ。
- **H1–H7 评分**：热度、遗漏、集群、周期、波动、定位、对抗，七维加权合成，并给出顺势榜/对冲榜。
- **严格评估**：299 期 walk-forward 回测、全历史 39 项条件扫描（BH-FDR）、7 项消融、按期聚类置信区间。
- **预注册前瞻检验**：200 期、单侧 α=0.01、规则冻结、append-only 登记与自动结算。
- **数据校验**：期号/日期双主键、防未来数据泄漏、防同期点位覆盖、23→27 项回归测试。

## 快速开始

```bash
# 0) 安装依赖
pip install -r requirements.txt

# 1) 同步最新开奖（福彩官网 + 一定牛备用源）
python kl8_sync.py --quick

# 2) 录入当期 20 个点位并执行全量分析
python kl8_daily.py --no-sync --raw-points "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20" --full-print

# 3) 运行测试
python -m pytest -q

# 4) 查看预注册前瞻检验进度
python kl8_forward_test.py --report
```

Windows 下也可以直接运行 `日常分析.bat`（把 20 个点位作为参数传入）。

## 数据文件

- `daily_points.txt`：每日 20 个核心点位，格式 `date:YYYY-MM-DD,period:YYYYNNN,points:NN NN ...`。
- `kl8_history_final.txt`：开奖历史，格式 `date:YYYY-MM-DD,period:YYYYNNN,numbers:NN-NN-...`。
- 两份文件均由脚本校验：20 个不重复号码、1–80、期号/日期一致；建议每天先同步再录入点位。

## 每日流程

1. **同步**：`python kl8_sync.py --quick`，确认昨日开奖已入库。
2. **录入点位**：`python kl8_daily.py --no-sync --raw-points "<20 个号码>" --full-print`；脚本会自动写入 `daily_points.txt` 顶部，目标期默认 = 最新开奖期 +1。
3. **复盘**：报告 `[0]` 节展示昨日开奖、形态、核心点位/扩展集合命中、昨日主攻组实战成绩。
4. **输出**：标准版/详细版报告落盘到 `output/reports/`；控制台最后给出本期主攻三元组纯净清单。

## 预注册前瞻检验

- 协议：[`docs/预注册_主攻三元组_20261110.md`](docs/预注册_主攻三元组_20261110.md)
- 起点：第 2026300 期；目标：200 个可评估期。
- 规则冻结：`compute_picks → select_main_groups / build_triples`；规则指纹覆盖 `kl8_core.py` + `kl8_methods.py` + `kl8_report.py` + 规则常量。
- 主要终点：主攻组「≥2 码」率 vs 精确超几何 15.26%，单侧 α=0.01。
- 成功需同时满足：p<0.01、前后半均 >15.26%、按期聚类 95%CI 下界 >15.26%；否则关闭该方向。
- 登记/结算：`output/forward_test/pre_registered_20261110.jsonl`（append-only）。

## 统计结论（截至 2026-11-11）

- 2121 期开奖（2020-10-28 至 2026-11-10）：16 项 i.i.d. 均匀零假设检验，BH-FDR 后无一显著（最小 q=0.248）。
- 核心点位长期命中率约 **25.28%**（按期聚类 95%CI [24.39%, 26.18%]），覆盖 25% 基线。
- 299 期 walk-forward：top5/top20/five/seven/cluster5 的 95%CI 全部覆盖基线；39 项条件扫描 0 项通过 FDR。
- 预注册前瞻检验已启动：第 2026300 期主攻 42-43-44 / 05-06-07 结算 0/2 组；当前进度以 `python kl8_forward_test.py --report` 为准。
- 完整复盘与文献评估：[`docs/命中率深度复盘_20261110.md`](docs/命中率深度复盘_20261110.md)。

## 项目结构

```
kl8_daily.py            每日流水线入口
kl8_core.py             数据层与防泄漏截断
kl8_methods.py          10 个分析模块
kl8_report.py           H1–H7 评分与报告
kl8_forward_test.py     200 期预注册前瞻检验
kl8_backtest.py         299 期 walk-forward 回测
kl8_edge_scan.py        全历史条件扫描（BH-FDR）
kl8_audit.py            模块耗时/接线/消融审计
kl8_sync.py             官方开奖同步
kl8_excel_export.py     文本报告 → Excel
test_kl8_comprehensive.py / test_kl8_eval.py   回归测试
每日分析指令.txt         可复用每日指令模板
docs/                   预注册协议、深度复盘、知识库
output/forward_test/    预注册登记与元数据
```

## 测试

```bash
python -m pytest -q
# 27 passed
```

## License

本项目采用 [MIT License](LICENSE)，Copyright (c) 2026 meteor-007。

## 免责声明

快乐8是随机开奖，任何方法都不能改变单号 25% 的边际概率和长期负期望。本项目仅用于数据分析、复盘与工程实践，不构成任何购彩建议；请遵守当地法律法规，理性参考、量力而行。