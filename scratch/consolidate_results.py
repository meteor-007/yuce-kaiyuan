# -*- coding: utf-8 -*-
"""合并主实验、C稳健性、随机化检验与窗口选择诊断，产出最终诊断 JSON/TXT。"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "diagnostics")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

main = json.load(open(os.path.join(OUT, "candidates_eval_20261110.json"), encoding="utf-8"))
rob = json.load(open(os.path.join(OUT, "candidates_eval_robustness_C_20261110.json"), encoding="utf-8"))
rnd = json.load(open(os.path.join(OUT, "candidates_eval_C_randomization_20261110.json"), encoding="utf-8"))

rows = main["extras"]["C_rows"]
wins = np.array([r["wins"] for r in rows], float)
mg = np.array([r["m"] for r in rows], float)
W = 40
pooled = np.array([wins[i:i + W].sum() / mg[i:i + W].sum() for i in range(len(rows) - W + 1)])
recent40 = wins[-40:].sum() / mg[-40:].sum()
window_diag = {
    "recent_40_pooled_ge2_rate": float(recent40),
    "recent_40_groups": f"{int(wins[-40:].sum())}/{int(mg[-40:].sum())}",
    "rolling_40_windows": int(len(pooled)),
    "rolling_mean": float(pooled.mean()),
    "rolling_sd": float(pooled.std(ddof=1)),
    "rolling_p05_p50_p95": [float(x) for x in np.percentile(pooled, [5, 50, 95])],
    "fraction_windows_ge_recent": float((pooled >= recent40).mean()),
    "interpretation": ("近40期主攻组≥2码合并率21.1%，处于260个重叠滚动窗口中约90分位；"
                       "40日窗口的29.78%单号命中率对应朴素二项z≈2.1，但这是大量窗口/口径中的一次选择，"
                       "不足以证明edge。"),
}

enum = rnd["eval299"]
c_rob = {e["label"]: e for e in rob["entries"]}
c_notes_field = (
    "主攻三元组共994组/299期（均值3.32组/期），按组合并≥2码=170/994=17.10%，逐期均值17.59%；"
    "同场随机三元组精确超几何基线逐期均值15.11%（固定三码无条件P(≥2)=15.26%、P(3)=1.39%）。"
    "名义配对差+2.48pp，差95%CI[+0.14,+4.81]pp，置换p=0.040。稳健性不足："
    "299期分半后前149期p=0.054、后150期p=0.304；全334点位期名义p=0.014，但2025年35期命中率25.7%明显偏高；"
    "11项检验Bonferroni阈值0.0045、BH-FDR后不显著；分数四分位≥2码率非单调(17.0/16.0/14.4/15.3%)；"
    "模型三元组平均含1.55个场内号 vs 候选池1.38个，场随机基线并非完全同分布。"
    "判定：边界性样本内信号，不足以作为edge部署。")
c_notes_pool = (
    "同一994组 vs 同候选池随机选择（每期从合规环形三元组随机抽m组，m=主攻组数）：逐期基线15.25%，"
    "配对差+2.34pp，差95%CI[+0.15,+4.52]pp，置换p=0.040；Stouffer Z=1.97（p=0.048）；"
    "但精确组合枚举的Fisher合并p≈1.0（离散检验过保守），证据对检验方法敏感。"
    f"分半：前149期p={c_rob['eval_299_first_half']['d_pool_p']:.3f}、"
    f"后150期p={c_rob['eval_299_second_half']['d_pool_p']:.3f}；"
    "334期虽名义p=0.014，但稳定性不成立。判定：名义边界，不构成可跟进edge。")

final_exps = []
for r in main["experiments"]:
    e = dict(r)
    if e["name"] == "C_field_random_triples":
        e["notes"] = c_notes_field
        e["beats_baseline"] = True  # 名义配对差CI下界>0
    elif e["name"] == "C_candidate_pool_random":
        e["notes"] = c_notes_pool
        e["beats_baseline"] = True  # 名义配对差CI下界>0；但见notes稳健性
    final_exps.append(e)

conclusion = (
    "【模板】每日分析指令.txt 已是可复用模板（占位符+执行校验+验收标准），本次无需改动生产文件。\n\n"
    "【核心结论】在严格的 walk-forward、仅用预测期之前数据的口径下（2026001–2026299，299期），"
    "四个候选方法均未提供可复制的正edge：\n"
    "A. Dirichlet-multinomial(α=1/10/100)：均匀对称先验下后验均值排序与原始频率完全一致（α不改变选号），"
    "top5=1.238/1.25 (lift 0.990, p=0.82)，top20=4.91/5.0 (0.981, p=0.33)；换窗30/200/全历史 lift5 在0.90–1.05间摆动，属窗口噪声。\n"
    "B. 逻辑回归/梯度提升：严格分块重训（每20期用此前全部开奖），top5=1.331/1.25 (lift 1.065, p=0.14)，"
    "top20≈5.02/5.0 (1.005, p=0.79)；LR与GBDT的top5集合平均仅重叠30%，命中相关0.22，却给出同样的均值，说明是同一噪声水平。\n"
    "C. 点位场三元组：主攻组≥2码17.1%（994组）vs 场随机逐期基线15.11%（固定三码无条件15.26%）/候选池15.25%，名义p≈0.04，是最接近显著的一项；"
    "但分半不稳定（后半p=0.30）、334期靠2025年35期抬高、11项检验BH-FDR后不显著、分数分位非单调，只能判为边界性噪声/选择偏差，不能部署。\n"
    "D. 覆盖轮盘：68注对子覆盖设计保证h≥2时必有≥2码注，覆盖98.66% vs 随机注96.64%（p≈5e-5），但逐注≥2码率14.98% vs 15.11%（p=0.22），"
    "单号边际命中率24.90% vs 25.00%（p=0.84）——覆盖设计只重排方差/保证结构，不改变期望返奖率，成本约68倍，不构成edge。\n\n"
    "【为什么命中率看起来低】快乐8是80选20的无放回抽样，任何固定号码单期命中概率恒为25%；固定三码P(≥2)=15.26%、P(3连)=1.39%，"
    "主攻3.3组/期时每日期望仅约0.51组≥2码，'多数日子没有全中'是数学必然而非模型失效。"
    "近40日主攻组≥2码合并率21.1%，处于260个重叠滚动40期窗口中约90分位（滚动均值16.7%、95分位22.2%）；"
    "40日单号命中率29.78%对应朴素二项z≈2.1，但40日只是众多可选窗口之一且面板同时展示多个指标，属窗口/指标选择效应。"
    "现有全历史39项条件扫描也全部CI下界≤25%。\n\n"
    "【文献/GitHub对照】公开的彩票/Keno预测项目（LSTM、Transformer、热冷号、马尔可夫）大多展示样本内拟合或单窗口演示；"
    "凡做严格样本外检验的，结果都回归超几何基线。热手/赌徒谬误（Gilovich et al. 1985; Tversky & Kahneman 1971）解释了为何'追热/补冷'直觉强但无预测力；"
    "Dirichlet-multinomial平滑在对称先验下不改变排序（本次已实证）；覆盖设计文献（Wikipedia: Lottery wheeling / Covering design）明确：轮盘只改变结果分布与保底结构，"
    "不改变期望值。因此不存在可结合文献'整改'出命中率edge的空间；能整改的只有流程诚实度、成本与展示。\n\n"
    "【建议】1) 不以提升命中率为目标继续调权重/加模块；2) 将系统定位为结构化描述与复盘工具，保留严格walk-forward与多点位期CI口径；"
    "3) 若仍想验证C的边际信号，需预注册前瞻检验（例如未来200期、单侧α=0.01、固定选择规则），预期会回归15.3%；"
    "4) 唯一合法的'提高命中体验'手段是增加注数/覆盖（成本同比例上升，EV不变），不应表述为预测优势；"
    "5) 报告继续使用'CI覆盖基线=无优势'的判读口径。\n\n"
    "【产物】scratch/eval_candidates.py、scratch/robustness_c.py、scratch/randomization_c.py；"
    "output/diagnostics/candidates_eval_20261110.json、candidates_eval_robustness_C_20261110.json、"
    "candidates_eval_C_randomization_20261110.json、candidates_eval_final_20261110.json/.txt。生产代码未改动。"
)

final = {
    "meta": {
        **main["meta"],
        "beats_baseline_rule": ("计数型(A/B/D单号)：均值CI下界>基线；率型配对(C/D)：逐期配对差95%CI下界>0。"
                                "该规则用于原始实验判定，稳健性与多重比较结论见conclusion。"),
        "statistical_tests": ("计数型：mean±t(0.975,n-1)·se，p=单样本双侧t检验 vs 理论基线；"
                              "率型C/D：逐期配对差，bootstrap百分位CI，符号翻转置换p(20000次)；"
                              "C另做精确条件随机化(组合枚举/Stouffer)与分半、334期稳健性。"),
        "data_window": "开奖2020001–2026299(2120期)；评估2026001–2026299(299期，均有每日20点位)；另用全部334个点位期复核。",
        "generated": "2026-11-10",
    },
    "experiments": final_exps,
    "diagnostics": {
        "window_selection": window_diag,
        "C_robustness": c_rob["eval_299"],
        "C_robustness_all334": c_rob["all_point_periods_334"],
        "C_robustness_halves": [c_rob["eval_299_first_half"], c_rob["eval_299_second_half"]],
        "C_randomization_exact": rnd,
        "D_design": main["extras"]["D_design"],
        "multiple_testing": main["extras"]["multiple_testing_note"],
        "literature_notes": [
            "快乐8每期80选20无放回：固定号码边际命中率恒为20/80=25%，固定三码P(≥2)=15.26%、P(3)=1.39%。",
            "Dirichlet-multinomial对称均匀先验：后验均值(n_i+α)/(Σn+80α)对n_i单调，α不改变排序。",
            "热手/赌徒谬误：Gilovich, Vallone & Tversky (1985); Tversky & Kahneman (1971) —— 人类在独立随机序列中过度解读冷热。",
            "彩票轮盘/覆盖设计：只改变注间相关与保底结构，不改变期望值（Wikipedia: Lottery wheeling; Covering design）。",
            "公开GitHub彩票/Keno预测项目多为样本内演示；严格walk-forward后普遍回归基线（本项目39项扫描亦无CI下界>25%）。",
        ],
    },
    "conclusion": conclusion,
}

jp = os.path.join(OUT, "candidates_eval_final_20261110.json")
with open(jp, "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=1)

lines = ["候选方法严格 walk-forward 评估 · 最终诊断（2026-11-10）", "=" * 90]
for e in final_exps:
    lines.append(f"{e['name']:28s} n={e['n']:4d} mean={e['mean']:.4f} base={e['base']:.4f} "
                 f"lift={e['lift']:.3f} CI[{e['ci_lo']:.4f},{e['ci_hi']:.4f}] p={e['p']:.4g} "
                 f"beats={e['beats_baseline']}")
    lines.append(f"    {e['notes']}")
lines += ["", "【窗口选择诊断】", json.dumps(window_diag, ensure_ascii=False, indent=1),
          "", "【C 稳健性】", json.dumps(c_rob["eval_299"], ensure_ascii=False, indent=1),
          json.dumps(c_rob["all_point_periods_334"], ensure_ascii=False, indent=1),
          "", conclusion]
tp = os.path.join(OUT, "candidates_eval_final_20261110.txt")
with open(tp, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"[saved] {jp}")
print(f"[saved] {tp}")
print(f"[entries] {len(final_exps)}")
for e in final_exps:
    print(f"  {e['name']:28s} lift={e['lift']:.3f} p={e['p']:.4g} beats={e['beats_baseline']}")