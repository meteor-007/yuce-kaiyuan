产出目录（与代码/数据源分离，避免根目录膨胀）

  reports/   日常分析报告
             分析报告_YYYYMMDD.txt        标准版（六节正文）
             分析报告_YYYYMMDD_详细.txt   详细版（正文 + 附录A-D）

  backtest/  回测与调权缓存
             kl8_backtest_result.json
             kl8_H_stash.npz

  diagnostics/  条件命中率全扫描等只读诊断
                edge_scan_latest.txt / .json
                （运行: python kl8_edge_scan.py）

数据源仍在项目根目录：daily_points.txt、kl8_history_final.txt
