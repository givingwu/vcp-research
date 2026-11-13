# 真实美股回测执行状态（2026-11-13）

**状态：未完成真实市场回测。**

已完成：冻结 20 只美股 + SPY 样本名单；Yahoo Finance 历史日线采集脚本；数据覆盖与完整性门禁；2010–2019 训练、2020–2022 验证、2023–2025 样本外测试；SPY 买入持有对照；报告生成脚本；6 项自动化测试通过。

阻塞：本运行环境尝试访问 `stooq.com` 与 `query1.finance.yahoo.com`，均因 DNS / 外部网络不可用而失败。没有得到任何历史美股 K 线，因此不存在真实股票的收益率、最大回撤、Sharpe 或交易笔数结果。

运行顺序：

```
pip install -e . pytest yfinance
pytest -q
python fetch_real.py --start 2010-01-01 --end 2026-01-01
python run_real.py
```

若成功，`data_real/manifest.json` 记录实际获取的股票及日期范围，`results_real/REPORT.md` 和 `summary.json` 才会产生真实数据结果。任何缺失数据都会导致失败退出，不会退回随机模拟数据。

研究限制：2026 年选定的 20 只存续股存在幸存者偏差；可作为探索但不可做正式策略有效性证明。数据来源为 Yahoo Finance，复权 OHLCV 仍须审计；视频主观 VCP 不等于这里的算法近似实现。