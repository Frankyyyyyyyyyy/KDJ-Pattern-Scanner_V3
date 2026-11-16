"""
信号回测工具
============

入场规则:
    每个信号事件的 Episode_Start 当日**收盘价**入场（多头）。

平仓规则:
    入场后第 N 个交易日的**收盘价**平仓（默认 N=15）。
    若数据不够，事件被跳过。

输出:
    1) 单笔交易明细 CSV
    2) 按入场时可知强度（Start_Strength）分组的统计。
       不使用区间事后峰值强度（Peak_Strength），以免把后来才升级的交易算进更高档。

用法:
    python backtest.py                     # 全部 + 默认 15 天
    python backtest.py --strength LOW      # 只回测 LOW
    python backtest.py --hold-days 20      # 改为持仓 20 天
    python backtest.py --no-overlap        # 同一笔未平仓前不开新仓
"""

from __future__ import annotations
import argparse

import numpy as np
import pandas as pd

from data_fetcher import fetch_esmain_daily
from scanner import build_signal_panel, extract_signal_episodes
from signal_strength import SignalStrength
from config import THRESHOLDS, OUTPUT_DIR, DATA_SOURCE_PRIORITY, logger


def backtest_episodes(daily: pd.DataFrame,
                      episodes: pd.DataFrame,
                      hold_days: int,
                      no_overlap: bool = False) -> pd.DataFrame:
    """
    返回一张 trade-by-trade 表：
        Entry_Date / Entry_Close / Exit_Date / Exit_Close
        Hold_Days / Return_Pct / Strength
    """
    if episodes.empty:
        return pd.DataFrame()

    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index)
    trading_days = daily.index.sort_values()

    rows = []
    last_exit = pd.Timestamp.min
    skipped_overlap = 0
    skipped_insufficient_data = 0

    for _, ep in episodes.iterrows():
        entry_date = pd.Timestamp(ep["Episode_Start"])
        if entry_date not in trading_days:
            # 用第一个 ≥ entry_date 的交易日
            future = trading_days[trading_days >= entry_date]
            if future.empty:
                continue
            entry_date = future[0]

        if no_overlap and entry_date < last_exit:
            skipped_overlap += 1
            continue

        # 取入场日在 trading_days 中的位置，往后数 hold_days
        idx = trading_days.get_loc(entry_date)
        if idx + hold_days >= len(trading_days):
            skipped_insufficient_data += 1
            continue

        exit_date = trading_days[idx + hold_days]
        entry_close = float(daily.loc[entry_date, "Close"])
        exit_close = float(daily.loc[exit_date, "Close"])
        ret_pct = (exit_close - entry_close) / entry_close * 100

        rows.append({
            "Entry_Date":  entry_date.strftime("%Y-%m-%d"),
            "Entry_Close": round(entry_close, 2),
            "Exit_Date":   exit_date.strftime("%Y-%m-%d"),
            "Exit_Close":  round(exit_close, 2),
            "Hold_Days":   hold_days,
            "Return_Pct":  round(ret_pct, 3),
            "Strength":    ep.get("Start_Strength", ep.get("Peak_Strength")),
            "Peak_Strength": ep.get("Peak_Strength"),
            "Episode_Days": int(ep["Days"]),
            "Min_DA_J":    ep.get("Min_DA_J"),
            "Min_WK_J":    ep.get("Min_WK_J"),
            "Min_MO_J":    ep.get("Min_MO_J"),
        })
        last_exit = exit_date

    if skipped_overlap:
        logger.info(f"--no-overlap 跳过 {skipped_overlap} 笔（持仓期内）")
    if skipped_insufficient_data:
        logger.info(f"数据不足跳过 {skipped_insufficient_data} 笔（区段末尾）")

    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, group: str = "Strength") -> pd.DataFrame:
    """按 group 分组的统计。"""
    if trades.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        r = g["Return_Pct"]
        return pd.Series({
            "N":           len(g),
            "WinRate_%":   round((r > 0).mean() * 100, 2),
            "Mean_%":      round(r.mean(), 3),
            "Median_%":    round(r.median(), 3),
            "Std_%":       round(r.std(), 3),
            "Min_%":       round(r.min(), 3),
            "P25_%":       round(r.quantile(0.25), 3),
            "P75_%":       round(r.quantile(0.75), 3),
            "Max_%":       round(r.max(), 3),
            "Sharpe(naive)": round(r.mean() / r.std(), 3) if r.std() > 0 else float("nan"),
        })

    g = trades.groupby(group, dropna=False)[["Return_Pct"]].apply(_agg).reset_index()
    # 强度排序
    rank_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "ALL": 4}
    g = g.sort_values(group, key=lambda s: s.map(lambda x: rank_map.get(str(x), 99)))
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None,
                        help="扫描开始日期 YYYY-MM-DD（默认 11 年前）")
    parser.add_argument("--end", default=None,
                        help="扫描结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--hold-days", type=int, default=15,
                        help="持仓交易日数，默认 15")
    parser.add_argument("--strength", choices=["LOW", "MEDIUM", "HIGH", "ALL"],
                        default="ALL", help="只回测某档强度（默认 ALL）")
    parser.add_argument("--no-overlap", action="store_true",
                        help="持仓期内不开新仓（保守，串行回测）")
    parser.add_argument("--source", default=",".join(DATA_SOURCE_PRIORITY))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = args.start or (pd.Timestamp(end_date) - pd.DateOffset(years=11)).strftime("%Y-%m-%d")

    logger.info(f"扫描区间: {start_date} → {end_date}")
    logger.info(f"持仓:     {args.hold_days} 个交易日")
    logger.info(f"强度筛选: {args.strength}  no-overlap={args.no_overlap}")

    daily = fetch_esmain_daily(start_date, end_date,
                               source_priority=[s.strip() for s in args.source.split(",")])
    if daily.empty:
        logger.error("数据获取失败")
        return

    panel = build_signal_panel(
        daily,
        monthly_j_thr=THRESHOLDS.monthly_j,
        weekly_j_thr=THRESHOLDS.weekly_j,
        daily_j_thr=THRESHOLDS.daily_j,
    )
    window_start = pd.Timestamp(end_date) - pd.DateOffset(years=10)
    panel_10y = panel[panel["Date"] >= window_start].copy()
    episodes = extract_signal_episodes(panel_10y)

    if args.strength != "ALL":
        strength_col = "Start_Strength" if "Start_Strength" in episodes.columns else "Peak_Strength"
        if strength_col != "Start_Strength":
            logger.warning("episodes 缺少 Start_Strength，回退到 Peak_Strength（可能含前视）")
        episodes = episodes[episodes[strength_col] == args.strength]
        logger.info(f"按入场强度过滤至 {args.strength} 后剩 {len(episodes)} 个事件")

    trades = backtest_episodes(daily, episodes, args.hold_days,
                               no_overlap=args.no_overlap)
    if trades.empty:
        logger.warning("没有可回测的交易")
        return

    # 输出
    suffix = f"{args.strength}_h{args.hold_days}"
    if args.no_overlap:
        suffix += "_noov"
    trades_path = f"{args.out_dir}/backtest_trades_{suffix}.csv"
    summary_path = f"{args.out_dir}/backtest_summary_{suffix}.csv"
    trades.to_csv(trades_path, index=False)

    by_strength = summarize(trades, "Strength")
    overall = summarize(trades.assign(Strength="ALL"), "Strength")
    summary = pd.concat([by_strength, overall], ignore_index=True)
    summary.to_csv(summary_path, index=False)

    print("\n=== 按强度统计 ===")
    print(summary.to_string(index=False))
    print(f"\n所有交易明细已写入: {trades_path}")
    print(f"统计摘要已写入:     {summary_path}")


if __name__ == "__main__":
    main()
