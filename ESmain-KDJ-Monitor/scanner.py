"""
ESmain KDJ 信号扫描器
=====================

触发：日线 J 必须满足。
强度：根据周线 / 月线是否同时满足，分 HIGH / MEDIUM / LOW 三档。

防 lookahead-bias:
    每个评估日 T 只引用 *严格早于* T 的已完成 K 线
    （pd.merge_asof allow_exact_matches=False；日线则用 .shift(1)）。
"""

from __future__ import annotations
import argparse
import os

import pandas as pd

from indicators import calculate_kdj, daily_to_weekly, daily_to_monthly
from data_fetcher import fetch_esmain_daily
from signal_strength import classify, SignalStrength
from config import THRESHOLDS, OUTPUT_DIR, DATA_SOURCE_PRIORITY, logger


# ----------------------------------------------------------------------
# 核心：构造每日诊断面板
# ----------------------------------------------------------------------
def build_signal_panel(daily: pd.DataFrame,
                       monthly_j_thr: float = THRESHOLDS.monthly_j,
                       weekly_j_thr: float = THRESHOLDS.weekly_j,
                       daily_j_thr: float = THRESHOLDS.daily_j) -> pd.DataFrame:
    """对每个日线交易日 T 注入三条 J 值 + 三档强度标记。"""
    if daily.empty:
        return daily

    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index)
    daily.index.name = "Date"

    # 日 / 周 / 月 各自 KDJ
    daily_kdj = calculate_kdj(daily).rename(
        columns={"K": "DA_K", "D": "DA_D", "J": "DA_J"})
    weekly_kdj = calculate_kdj(daily_to_weekly(daily)).rename(
        columns={"K": "WK_K", "D": "WK_D", "J": "WK_J"})
    weekly_kdj.index.name = "Date"
    monthly_kdj = calculate_kdj(daily_to_monthly(daily)).rename(
        columns={"K": "MO_K", "D": "MO_D", "J": "MO_J"})
    monthly_kdj.index.name = "Date"

    # 严格只引用早于 T 的已完成 K 线
    daily_idx = daily_kdj.reset_index().sort_values("Date").reset_index(drop=True)
    wk = weekly_kdj.reset_index().rename(columns={"Date": "WK_Date"}).sort_values("WK_Date")
    mo = monthly_kdj.reset_index().rename(columns={"Date": "MO_Date"}).sort_values("MO_Date")

    merged = pd.merge_asof(
        daily_idx, wk[["WK_Date", "WK_J"]],
        left_on="Date", right_on="WK_Date",
        direction="backward", allow_exact_matches=False,
    )
    merged = pd.merge_asof(
        merged, mo[["MO_Date", "MO_J"]],
        left_on="Date", right_on="MO_Date",
        direction="backward", allow_exact_matches=False,
    )

    # 日线 J 也用前一交易日的值
    merged["DA_J_prev"] = merged["DA_J"].shift(1)

    # 三个布尔 OK + 强度
    def _row_strength(r):
        c = classify(r["DA_J_prev"], r["WK_J"], r["MO_J"],
                     daily_j_thr, weekly_j_thr, monthly_j_thr)
        return pd.Series({
            "DA_OK": c.daily_ok,
            "WK_OK": c.weekly_ok,
            "MO_OK": c.monthly_ok,
            "Strength": c.strength.value,
            "Signal": c.triggers,
        })

    flags = merged.apply(_row_strength, axis=1)
    merged = pd.concat([merged, flags], axis=1)

    return merged


# ----------------------------------------------------------------------
# 把"连续信号天"折叠成事件
# ----------------------------------------------------------------------
def extract_signal_episodes(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["Signal"] = df["Signal"].astype(bool)
    if df["Signal"].sum() == 0:
        return pd.DataFrame()

    # 一个区段内可能强度变化（LOW → MEDIUM → HIGH 或反之），
    # 区段以"Signal True"的连续段定义；强度取该区段内最高档。
    grp = (df["Signal"] != df["Signal"].shift(1)).cumsum()
    out = []
    for _, sub in df[df["Signal"]].groupby(grp[df["Signal"]]):
        first, last = sub.iloc[0], sub.iloc[-1]
        # 区段内峰值强度
        peak_strength = max(
            (SignalStrength(s) for s in sub["Strength"]),
            key=lambda s: s.rank,
        )
        out.append({
            "Episode_Start":   first["Date"].strftime("%Y-%m-%d"),
            "Episode_End":     last["Date"].strftime("%Y-%m-%d"),
            "Days":            len(sub),
            "Peak_Strength":   peak_strength.value,
            "Start_Strength":  first["Strength"],
            "Start_Close":     round(float(first["Close"]), 2),
            "End_Close":       round(float(last["Close"]), 2),
            "Start_DA_J":      round(float(first["DA_J_prev"]), 2),
            "Min_DA_J":        round(float(sub["DA_J_prev"].min()), 2),
            "Start_WK_J":      round(float(first["WK_J"]), 2) if pd.notna(first["WK_J"]) else None,
            "Min_WK_J":        round(float(sub["WK_J"].min()), 2) if sub["WK_J"].notna().any() else None,
            "Start_MO_J":      round(float(first["MO_J"]), 2) if pd.notna(first["MO_J"]) else None,
            "Min_MO_J":        round(float(sub["MO_J"].min()), 2) if sub["MO_J"].notna().any() else None,
            "Ref_Weekly_Bar":  first["WK_Date"].strftime("%Y-%m-%d") if pd.notna(first["WK_Date"]) else "",
            "Ref_Monthly_Bar": first["MO_Date"].strftime("%Y-%m-%d") if pd.notna(first["MO_Date"]) else "",
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 命令行
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ESmain KDJ 历史扫描")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--monthly-j", type=float, default=THRESHOLDS.monthly_j)
    parser.add_argument("--weekly-j", type=float, default=THRESHOLDS.weekly_j)
    parser.add_argument("--daily-j", type=float, default=THRESHOLDS.daily_j)
    parser.add_argument("--source", default=",".join(DATA_SOURCE_PRIORITY))
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--min-strength", choices=["LOW", "MEDIUM", "HIGH"],
                        default="LOW",
                        help="只输出大于等于此档的信号")
    args = parser.parse_args()

    end_date = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = args.start or (pd.Timestamp(end_date) - pd.DateOffset(years=11)).strftime("%Y-%m-%d")

    logger.info(f"扫描区间: {start_date} → {end_date}")
    logger.info(f"阈值:     日J<{args.daily_j} | 周J<{args.weekly_j} | 月J<{args.monthly_j}")
    logger.info(f"最低强度: {args.min_strength}")

    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    daily = fetch_esmain_daily(start_date, end_date, source_priority=sources)
    if daily.empty:
        logger.error("没有任何数据源能拉到数据")
        return

    panel = build_signal_panel(
        daily, args.monthly_j, args.weekly_j, args.daily_j)
    window_start = pd.Timestamp(end_date) - pd.DateOffset(years=10)
    panel_10y = panel[panel["Date"] >= window_start].copy()

    os.makedirs(args.out_dir, exist_ok=True)
    panel_path = os.path.join(args.out_dir, "esmain_kdj_panel_daily.csv")
    panel_10y[["Date", "Close", "DA_J_prev", "WK_Date", "WK_J",
               "MO_Date", "MO_J", "DA_OK", "WK_OK", "MO_OK",
               "Strength", "Signal"]] \
        .to_csv(panel_path, index=False, date_format="%Y-%m-%d")

    episodes = extract_signal_episodes(panel_10y)

    # 强度过滤
    if not episodes.empty and args.min_strength != "LOW":
        min_rank = SignalStrength(args.min_strength).rank
        episodes = episodes[
            episodes["Peak_Strength"].map(
                lambda s: SignalStrength(s).rank) >= min_rank
        ].reset_index(drop=True)

    sig_path = os.path.join(args.out_dir, "esmain_kdj_signals.csv")
    episodes.to_csv(sig_path, index=False)

    # 按强度统计
    by_strength = (episodes["Peak_Strength"].value_counts().to_dict()
                   if not episodes.empty else {})

    summary_path = os.path.join(args.out_dir, "esmain_kdj_summary.txt")
    src = daily.attrs.get("source", "unknown")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ESmain KDJ 扫描汇总\n" + "=" * 60 + "\n")
        f.write(f"数据源:        {src}\n")
        f.write(f"扫描区间:      {window_start.strftime('%Y-%m-%d')} → {end_date}\n")
        f.write(f"日线行数:      {len(panel_10y)}\n")
        f.write(f"信号天数:      {int(panel_10y['Signal'].sum())}\n")
        f.write(f"信号事件数:    {len(episodes)}\n")
        f.write(f"  强 HIGH:    {by_strength.get('HIGH', 0)}\n")
        f.write(f"  中 MEDIUM:  {by_strength.get('MEDIUM', 0)}\n")
        f.write(f"  弱 LOW:     {by_strength.get('LOW', 0)}\n")
        f.write(f"阈值:          日J<{args.daily_j} | 周J<{args.weekly_j} | 月J<{args.monthly_j}\n\n")
        if not episodes.empty:
            f.write(episodes.to_string(index=False))
        else:
            f.write("（无）\n")

    logger.info(f"日线诊断面板:   {panel_path}")
    logger.info(f"信号事件清单:   {sig_path}")
    logger.info(f"汇总:           {summary_path}")
    logger.info(f"过去 10 年共 {len(episodes)} 次事件 "
                f"(强:{by_strength.get('HIGH', 0)} / "
                f"中:{by_strength.get('MEDIUM', 0)} / "
                f"弱:{by_strength.get('LOW', 0)})")

    if not episodes.empty:
        print("\n=== 信号事件清单 ===")
        cols = ["Episode_Start", "Episode_End", "Days", "Peak_Strength",
                "Start_Close", "Min_DA_J", "Min_WK_J", "Min_MO_J"]
        print(episodes[cols].to_string(index=False))


if __name__ == "__main__":
    main()
