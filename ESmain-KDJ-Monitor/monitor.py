"""
每日运行入口（cron / GitHub Actions）
====================================

流程：
    1. 拉取最近 ~14 个月数据（KDJ 暖机够用）
    2. 计算今日（最后一个交易日）的三条 J 值
    3. 判定强度：
         HIGH   月+周+日 三确认
         MEDIUM 日+周 或 日+月 两确认
         LOW    仅日线
    4. 只要日线满足（即 LOW 及以上），就发邮件
    5. 同时把每次告警追加到 logs/alerts.csv，方便日后回溯

直接运行：
    python monitor.py
    python monitor.py --dry-run                # 只判定不发邮件
    python monitor.py --min-strength MEDIUM    # 弱信号不发，仅 MEDIUM/HIGH 发
"""

from __future__ import annotations
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import notifier
from config import THRESHOLDS, DATA_SOURCE_PRIORITY, EMAIL, LOG_DIR, logger
from data_fetcher import fetch_esmain_daily
from scanner import build_signal_panel
from signal_strength import SignalStrength

ALERT_LOG = LOG_DIR / "alerts.csv"


def _append_alert_log(row: dict) -> None:
    new_file = not ALERT_LOG.exists()
    with ALERT_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="只判定不发邮件、不写 alert 日志")
    parser.add_argument("--source", default=",".join(DATA_SOURCE_PRIORITY))
    parser.add_argument("--min-strength", choices=["LOW", "MEDIUM", "HIGH"],
                        default="LOW",
                        help="低于此档不发邮件（默认 LOW，即只要日线满足都发）")
    args = parser.parse_args()

    today = pd.Timestamp.today().normalize()
    start = (today - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    daily = fetch_esmain_daily(start, end, source_priority=sources)
    if daily.empty:
        logger.error("数据获取失败，退出")
        sys.exit(1)

    panel = build_signal_panel(
        daily,
        monthly_j_thr=THRESHOLDS.monthly_j,
        weekly_j_thr=THRESHOLDS.weekly_j,
        daily_j_thr=THRESHOLDS.daily_j,
    )
    if panel.empty:
        logger.error("无法构造诊断面板")
        sys.exit(1)

    last = panel.iloc[-1]
    eval_date = last["Date"]
    strength = SignalStrength(last["Strength"])
    triggered = bool(last["Signal"])

    logger.info(
        f"评估日 {eval_date.date()}  "
        f"日J(prev)={last['DA_J_prev']:.2f}  "
        f"周J={last['WK_J']:.2f}  月J={last['MO_J']:.2f}  → "
        f"Signal={triggered}  Strength={strength.value}"
    )

    if not triggered:
        logger.info("今日不满足触发条件，无邮件")
        return

    # 强度过滤
    min_strength = SignalStrength(args.min_strength)
    if strength.rank < min_strength.rank:
        logger.info(f"信号强度 {strength.value} 低于 --min-strength={args.min_strength}，跳过邮件")
        return

    # 准备邮件
    da_thr = THRESHOLDS.daily_j
    wk_thr = THRESHOLDS.weekly_j
    mo_thr = THRESHOLDS.monthly_j

    da_v = float(last["DA_J_prev"])
    wk_v = float(last["WK_J"]) if pd.notna(last["WK_J"]) else float("nan")
    mo_v = float(last["MO_J"]) if pd.notna(last["MO_J"]) else float("nan")

    fields = {
        "信号强度":          strength.label_cn,
        "评估日":           eval_date.strftime("%Y-%m-%d"),
        "ESmain 收盘":      f"{last['Close']:.2f}",
        "日线 J（前一日）":  f"{da_v:+.2f}  vs 阈值 < {da_thr}  → {'✓' if last['DA_OK'] else '✗'}",
        "周线 J":           f"{wk_v:+.2f}  vs 阈值 < {wk_thr}  → {'✓' if last['WK_OK'] else '✗'}",
        "月线 J":           f"{mo_v:+.2f}  vs 阈值 < {mo_thr}  → {'✓' if last['MO_OK'] else '✗'}",
        "周线参考 K 线":    pd.Timestamp(last['WK_Date']).strftime("%Y-%m-%d") if pd.notna(last['WK_Date']) else "-",
        "月线参考 K 线":    pd.Timestamp(last['MO_Date']).strftime("%Y-%m-%d") if pd.notna(last['MO_Date']) else "-",
        "数据源":           daily.attrs.get("source", "unknown"),
    }
    note = (
        f"信号分级规则：\n"
        f"  强 HIGH   = 月+周+日 三个时间框架同时超卖\n"
        f"  中 MEDIUM = 日+周 或 日+月 两个时间框架同时超卖\n"
        f"  弱 LOW    = 仅日线超卖（高时间框架未确认）\n"
        f"\n本次为 {strength.label_cn}。"
    )

    if args.dry_run:
        logger.info("DRY-RUN：不发邮件，不写日志")
        for k, v in fields.items():
            print(f"  {k}: {v}")
        return

    if not EMAIL.enabled:
        logger.warning("邮件未配置（SENDER/RECEIVER/PASSWORD），但信号已触发！")
        sys.exit(2)

    subject = f"[ESmain KDJ] {strength.value} 信号 — {eval_date.strftime('%Y-%m-%d')}"
    ok = notifier.send(subject=subject, fields=fields, note=note,
                       strength_color=strength.color_hex)
    if not ok:
        logger.error("邮件发送失败")
        sys.exit(2)

    # 追加到 alerts.csv 日志
    _append_alert_log({
        "alert_ts_utc":  datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "eval_date":     eval_date.strftime("%Y-%m-%d"),
        "strength":      strength.value,
        "close":         f"{last['Close']:.2f}",
        "daily_j_prev":  f"{da_v:.2f}",
        "weekly_j":      f"{wk_v:.2f}",
        "monthly_j":     f"{mo_v:.2f}",
        "data_source":   daily.attrs.get("source", "unknown"),
    })
    logger.info(f"已写入告警日志 {ALERT_LOG}")


if __name__ == "__main__":
    main()
