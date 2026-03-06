#!/usr/bin/env python3
"""
build_dashboard.py — 合并信号 CSV 并同步 dashboard
=====================================================
GitHub Actions 每天扫描完成后调用此脚本：
  1. 合并当前季度 + archive 目录下的历史 CSV
  2. 输出 signals_latest.csv（供 index.html 通过 fetch 加载）

index.html 通过 JavaScript fetch('signals_latest.csv') 动态加载数据，
无需将 CSV 嵌入 HTML 文件中。

Usage:
    python build_dashboard.py
"""

import glob
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_all_signal_csvs():
    """找到当前季度 + archive 下所有历史 CSV，按文件名排序"""
    csvs = []
    # 当前目录下的季度 CSV
    csvs += glob.glob(os.path.join(BASE_DIR, "signals_*Q*.csv"))
    # archive 目录下的历史 CSV
    csvs += glob.glob(os.path.join(BASE_DIR, "archive", "signals_*Q*.csv"))
    return sorted(set(csvs))


def merge_all_csvs(csv_files):
    """合并所有 CSV 文件，去重并排序，返回 DataFrame"""
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            frames.append(df)
            print(f"  ✓ {os.path.basename(f)}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)}: {e}")

    if not frames:
        return None

    df_all = pd.concat(frames, ignore_index=True)

    # 统一日期格式并去重
    df_all["Date (日期)"] = pd.to_datetime(df_all["Date (日期)"]).dt.strftime("%Y-%m-%d")
    df_all = df_all.drop_duplicates(
        subset=["Date (日期)", "Ticker (股票代码)", "Direction (方向)"],
        keep="last",
    )

    # 排序：按方向、类型、代码、日期
    df_all = df_all.sort_values(
        by=["Direction (方向)", "Type (类型)", "Ticker (股票代码)", "Date (日期)"],
        ascending=[False, True, True, False],
    )

    print(f"\n  合并后总计 Total: {len(df_all)} signals")
    return df_all


def main():
    print("=" * 50)
    print("Build Dashboard — 重建仪表盘数据")
    print("=" * 50)

    # 1. Find all CSVs
    print("\n[1/2] 查找信号文件 Finding signal files...")
    csv_files = find_all_signal_csvs()

    if not csv_files:
        print("  ⚠ 未找到任何信号 CSV No signal CSVs found")
        return

    # 2. Merge CSVs → signals_latest.csv
    print(f"\n[2/2] 合并 {len(csv_files)} 个文件 Merging...")
    df_all = merge_all_csvs(csv_files)

    if df_all is None or df_all.empty:
        print("\n⚠ No data to write")
        return

    latest_csv_path = os.path.join(BASE_DIR, "signals_latest.csv")
    df_all.to_csv(latest_csv_path, index=False, encoding="utf-8")
    print(f"  signals_latest.csv written ({len(df_all)} signals)")

    print("\n✓ Done!")
    print(f"  index.html 通过 fetch('signals_latest.csv') 动态加载数据")


if __name__ == "__main__":
    main()
