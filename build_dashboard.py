#!/usr/bin/env python3
"""
build_dashboard.py — 将最新信号 CSV 嵌入 dashboard.html
==========================================================
GitHub Actions 每天扫描完成后调用此脚本，
把 signals_YYYYQN.csv 的数据写进 dashboard.html 的 JS 变量中，
这样 GitHub Pages 部署后用户打开网页就能直接看到最新信号。

同时也会合并 archive/ 目录下的历史季度 CSV，
让 dashboard "全部 All" 模式下可以看到跨季度的完整历史。

Usage:
    python build_dashboard.py
"""

import glob
import os
import re
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
    """合并所有 CSV 文件，去重并排序"""
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            frames.append(df)
            print(f"  ✓ {os.path.basename(f)}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)}: {e}")

    if not frames:
        return ""

    df_all = pd.concat(frames, ignore_index=True)

    # 统一日期格式并去重
    df_all["Date (日期)"] = pd.to_datetime(df_all["Date (日期)"]).dt.strftime("%Y-%m-%d")
    df_all = df_all.drop_duplicates(
        subset=["Date (日期)", "Ticker (股票代码)", "Direction (方向)"],
        keep="last",
    )

    # 排序：按日期降序
    df_all = df_all.sort_values(
        by=["Direction (方向)", "Type (类型)", "Ticker (股票代码)", "Date (日期)"],
        ascending=[False, True, True, False],
    )

    print(f"\n  合并后总计 Total: {len(df_all)} signals")
    return df_all.to_csv(index=False, encoding="utf-8")


def embed_csv_into_html(csv_text):
    """将 CSV 文本嵌入 dashboard.html 的 JS 变量"""
    html_path = os.path.join(BASE_DIR, "dashboard.html")

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 转义 JS 模板字符串特殊字符
    csv_escaped = csv_text.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    # 替换 CSV_DATA 变量
    html_new = re.sub(
        r"const CSV_DATA = `.*?`;",
        f"const CSV_DATA = `{csv_escaped}`;",
        html,
        flags=re.DOTALL,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_new)

    print(f"  dashboard.html updated ({len(html_new):,} bytes)")


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

    # 2. Merge and embed
    print(f"\n[2/2] 合并 {len(csv_files)} 个文件并嵌入 Merging & embedding...")
    csv_text = merge_all_csvs(csv_files)

    if csv_text:
        embed_csv_into_html(csv_text)
        print("\n✓ Done!")
    else:
        print("\n⚠ No data to embed")


if __name__ == "__main__":
    main()
