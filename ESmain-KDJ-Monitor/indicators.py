"""
KDJ 技术指标计算模块
====================

参考自 KDJ-Pattern-Scanner_V3/indicators.py，针对单一标的（ESmain）做了简化。

KDJ 公式（标准 9-3-3 参数）:
    RSV  = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) * 100
    K(t) = 2/3 * K(t-1) + 1/3 * RSV(t)
    D(t) = 2/3 * D(t-1) + 1/3 * K(t)
    J(t) = 3*K(t) - 2*D(t)

其中 N = 9（默认周期）。K、D 初始值为 50。
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """计算 K、D、J 三条线。

    参数
    ----
    df : DataFrame，必须包含列 ['High', 'Low', 'Close']。
    period : int，KDJ 的回看周期，默认 9。

    返回
    ----
    带新增列 ['K', 'D', 'J'] 的 DataFrame 副本。

    暖机约定（与主项目一致，默认数值兼容旧实现）：
      K/D 初值为 50；RSV 在前 period-1 根尚未定义时不更新 K/D（保持 50）。
      不再对缺失 RSV 先 fillna(50) 再迭代。对无中段 NaN 的干净序列，结果与旧实现等价。
      这是本仓库约定，不是唯一市场标准。
    """
    df = df.copy()
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()

    denom = (high_max - low_min)
    rsv = (df['Close'] - low_min) / denom * 100
    rsv = rsv.mask(denom == 0, 50)
    rsv = rsv.replace([np.inf, -np.inf], np.nan)

    k_list, d_list = [], []
    k, d = 50.0, 50.0
    for r in rsv:
        if np.isnan(r):
            k_list.append(k)
            d_list.append(d)
        else:
            k = (2 / 3) * k + (1 / 3) * r
            d = (2 / 3) * d + (1 / 3) * k
            k_list.append(k)
            d_list.append(d)

    df['K'] = k_list
    df['D'] = d_list
    df['J'] = 3 * df['K'] - 2 * df['D']
    return df


def daily_to_weekly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """日线 OHLC 重采样到周线（W-FRI 收盘）。"""
    df = df_daily.copy()
    df.index = pd.to_datetime(df.index)
    agg = {
        'Open':  'first',
        'High':  'max',
        'Low':   'min',
        'Close': 'last',
    }
    if 'Volume' in df.columns:
        agg['Volume'] = 'sum'
    weekly = df.resample('W-FRI').agg(agg).dropna(subset=['Close'])
    return weekly


def daily_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    """日线 OHLC 重采样到月线（自然月最后一个交易日）。"""
    df = df_daily.copy()
    df.index = pd.to_datetime(df.index)
    agg = {
        'Open':  'first',
        'High':  'max',
        'Low':   'min',
        'Close': 'last',
    }
    if 'Volume' in df.columns:
        agg['Volume'] = 'sum'
    monthly = df.resample('ME').agg(agg).dropna(subset=['Close'])
    return monthly
