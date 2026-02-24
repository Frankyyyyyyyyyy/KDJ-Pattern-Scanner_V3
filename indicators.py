import pandas as pd
import numpy as np
from datetime import timedelta

def calculate_kdj(df, period=9):
    """Calculates K, D, J values."""
    df = df.copy()
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()

    denom = (high_max - low_min)
    rsv = (df['Close'] - low_min) / denom * 100
    rsv = rsv.mask(denom == 0, 50)
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50)

    k_list, d_list = [], []
    k, d = 50, 50

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


def calculate_atr(df, period=14):
    """
    计算 Average True Range (ATR)。

    ATR = 14日 True Range 的指数移动平均
    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    """
    df = df.copy()
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    return df


def get_weekly_kdj_snapshot(daily_df, current_date, weekly_kdj_df):
    """
    Simulates the 'Running' Weekly KDJ for a specific date (avoiding lookahead bias).
    """
    current_weekday = current_date.weekday()
    week_start = current_date - timedelta(days=current_weekday)

    prior_weeks = weekly_kdj_df[weekly_kdj_df.index < week_start]
    if prior_weeks.empty:
        prev_k, prev_d = 50, 50
    else:
        prev_k = prior_weeks.iloc[-1]['K']
        prev_d = prior_weeks.iloc[-1]['D']

    mask = (daily_df.index >= week_start) & (daily_df.index <= current_date)
    week_so_far = daily_df.loc[mask]

    if week_so_far.empty:
        prev_j = 3 * prev_k - 2 * prev_d
        return prev_k, prev_d, prev_j

    curr_high = week_so_far['High'].max()
    curr_low = week_so_far['Low'].min()
    curr_close = week_so_far['Close'].iloc[-1]

    last_8_weeks = prior_weeks.iloc[-8:]
    if not last_8_weeks.empty:
        prev_high_9 = last_8_weeks['High'].max()
        prev_low_9 = last_8_weeks['Low'].min()
        period_high = max(prev_high_9, curr_high) if not pd.isna(prev_high_9) else curr_high
        period_low = min(prev_low_9, curr_low) if not pd.isna(prev_low_9) else curr_low
    else:
        period_high, period_low = curr_high, curr_low

    if period_high == period_low:
        rsv = 50
    else:
        rsv = (curr_close - period_low) / (period_high - period_low) * 100

    k = (2 / 3) * prev_k + (1 / 3) * rsv
    d = (2 / 3) * prev_d + (1 / 3) * k
    j = 3 * k - 2 * d

    return k, d, j
