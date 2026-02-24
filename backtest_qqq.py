#!/usr/bin/env python3
"""
Backtest: Check if KDJ Pattern Scanner can detect Short signals
during specific QQQ decline periods.
"""
import sys
import os

# Add project to path
sys.path.insert(0, '/sessions/great-eager-sagan/mnt/KDJ-Pattern-Scanner_V2')
os.chdir('/sessions/great-eager-sagan/mnt/KDJ-Pattern-Scanner_V2')

import pandas as pd
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

from config import STRATEGY_CONFIG
from indicators import calculate_kdj, calculate_atr, get_weekly_kdj_snapshot
from patterns import identify_patterns
from filters import check_j_filter, check_prior_trend, check_divergence, check_pattern_filter
import yfinance as yf

# ============================================
# Define test periods (QQQ declines)
# ============================================
TEST_PERIODS = [
    {
        'label': 'QQQ 2026-02-05 ~ 2026-02-11',
        'start': '2026-02-05',
        'end': '2026-02-11',
        'data_start': '2024-06-01',  # ~1.5yr lookback for indicators
        'data_end': '2026-02-15',
    },
    {
        'label': 'QQQ 2025-02-18 ~ 2025-02-20',
        'start': '2025-02-18',
        'end': '2025-02-20',
        'data_start': '2023-06-01',
        'data_end': '2025-02-25',
    },
    {
        'label': 'QQQ 2021-12-28 ~ 2021-12-31',
        'start': '2021-12-28',
        'end': '2021-12-31',
        'data_start': '2020-06-01',
        'data_end': '2022-01-10',
    },
    {
        'label': 'QQQ 2018-08-30 (single day)',
        'start': '2018-08-30',
        'end': '2018-08-30',
        'data_start': '2017-02-01',
        'data_end': '2018-09-10',
    },
    {
        'label': 'QQQ 2018-10-01 ~ 2018-10-03',
        'start': '2018-10-01',
        'end': '2018-10-03',
        'data_start': '2017-04-01',
        'data_end': '2018-10-10',
    },
]


def fetch_qqq_data(data_start, data_end):
    """Fetch QQQ data from yfinance."""
    logger.info(f"Fetching QQQ data: {data_start} ~ {data_end}")
    try:
        df = yf.download('QQQ', start=data_start, end=data_end, auto_adjust=True, progress=False)
        if df.empty:
            logger.error(f"Failed to fetch QQQ data for {data_start} ~ {data_end}")
            return df
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Ensure timezone-naive index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        logger.info(f"  Got {len(df)} rows, range: {df.index[0]} ~ {df.index[-1]}")
        return df
    except Exception as e:
        logger.error(f"yfinance error: {e}")
        return pd.DataFrame()


def run_signal_detection(df_raw, scan_start, scan_end, cfg):
    """
    Run the same signal detection logic as run_strategy.py
    but focused on a specific date range.
    Returns list of detected signals.
    """
    df = df_raw.copy()

    # Calculate indicators
    df = calculate_kdj(df)
    df = calculate_atr(df, period=cfg['atr_period'])
    df['Bearish_Patterns'], df['Bullish_Patterns'] = identify_patterns(df, j_values=df['J'])

    # Pre-calc MA
    df['MA10'] = df['Close'].rolling(cfg['ma_short']).mean()
    df['MA20'] = df['Close'].rolling(cfg['ma_long']).mean()

    # Weekly KDJ
    df_weekly = df.resample('W-FRI').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    ).dropna()
    df_weekly = calculate_kdj(df_weekly)

    # Scan window
    scan_start_ts = pd.Timestamp(scan_start)
    scan_end_ts = pd.Timestamp(scan_end)
    scan_window = df[(df.index >= scan_start_ts) & (df.index <= scan_end_ts)]

    if scan_window.empty:
        logger.warning(f"  No trading days in scan window {scan_start} ~ {scan_end}")
        # Try to show nearby dates
        nearby = df[(df.index >= scan_start_ts - pd.Timedelta(days=5)) &
                     (df.index <= scan_end_ts + pd.Timedelta(days=5))]
        if not nearby.empty:
            logger.info(f"  Nearby dates available: {nearby.index.tolist()}")
        return [], df, scan_window

    signals = []
    all_day_details = []

    for date, row in scan_window.iterrows():
        loc_idx = df.index.get_loc(date)
        daily_j = row['J']

        wk_k, wk_d, wk_j = get_weekly_kdj_snapshot(df, date, df_weekly)

        prev_daily_j = df.iloc[loc_idx - 1]['J'] if loc_idx > 0 else None
        prev_wk_idx = df_weekly.index[df_weekly.index < date]
        prev_wk_j = df_weekly.loc[prev_wk_idx[-1]]['J'] if not prev_wk_idx.empty else None

        # Day details for debugging
        day_info = {
            'date': date.strftime('%Y-%m-%d'),
            'close': round(row['Close'], 2),
            'open': round(row['Open'], 2),
            'high': round(row['High'], 2),
            'low': round(row['Low'], 2),
            'daily_j': round(daily_j, 2),
            'prev_daily_j': round(prev_daily_j, 2) if prev_daily_j is not None else None,
            'weekly_j': round(wk_j, 2),
            'prev_weekly_j': round(prev_wk_j, 2) if prev_wk_j is not None else None,
            'bearish_patterns': row.get('Bearish_Patterns', []),
            'bullish_patterns': row.get('Bullish_Patterns', []),
        }

        # Check Short direction
        direction = "Short"

        # Divergence
        divergence = check_divergence(direction, df, df, date, loc_idx)

        # J filter
        j_pass = check_j_filter(direction, daily_j, prev_daily_j, wk_j, prev_wk_j)

        # Prior trend
        trend_pass = check_prior_trend(direction, df, loc_idx)

        # Pattern filter
        valid_patterns = check_pattern_filter(direction, df, row, loc_idx)

        day_info['short_j_pass'] = j_pass
        day_info['short_trend_pass'] = trend_pass
        day_info['short_pattern_pass'] = len(valid_patterns) > 0
        day_info['short_patterns'] = valid_patterns
        day_info['short_divergence'] = divergence

        if j_pass and trend_pass and valid_patterns:
            strength = "Strong" if len(valid_patterns) > 0 else "Weak"
            if divergence:
                strength += " + Div"
            signals.append({
                'date': date.strftime('%Y-%m-%d'),
                'direction': 'Short',
                'price': round(row['Close'], 2),
                'daily_j': round(daily_j, 2),
                'weekly_j': round(wk_j, 2),
                'patterns': ", ".join(valid_patterns),
                'strength': strength,
                'divergence': divergence,
            })

        # Also check Long direction for completeness
        direction_l = "Long"
        divergence_l = check_divergence(direction_l, df, df, date, loc_idx)
        j_pass_l = check_j_filter(direction_l, daily_j, prev_daily_j, wk_j, prev_wk_j)
        trend_pass_l = check_prior_trend(direction_l, df, loc_idx)
        valid_patterns_l = check_pattern_filter(direction_l, df, row, loc_idx)

        day_info['long_j_pass'] = j_pass_l
        day_info['long_trend_pass'] = trend_pass_l
        day_info['long_pattern_pass'] = len(valid_patterns_l) > 0
        day_info['long_patterns'] = valid_patterns_l
        day_info['long_divergence'] = divergence_l

        if j_pass_l and trend_pass_l and valid_patterns_l:
            strength_l = "Strong"
            if divergence_l:
                strength_l += " + Div"
            signals.append({
                'date': date.strftime('%Y-%m-%d'),
                'direction': 'Long',
                'price': round(row['Close'], 2),
                'daily_j': round(daily_j, 2),
                'weekly_j': round(wk_j, 2),
                'patterns': ", ".join(valid_patterns_l),
                'strength': strength_l,
                'divergence': divergence_l,
            })

        all_day_details.append(day_info)

    return signals, df, all_day_details


def print_day_details(details):
    """Print detailed daily analysis for debugging."""
    for d in details:
        print(f"\n  📅 {d['date']}  O:{d['open']} H:{d['high']} L:{d['low']} C:{d['close']}")
        print(f"     Daily J: {d['daily_j']} (prev: {d['prev_daily_j']}),  Weekly J: {d['weekly_j']} (prev: {d['prev_weekly_j']})")
        if d['bearish_patterns']:
            print(f"     🔴 Bearish Patterns: {d['bearish_patterns']}")
        if d['bullish_patterns']:
            print(f"     🟢 Bullish Patterns: {d['bullish_patterns']}")

        # Short filters
        short_status = []
        short_status.append(f"J_Filter={'✅' if d['short_j_pass'] else '❌'}")
        short_status.append(f"Prior_Trend={'✅' if d['short_trend_pass'] else '❌'}")
        short_status.append(f"Pattern={'✅' if d['short_pattern_pass'] else '❌'}")
        if d['short_divergence']:
            short_status.append("Divergence=✅")
        print(f"     Short: {' | '.join(short_status)}")
        if d['short_patterns']:
            print(f"       Confirmed patterns: {d['short_patterns']}")

        # Long filters
        long_status = []
        long_status.append(f"J_Filter={'✅' if d['long_j_pass'] else '❌'}")
        long_status.append(f"Prior_Trend={'✅' if d['long_trend_pass'] else '❌'}")
        long_status.append(f"Pattern={'✅' if d['long_pattern_pass'] else '❌'}")
        if d['long_divergence']:
            long_status.append("Divergence=✅")
        print(f"     Long:  {' | '.join(long_status)}")


def main():
    cfg = STRATEGY_CONFIG.copy()

    print("=" * 80)
    print("KDJ Pattern Scanner V2 - Backtest: QQQ Decline Detection")
    print("=" * 80)

    results_summary = []

    for i, period in enumerate(TEST_PERIODS):
        print(f"\n{'='*80}")
        print(f"📊 Test {i+1}: {period['label']}")
        print(f"{'='*80}")

        # Fetch data
        df = fetch_qqq_data(period['data_start'], period['data_end'])
        if df.empty:
            print(f"  ❌ NO DATA - Cannot test this period")
            results_summary.append({
                'period': period['label'],
                'result': 'NO DATA',
                'signals': 0,
            })
            continue

        # Validate data
        from data import validate_ohlc
        df, removed = validate_ohlc(df, 'QQQ')

        # Show price action during the period
        scan_start_ts = pd.Timestamp(period['start'])
        scan_end_ts = pd.Timestamp(period['end'])
        period_data = df[(df.index >= scan_start_ts) & (df.index <= scan_end_ts)]

        if not period_data.empty:
            start_price = period_data['Close'].iloc[0]
            end_price = period_data['Close'].iloc[-1]
            pct_change = (end_price - start_price) / start_price * 100
            high_in_period = period_data['High'].max()
            low_in_period = period_data['Low'].min()
            print(f"\n  Price Action: {start_price:.2f} → {end_price:.2f} ({pct_change:+.2f}%)")
            print(f"  Period High: {high_in_period:.2f}, Low: {low_in_period:.2f}")

            # Also show a few days before for context
            context_start = scan_start_ts - pd.Timedelta(days=7)
            context = df[(df.index >= context_start) & (df.index <= scan_end_ts + pd.Timedelta(days=2))]
            print(f"\n  Price context (before & during):")
            for idx, r in context.iterrows():
                marker = " <<<" if scan_start_ts <= idx <= scan_end_ts else ""
                print(f"    {idx.strftime('%Y-%m-%d')}  O:{r['Open']:.2f} H:{r['High']:.2f} L:{r['Low']:.2f} C:{r['Close']:.2f}  Vol:{int(r['Volume']):,}{marker}")

        # Run signal detection
        signals, df_enriched, day_details = run_signal_detection(
            df, period['start'], period['end'], cfg
        )

        # Print detailed day-by-day analysis
        if day_details:
            print("\n  --- Day-by-Day Signal Analysis ---")
            print_day_details(day_details)

        # Summary
        short_signals = [s for s in signals if s['direction'] == 'Short']
        long_signals = [s for s in signals if s['direction'] == 'Long']

        print(f"\n  🎯 RESULTS:")
        if short_signals:
            print(f"  ✅ {len(short_signals)} SHORT signal(s) detected!")
            for s in short_signals:
                print(f"     {s['date']} | Price: {s['price']} | J: {s['daily_j']}/{s['weekly_j']} | "
                      f"Patterns: {s['patterns']} | Strength: {s['strength']}")
        else:
            print(f"  ❌ No Short signals detected during this decline")

        if long_signals:
            print(f"  ℹ️  {len(long_signals)} LONG signal(s) also detected (counter-trend)")
            for s in long_signals:
                print(f"     {s['date']} | Price: {s['price']} | J: {s['daily_j']}/{s['weekly_j']} | "
                      f"Patterns: {s['patterns']}")

        results_summary.append({
            'period': period['label'],
            'result': 'DETECTED' if short_signals else 'MISSED',
            'short_signals': len(short_signals),
            'long_signals': len(long_signals),
        })

        # Brief pause between API calls
        if i < len(TEST_PERIODS) - 1:
            time.sleep(2)

    # Final Summary
    print(f"\n\n{'='*80}")
    print("📋 BACKTEST SUMMARY")
    print(f"{'='*80}")
    for r in results_summary:
        icon = "✅" if r['result'] == 'DETECTED' else ("⚠️" if r['result'] == 'NO DATA' else "❌")
        print(f"  {icon} {r['period']}: {r['result']}")
        if 'short_signals' in r:
            print(f"      Short signals: {r['short_signals']}, Long signals: {r['long_signals']}")

    detected = sum(1 for r in results_summary if r['result'] == 'DETECTED')
    total = len(results_summary)
    no_data = sum(1 for r in results_summary if r['result'] == 'NO DATA')
    testable = total - no_data
    print(f"\n  Detection Rate: {detected}/{testable} periods "
          f"({'N/A' if testable == 0 else f'{detected/testable*100:.0f}%'})")


if __name__ == '__main__':
    main()
