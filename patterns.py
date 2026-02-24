import pandas as pd
from config import STRATEGY_CONFIG

def identify_patterns(df, j_values=None):
    """
    Identifies both Bearish and Bullish patterns.

    Args:
        df: DataFrame with OHLCV data
        j_values: Optional Series of J values. When provided, Doji is classified
                  contextually: J > 60 → bearish only, J < 40 → bullish only,
                  40 ≤ J ≤ 60 → ignored (neutral zone).

    Returns two Series: bearish_patterns, bullish_patterns
    """
    cfg = STRATEGY_CONFIG
    bearish_series = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)
    bullish_series = pd.Series([[] for _ in range(len(df))], index=df.index, dtype=object)

    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]

        open_p, close_p = curr['Open'], curr['Close']
        high_p, low_p = curr['High'], curr['Low']

        body = abs(close_p - open_p)
        upper_shadow = high_p - max(open_p, close_p)
        lower_shadow = min(open_p, close_p) - low_p

        is_doji = body <= (open_p * cfg['doji_body_ratio'])

        # --- Bearish Patterns ---
        bearish = []

        if is_doji:
            if j_values is not None:
                j_val = j_values.iloc[i]
                if j_val > 60:
                    bearish.append("Doji (十字星)")
            else:
                bearish.append("Doji (十字星)")

        elif (upper_shadow >= cfg['shadow_body_multiplier'] * body) and \
             (upper_shadow >= cfg['shadow_other_multiplier'] * lower_shadow) and \
             (body <= open_p * cfg['candle_body_ratio']):
            bearish.append("Shooting Star (射击之星)")

        elif (lower_shadow >= cfg['shadow_body_multiplier'] * body) and \
             (lower_shadow >= cfg['shadow_other_multiplier'] * upper_shadow) and \
             (body <= open_p * cfg['candle_body_ratio']):
            bearish.append("Hanging Man (吊颈线)")

        elif (prev['Close'] > prev['Open']) and (close_p < open_p) and \
             (open_p >= prev['Close']) and (close_p <= prev['Open']):
            bearish.append("Bearish Engulfing (看跌吞没)")

        elif (prev['Close'] > prev['Open']) and (open_p > prev['Close']) and \
             (close_p < (prev['Open'] + prev['Close']) / 2) and (close_p > prev['Open']):
            bearish.append("Dark Cloud Cover (乌云盖顶)")

        bearish_series.iloc[i] = bearish

        # --- Bullish Patterns ---
        bullish = []

        if is_doji:
            if j_values is not None:
                j_val = j_values.iloc[i]
                if j_val < 40:
                    bullish.append("Doji (十字星)")
            else:
                bullish.append("Doji (十字星)")

        elif (lower_shadow >= cfg['shadow_body_multiplier'] * body) and \
             (lower_shadow >= cfg['shadow_other_multiplier'] * upper_shadow) and \
             (body <= open_p * cfg['candle_body_ratio']):
            bullish.append("Hammer (锤子线)")

        elif (upper_shadow >= cfg['shadow_body_multiplier'] * body) and \
             (upper_shadow >= cfg['shadow_other_multiplier'] * lower_shadow) and \
             (body <= open_p * cfg['candle_body_ratio']):
            bullish.append("Inverted Hammer (倒锤子线)")

        elif (prev['Close'] < prev['Open']) and (close_p > open_p) and \
             (open_p <= prev['Close']) and (close_p >= prev['Open']):
            bullish.append("Bullish Engulfing (看涨吞没)")

        elif (prev['Close'] < prev['Open']) and (open_p < prev['Close']) and \
             (close_p > (prev['Open'] + prev['Close']) / 2) and (close_p < prev['Open']):
            bullish.append("Piercing Line (刺透形态)")

        bullish_series.iloc[i] = bullish

    return bearish_series, bullish_series
