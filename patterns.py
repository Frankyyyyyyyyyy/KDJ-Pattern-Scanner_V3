import pandas as pd
from config import STRATEGY_CONFIG
from filters import check_prior_trend


def _prior_trend_direction(df, loc_idx):
    """Use the existing prior-trend filter to disambiguate identical candle shapes.

    Hammer and hanging man (and inverted hammer / shooting star) share the same
    geometry. When the prior-trend check is decisive, map:
      prior uptrend  -> bearish (hanging man / shooting star)
      prior downtrend -> bullish (hammer / inverted hammer)
    If neither direction qualifies, leave the shape untagged rather than
    emitting both labels.
    """
    can_short = check_prior_trend("Short", df, loc_idx)
    can_long = check_prior_trend("Long", df, loc_idx)
    if can_short and not can_long:
        return "Short"
    if can_long and not can_short:
        return "Long"
    return None


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
        is_upper_pin = False
        is_lower_pin = False

        # --- Bearish Patterns ---
        bearish = []

        if is_doji:
            if j_values is not None:
                j_val = j_values.iloc[i]
                if j_val > 60:
                    bearish.append("Doji (十字星)")
            else:
                bearish.append("Doji (十字星)")

        else:
            is_upper_pin = (
                (upper_shadow >= cfg['shadow_body_multiplier'] * body) and
                (upper_shadow >= cfg['shadow_other_multiplier'] * lower_shadow) and
                (body <= open_p * cfg['candle_body_ratio'])
            )
            is_lower_pin = (
                (lower_shadow >= cfg['shadow_body_multiplier'] * body) and
                (lower_shadow >= cfg['shadow_other_multiplier'] * upper_shadow) and
                (body <= open_p * cfg['candle_body_ratio'])
            )
            trend_dir = _prior_trend_direction(df, i) if (is_upper_pin or is_lower_pin) else None
            if is_upper_pin and trend_dir == "Short":
                bearish.append("Shooting Star (射击之星)")
            elif is_lower_pin and trend_dir == "Short":
                bearish.append("Hanging Man (吊颈线)")

        if (not is_doji) and (not is_upper_pin) and (not is_lower_pin):
            if (prev['Close'] > prev['Open']) and (close_p < open_p) and \
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

        else:
            is_lower_pin = (
                (lower_shadow >= cfg['shadow_body_multiplier'] * body) and
                (lower_shadow >= cfg['shadow_other_multiplier'] * upper_shadow) and
                (body <= open_p * cfg['candle_body_ratio'])
            )
            is_upper_pin = (
                (upper_shadow >= cfg['shadow_body_multiplier'] * body) and
                (upper_shadow >= cfg['shadow_other_multiplier'] * lower_shadow) and
                (body <= open_p * cfg['candle_body_ratio'])
            )
            trend_dir = _prior_trend_direction(df, i) if (is_upper_pin or is_lower_pin) else None
            if is_lower_pin and trend_dir == "Long":
                bullish.append("Hammer (锤子线)")
            elif is_upper_pin and trend_dir == "Long":
                bullish.append("Inverted Hammer (倒锤子线)")

        if (not is_doji) and (not is_upper_pin) and (not is_lower_pin):
            if (prev['Close'] < prev['Open']) and (close_p > open_p) and \
                 (open_p <= prev['Close']) and (close_p >= prev['Open']):
                bullish.append("Bullish Engulfing (看涨吞没)")

            elif (prev['Close'] < prev['Open']) and (open_p < prev['Close']) and \
                 (close_p > (prev['Open'] + prev['Close']) / 2) and (close_p < prev['Open']):
                bullish.append("Piercing Line (刺透形态)")

        bullish_series.iloc[i] = bullish

    return bearish_series, bullish_series
