from config import STRATEGY_CONFIG
import pandas as pd

def check_index_filter(direction, idx_daily_j, idx_j_prev, idx_patterns, divergence):
    """
    Index 过滤器：检查 QQQ 是否满足方向条件。

    Short: QQQ J>80, 或有看跌形态, 或 J>70 且下降, 或看跌背离
    Long:  QQQ J<20, 或有看涨形态, 或 J<30 且上升, 或看涨背离
    """
    cfg = STRATEGY_CONFIG

    if direction == "Short":
        j_changing = idx_daily_j < idx_j_prev if idx_j_prev is not None else False
        return (idx_daily_j > cfg['idx_j_overbought']) or \
               (len(idx_patterns) > 0) or \
               (idx_daily_j > cfg['idx_j_overbought_falling'] and j_changing) or \
               divergence
    else:
        j_changing = idx_daily_j > idx_j_prev if idx_j_prev is not None else False
        return (idx_daily_j < cfg['idx_j_oversold']) or \
               (len(idx_patterns) > 0) or \
               (idx_daily_j < cfg['idx_j_oversold_rising'] and j_changing) or \
               divergence


def check_ma_cross_filter(direction, df, loc_idx):
    """
    MA 交叉过滤器：

    Short: 近 N 天内没有金叉（MA_short 上穿 MA_long）→ 返回 True 表示通过
    Long:  近 N 天内没有死叉（MA_short 下穿 MA_long）→ 返回 True 表示通过
    """
    cfg = STRATEGY_CONFIG
    lookback = cfg['ma_cross_lookback']

    for i in range(loc_idx - lookback, loc_idx + 1):
        if i <= 0:
            continue
        ma_s = df['MA10'].iloc[i]
        ma_l = df['MA20'].iloc[i]
        p_ma_s = df['MA10'].iloc[i - 1]
        p_ma_l = df['MA20'].iloc[i - 1]

        if pd.isna(ma_s) or pd.isna(p_ma_s):
            continue

        if direction == "Short":
            if p_ma_s <= p_ma_l and ma_s > ma_l:
                return False
        else:
            if p_ma_s >= p_ma_l and ma_s < ma_l:
                return False

    return True


def check_j_filter(direction, daily_j, prev_daily_j, wk_j, prev_wk_j):
    """
    J 值过滤器：

    Short: 日线 J>75 或 (J>60 且下降), 且周线 J>80 或下降
    Long:  日线 J<25 或 (J<40 且上升), 且周线 J<20 或上升
    """
    cfg = STRATEGY_CONFIG

    if direction == "Short":
        is_daily_falling = daily_j < prev_daily_j if prev_daily_j is not None else False
        is_wk_falling = wk_j < prev_wk_j if prev_wk_j is not None else False

        pass_daily = (daily_j > cfg['short_j_high']) or \
                     (is_daily_falling and daily_j > cfg['short_j_falling'])
        pass_weekly = (wk_j > cfg['short_wk_j_high']) or is_wk_falling
    else:
        is_daily_rising = daily_j > prev_daily_j if prev_daily_j is not None else False
        is_wk_rising = wk_j > prev_wk_j if prev_wk_j is not None else False

        pass_daily = (daily_j < cfg['long_j_low']) or \
                     (is_daily_rising and daily_j < cfg['long_j_rising'])
        pass_weekly = (wk_j < cfg['long_wk_j_low']) or is_wk_rising

    return pass_daily and pass_weekly


def check_prior_trend(direction, df, loc_idx):
    """
    前置趋势检查：形态的有效性依赖前置趋势方向。

    Short（看跌形态）: 前 N 天内，从最低收盘价到当前价的涨幅 >= 阈值
    Long （看涨形态）: 前 N 天内，从最高收盘价到当前价的跌幅 >= 阈值

    没有前置趋势的反转形态大概率是噪音。
    """
    cfg = STRATEGY_CONFIG
    trend_days = cfg['prior_trend_days']
    min_pct = cfg['prior_trend_min_pct']

    trend_start = max(0, loc_idx - trend_days)
    if trend_start >= loc_idx:
        return False

    # 取前 N 天的收盘价窗口（不含当天）
    window = df.iloc[trend_start:loc_idx]
    end_close = df.iloc[loc_idx]['Close']

    if direction == "Short":
        # 做空：从窗口内最低收盘价到当前价的涨幅
        low_close = window['Close'].min()
        if low_close == 0:
            return False
        pct_change = (end_close - low_close) / low_close
        return pct_change >= min_pct
    else:
        # 做多：从窗口内最高收盘价到当前价的跌幅
        high_close = window['Close'].max()
        if high_close == 0:
            return False
        pct_change = (end_close - high_close) / high_close
        return pct_change <= -min_pct


def check_divergence(direction, df, df_index, date, loc_idx):
    """
    真正的 KDJ 背离检测：价格与 J 值的背离。

    顶背离（Short）: 价格创近 N 日新高，但 J 值没有创新高
                     → 价格上涨动能衰竭，潜在见顶信号
    底背离（Long）:  价格创近 N 日新低，但 J 值没有创新低
                     → 价格下跌动能衰竭，潜在见底信号
    """
    cfg = STRATEGY_CONFIG
    lookback = cfg['divergence_lookback']

    div_start = max(0, loc_idx - lookback)
    if div_start >= loc_idx - 1:
        return False

    window = df.iloc[div_start:loc_idx + 1]
    if len(window) < 3:
        return False

    current_close = window['Close'].iloc[-1]
    current_j = window['J'].iloc[-1]
    prev_close_max = window['Close'].iloc[:-1].max()
    prev_close_min = window['Close'].iloc[:-1].min()
    prev_j_max = window['J'].iloc[:-1].max()
    prev_j_min = window['J'].iloc[:-1].min()

    if direction == "Short":
        # 顶背离：价格创新高但 J 没创新高
        price_new_high = current_close >= prev_close_max
        j_no_new_high = current_j < prev_j_max
        return price_new_high and j_no_new_high
    else:
        # 底背离：价格创新低但 J 没创新低
        price_new_low = current_close <= prev_close_min
        j_no_new_low = current_j > prev_j_min
        return price_new_low and j_no_new_low


def check_pattern_filter(direction, df, row, loc_idx):
    """
    K线形态确认：

    近 5 天内出现对应方向形态，且：
    - Short: 当前收盘价 ≤ 形态日最高价
    - Long:  当前收盘价 ≥ 形态日最低价
    - 形态日成交量 > 近5天均量的 80%
    """
    cfg = STRATEGY_CONFIG
    pattern_col = 'Bearish_Patterns' if direction == "Short" else 'Bullish_Patterns'
    lookback = cfg['lookback_days']

    valid_patterns = []
    p_start = max(0, loc_idx - (lookback - 1))

    for i in range(p_start, loc_idx + 1):
        p_row = df.iloc[i]
        if len(p_row[pattern_col]) > 0:
            if direction == "Short":
                price_ok = row['Close'] <= p_row['High']
            else:
                price_ok = row['Close'] >= p_row['Low']

            if price_ok:
                vol_start = max(0, i - lookback)
                vol_win = df.iloc[vol_start:i]['Volume']  # Exclude current day
                if vol_win.empty or p_row['Volume'] > cfg['volume_ratio'] * vol_win.mean():
                    valid_patterns.extend(p_row[pattern_col])

    return valid_patterns
