# -*- coding: utf-8 -*-
"""Pattern disambiguation and position-config defaults."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import STRATEGY_CONFIG  # noqa: E402
from patterns import identify_patterns  # noqa: E402


def _base_row(close, high=None, low=None, open_=None, volume=10000):
    open_ = close if open_ is None else open_
    high = close if high is None else high
    low = close if low is None else low
    return {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}


def _lower_pin_row(price):
    """Lower-shadow pin that is NOT a doji (body > 0.8% of open, <= 2%)."""
    open_ = price
    close = price - 1.2  # body 1.2; doji threshold at open=100+ is ~0.8-1.3
    # Scale body to stay inside (doji_body_ratio, candle_body_ratio) for typical prices 90-120
    body = abs(close - open_)
    if body <= open_ * 0.008:
        close = open_ - (open_ * 0.012)
    if abs(close - open_) > open_ * 0.02:
        close = open_ - (open_ * 0.015)
    body = abs(close - open_)
    low = min(open_, close) - max(4.0, 2.0 * body)
    high = max(open_, close) + 0.1
    return _base_row(close, high=high, low=low, open_=open_)


class TestOptionalFiltersDefaultOff(unittest.TestCase):
    def test_flags_default_false(self):
        self.assertFalse(STRATEGY_CONFIG.get("enable_index_filter"))
        self.assertFalse(STRATEGY_CONFIG.get("enable_ma_cross_filter"))

    def test_position_defaults_match_legacy_behavior(self):
        self.assertEqual(STRATEGY_CONFIG.get("risk_per_trade_dollars"), 500)
        self.assertEqual(STRATEGY_CONFIG.get("atr_stop_multiplier"), 2.0)


class TestPinBarDisambiguation(unittest.TestCase):
    def test_uptrend_lower_pin_is_hanging_man_not_hammer(self):
        # 12-day uptrend > 3%, then a lower-shadow pin bar
        rows = []
        idx = pd.bdate_range("2024-01-02", periods=12)
        price = 100.0
        for i, d in enumerate(idx[:-1]):
            rows.append(_base_row(price))
            price += 1.0  # +10% over 11 steps
        rows.append(_lower_pin_row(price))
        df = pd.DataFrame(rows, index=idx)
        bear, bull = identify_patterns(df)
        last_bear = bear.iloc[-1]
        last_bull = bull.iloc[-1]
        self.assertIn("Hanging Man (吊颈线)", last_bear)
        self.assertNotIn("Hammer (锤子线)", last_bull)

    def test_downtrend_lower_pin_is_hammer_not_hanging_man(self):
        rows = []
        idx = pd.bdate_range("2024-01-02", periods=12)
        price = 110.0
        for i, d in enumerate(idx[:-1]):
            rows.append(_base_row(price))
            price -= 1.0
        rows.append(_lower_pin_row(price))
        df = pd.DataFrame(rows, index=idx)
        bear, bull = identify_patterns(df)
        last_bear = bear.iloc[-1]
        last_bull = bull.iloc[-1]
        self.assertIn("Hammer (锤子线)", last_bull)
        self.assertNotIn("Hanging Man (吊颈线)", last_bear)

    def test_no_prior_trend_leaves_pin_untagged(self):
        # Flat series, last bar is a lower pin
        idx = pd.bdate_range("2024-01-02", periods=12)
        rows = [_base_row(100.0) for _ in idx[:-1]]
        rows.append(_lower_pin_row(100.0))
        df = pd.DataFrame(rows, index=idx)
        bear, bull = identify_patterns(df)
        self.assertEqual(bear.iloc[-1], [])
        self.assertEqual(bull.iloc[-1], [])


if __name__ == "__main__":
    unittest.main()
