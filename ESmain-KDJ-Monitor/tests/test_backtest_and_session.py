# -*- coding: utf-8 -*-
"""ES backtest uses entry-time strength; session bounds are conservative and not NYSE-based."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest import backtest_episodes  # noqa: E402
from data_fetcher import (  # noqa: E402
    yahoo_exclusive_end,
    last_trusted_bar_date,
    drop_untrusted_trailing_bars,
)
from indicators import calculate_kdj  # noqa: E402
import numpy as np
from scanner import extract_signal_episodes  # noqa: E402
from signal_strength import SignalStrength  # noqa: E402


def _daily(start, periods, close=4000.0):
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({
        "Open": close, "High": close + 10, "Low": close - 10, "Close": close, "Volume": 1,
    }, index=idx)


class TestYahooEnd(unittest.TestCase):
    def test_inclusive_to_exclusive(self):
        self.assertEqual(yahoo_exclusive_end("2024-06-28"), "2024-06-29")


class TestSessionBounds(unittest.TestCase):
    def test_weekday_before_close_excludes_today(self):
        # Wednesday 16:00 ET — before 17:00 cutoff
        now = pd.Timestamp("2024-06-05 16:00", tz="America/New_York")
        self.assertEqual(last_trusted_bar_date(now), pd.Timestamp("2024-06-04"))

    def test_weekday_after_close_keeps_today(self):
        now = pd.Timestamp("2024-06-05 17:00", tz="America/New_York")
        self.assertEqual(last_trusted_bar_date(now), pd.Timestamp("2024-06-05"))

    def test_sunday_drops_weekend_bar(self):
        now = pd.Timestamp("2024-06-09 20:00", tz="America/New_York")  # Sunday night
        self.assertEqual(last_trusted_bar_date(now), pd.Timestamp("2024-06-07"))  # Friday

    def test_drop_trailing(self):
        idx = pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05"])
        df = pd.DataFrame({"Close": [1, 2, 3]}, index=idx)
        now = pd.Timestamp("2024-06-05 16:00", tz="America/New_York")
        clipped = drop_untrusted_trailing_bars(df, now_et=now)
        self.assertEqual(clipped.index.max(), pd.Timestamp("2024-06-04"))


class TestBacktestEntryStrength(unittest.TestCase):
    def test_groups_by_start_strength_not_peak(self):
        daily = _daily("2020-01-02", 80, close=4000.0)
        # One episode that starts LOW and later would have been labeled HIGH by peak logic
        episodes = pd.DataFrame([
            {
                "Episode_Start": daily.index[10].strftime("%Y-%m-%d"),
                "Episode_End": daily.index[14].strftime("%Y-%m-%d"),
                "Days": 5,
                "Peak_Strength": "HIGH",
                "Start_Strength": "LOW",
                "Start_Close": 4000.0,
                "End_Close": 4000.0,
            }
        ])
        trades = backtest_episodes(daily, episodes, hold_days=5, no_overlap=False)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Strength"], "LOW")
        self.assertEqual(trades.iloc[0]["Peak_Strength"], "HIGH")

    def test_kdj_warmup_compatible_with_legacy_on_clean_data(self):
        df = _daily("2020-01-02", 40, close=4000.0)
        df["Close"] = 4000 + np.arange(40)
        df["High"] = df["Close"] + 5
        df["Low"] = df["Close"] - 5
        df["Open"] = df["Close"]
        out = calculate_kdj(df)
        self.assertTrue((out["K"].iloc[:8] == 50).all())


if __name__ == "__main__":
    unittest.main()
