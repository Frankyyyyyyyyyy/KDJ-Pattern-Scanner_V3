# -*- coding: utf-8 -*-
"""Offline cache refresh, range, source-consistency, and yahoo end-date tests."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    yahoo_exclusive_end,
    cache_fetch_start,
    overlap_price_mismatch,
    update_price_cache_year,
    load_cached_data,
    _read_price_cache,
)


def _ohlc_frame(dates, close_start=100.0, step=0.5):
    rows = []
    px = close_start
    for d in dates:
        rows.append({
            "Open": px, "High": px + 1, "Low": px - 1, "Close": px, "Volume": 1000,
        })
        px += step
    df = pd.DataFrame(rows, index=pd.to_datetime(dates))
    df.index.name = "Date"
    return df


class TestDateHelpers(unittest.TestCase):
    def test_yahoo_end_is_exclusive(self):
        self.assertEqual(yahoo_exclusive_end("2024-01-31"), "2024-02-01")
        self.assertEqual(yahoo_exclusive_end("2024-12-31"), "2025-01-01")
        self.assertIsNone(yahoo_exclusive_end(None))

    def test_fetch_start_includes_overlap(self):
        start = pd.Timestamp("2024-01-01")
        last = pd.Timestamp("2024-01-20")
        got = cache_fetch_start(last, start, overlap_days=7)
        # 7-day window ending on last inclusive: 2024-01-14 .. 2024-01-20
        self.assertEqual(got, pd.Timestamp("2024-01-14"))
        # No cache: start from range start
        self.assertEqual(cache_fetch_start(None, start, overlap_days=7), start)


class TestCacheUpdate(unittest.TestCase):
    def test_overlap_refresh_replaces_last_bar_without_api(self):
        with tempfile.TemporaryDirectory() as td:
            year = 2024
            existing_dates = pd.bdate_range("2024-01-02", "2024-01-10")
            old = []
            for d in existing_dates:
                px = 100.0  # flat closes, so overlap will match
                old.append({
                    "Date": d, "Ticker": "AAA",
                    "Open": px, "High": px + 1, "Low": px - 1, "Close": px,
                    "Volume": 1000, "Source": "polygon",
                })
            cache_path = Path(td) / f"prices_{year}.csv"
            pd.DataFrame(old).assign(
                Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%Y-%m-%d")
            ).to_csv(cache_path, index=False, encoding="utf-8-sig")

            def poly_fn(ticker, start, end, timespan="day"):
                # Honor the requested range; keep closes at 100 so overlap matches
                fetch_dates = pd.bdate_range(start, end)
                return _ohlc_frame(fetch_dates, close_start=100.0, step=0.0)

            def yahoo_fn(*args, **kwargs):
                return {}

            # First: no mismatch (close stays 100), overlap rows refreshed, new days appended
            update_price_cache_year(
                ["AAA"], "2024-01-02", "2024-01-12",
                cache_dir=td,
                fetch_polygon_fn=poly_fn,
                fetch_yahoo_fn=yahoo_fn,
                overlap_days=7,
                mismatch_tol=0.05,
            )
            loaded = load_cached_data(["AAA"], "2024-01-02", "2024-01-12", cache_dir=td)
            df = loaded["AAA"]
            self.assertIn(pd.Timestamp("2024-01-11"), df.index)
            self.assertIn(pd.Timestamp("2024-01-12"), df.index)
            # original early day kept
            self.assertIn(pd.Timestamp("2024-01-02"), df.index)

    def test_mismatch_invalidates_year_and_refetches(self):
        with tempfile.TemporaryDirectory() as td:
            dates = pd.bdate_range("2024-01-02", "2024-01-10")
            old = []
            for i, d in enumerate(dates):
                px = 100 + i
                old.append({
                    "Date": d, "Ticker": "BBB",
                    "Open": px, "High": px + 1, "Low": px - 1, "Close": px,
                    "Volume": 1000, "Source": "polygon",
                })
            pd.DataFrame(old).assign(
                Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%Y-%m-%d")
            ).to_csv(Path(td) / "prices_2024.csv", index=False, encoding="utf-8-sig")

            calls = {"n": 0, "starts": []}

            def poly_fn(ticker, start, end, timespan="day"):
                calls["n"] += 1
                calls["starts"].append(start)
                # Completely different closes -> mismatch
                all_dates = pd.bdate_range(start, end)
                return _ohlc_frame(all_dates, close_start=50.0, step=0.0)

            def yahoo_fn(*args, **kwargs):
                return {}

            update_price_cache_year(
                ["BBB"], "2024-01-02", "2024-01-10",
                cache_dir=td,
                fetch_polygon_fn=poly_fn,
                fetch_yahoo_fn=yahoo_fn,
                overlap_days=7,
                mismatch_tol=0.01,
            )
            # First fetch from overlap start, then refetch from year/range start
            self.assertGreaterEqual(calls["n"], 2)
            self.assertEqual(calls["starts"][-1], "2024-01-02")
            df = load_cached_data(["BBB"], "2024-01-02", "2024-01-10", cache_dir=td)["BBB"]
            # After invalidation, closes should be the new 50 series, not 100+
            self.assertTrue((df["Close"] == 50.0).all())

    def test_overlap_mismatch_helper(self):
        old = pd.DataFrame({
            "Date": [pd.Timestamp("2024-01-10")],
            "Ticker": ["X"],
            "Close": [100.0],
        })
        new = pd.DataFrame({
            "Date": [pd.Timestamp("2024-01-10")],
            "Ticker": ["X"],
            "Close": [103.0],
        })
        self.assertTrue(overlap_price_mismatch(old, new, tol=0.01))
        self.assertFalse(overlap_price_mismatch(old, old, tol=0.01))


if __name__ == "__main__":
    unittest.main()
