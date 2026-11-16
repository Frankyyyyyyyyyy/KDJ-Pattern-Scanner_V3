# -*- coding: utf-8 -*-
"""KDJ warmup convention tests (project convention, not a unique standard)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indicators import calculate_kdj  # noqa: E402


def _ohlc(n=30, start="2020-01-02"):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(np.linspace(100, 110, n), index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1000,
    })


def _legacy_fillna_kdj(df, period=9):
    """Reproduce the previous fillna(50) implementation for compatibility checks."""
    df = df.copy()
    low_min = df["Low"].rolling(window=period).min()
    high_max = df["High"].rolling(window=period).max()
    denom = high_max - low_min
    rsv = (df["Close"] - low_min) / denom * 100
    rsv = rsv.mask(denom == 0, 50)
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50)
    k, d = 50.0, 50.0
    ks, ds = [], []
    for r in rsv:
        k = (2 / 3) * k + (1 / 3) * r
        d = (2 / 3) * d + (1 / 3) * k
        ks.append(k)
        ds.append(d)
    out = df.copy()
    out["K"] = ks
    out["D"] = ds
    out["J"] = 3 * out["K"] - 2 * out["D"]
    return out


class TestKDJWarmup(unittest.TestCase):
    def test_seed_stays_50_until_rsv_defined(self):
        df = calculate_kdj(_ohlc(20), period=9)
        self.assertTrue((df["K"].iloc[:8] == 50).all())
        self.assertTrue((df["D"].iloc[:8] == 50).all())
        # After RSV is defined, K/D may move
        self.assertFalse(np.allclose(df["K"].iloc[8:], 50))

    def test_clean_series_matches_legacy_fillna(self):
        raw = _ohlc(40)
        new = calculate_kdj(raw)
        old = _legacy_fillna_kdj(raw)
        self.assertTrue(np.allclose(new["K"], old["K"]))
        self.assertTrue(np.allclose(new["D"], old["D"]))
        self.assertTrue(np.allclose(new["J"], old["J"]))

    def test_j_identity(self):
        df = calculate_kdj(_ohlc(25))
        self.assertTrue(np.allclose(df["J"], 3 * df["K"] - 2 * df["D"]))


if __name__ == "__main__":
    unittest.main()
