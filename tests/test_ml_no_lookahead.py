# -*- coding: utf-8 -*-
"""Offline tests: complete-horizon labels, point-in-time training, strategy J."""
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
from ml_predictor import (  # noqa: E402
    get_triple_barrier_labels,
    calculate_advanced_features,
    build_point_in_time_train_predict,
)


def _series(n=80, start="2018-01-02"):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range(start, periods=n)
    close = 100 + np.cumsum(rng.normal(0, 0.8, n))
    high = close + rng.uniform(0.2, 1.2, n)
    low = close - rng.uniform(0.2, 1.2, n)
    open_ = close + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1_000_000,
    }, index=idx)


class TestTripleBarrier(unittest.TestCase):
    def test_incomplete_horizon_is_nan_not_zero(self):
        df = _series(30)
        horizon = 10
        labels = get_triple_barrier_labels(df, horizon=horizon, direction="Long")
        # Last `horizon` rows cannot observe a full future window
        self.assertTrue(labels.iloc[-horizon:].isna().all())
        self.assertTrue(labels.iloc[:-horizon].notna().all())
        self.assertFalse((labels.iloc[-horizon:] == 0).any())


class TestPointInTime(unittest.TestCase):
    def test_predict_row_excluded_and_labels_use_only_visible_future(self):
        df = _series(120)
        asof = df.index[-1]
        packed = build_point_in_time_train_predict(df, end_date=asof, direction="Long", horizon=10)
        self.assertIsNotNone(packed["predict_idx"])
        self.assertEqual(packed["predict_idx"], asof)
        train = packed["train_df"]
        self.assertFalse((train.index >= asof).any())
        # Any training label needs 10 future bars, all of which must be <= asof
        # so the last allowed train index is the 11th-from-last bar (iloc[-11])
        last_allowed = df.index[-11]
        self.assertTrue((train.index <= last_allowed).all())
        # Features J matches strategy KDJ
        feat = packed["features"]
        kdj = calculate_kdj(df[df.index <= asof])
        common = feat.dropna(subset=["KDJ_J"]).index.intersection(kdj.dropna(subset=["J"]).index)
        self.assertGreater(len(common), 10)
        self.assertTrue(np.allclose(feat.loc[common, "KDJ_J"], kdj.loc[common, "J"]))


class TestFeatureJ(unittest.TestCase):
    def test_advanced_features_j_matches_calculate_kdj(self):
        df = _series(60)
        feat = calculate_advanced_features(df)
        kdj = calculate_kdj(df)
        mask = feat["KDJ_J"].notna() & kdj["J"].notna()
        self.assertTrue(np.allclose(feat.loc[mask, "KDJ_J"], kdj.loc[mask, "J"]))


if __name__ == "__main__":
    unittest.main()
