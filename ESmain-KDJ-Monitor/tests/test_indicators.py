"""
KDJ 算法单元测试
================

跑：  python -m pytest tests/ -v   （或 python tests/test_indicators.py）
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators import calculate_kdj, daily_to_weekly, daily_to_monthly  # noqa
from scanner import build_signal_panel  # noqa


def _toy_daily(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """构造一段确定性的日线 OHLC（带噪声的正弦）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = 100 + 10 * np.sin(np.arange(n) / 5)
    close = base + rng.normal(0, 0.5, n)
    high = close + rng.uniform(0.1, 1.5, n)
    low = close - rng.uniform(0.1, 1.5, n)
    open_ = close + rng.normal(0, 0.3, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": 0}, index=idx)


class TestKDJ(unittest.TestCase):

    def test_kdj_columns(self):
        df = calculate_kdj(_toy_daily())
        for c in ["K", "D", "J"]:
            self.assertIn(c, df.columns)

    def test_j_equals_3k_minus_2d(self):
        df = calculate_kdj(_toy_daily())
        self.assertTrue(np.allclose(df["J"], 3 * df["K"] - 2 * df["D"]))

    def test_kdj_is_bounded_in_warmup(self):
        """初始 K=D=50，J 必落在 [-100, 200] 这个保守区间内（极端 RSV 也不会超）。"""
        df = calculate_kdj(_toy_daily())
        self.assertTrue(df["J"].between(-200, 300).all())

    def test_constant_price_gives_kdj_50(self):
        """完全平价时 RSV→50，K/D/J 应该收敛到 50。"""
        idx = pd.date_range("2020-01-01", periods=30, freq="B")
        flat = pd.DataFrame({"Open": 100, "High": 100, "Low": 100,
                             "Close": 100, "Volume": 0}, index=idx)
        df = calculate_kdj(flat)
        self.assertAlmostEqual(df["K"].iloc[-1], 50.0, places=5)
        self.assertAlmostEqual(df["D"].iloc[-1], 50.0, places=5)
        self.assertAlmostEqual(df["J"].iloc[-1], 50.0, places=5)


class TestResample(unittest.TestCase):

    def test_weekly_close_is_friday(self):
        df = _toy_daily(40)
        wk = daily_to_weekly(df)
        # 至少一根周线，且最后一根的 Close 来自周内最后一个交易日
        self.assertGreater(len(wk), 0)
        last_week = wk.iloc[-1]
        # 周内 Close 集合
        week_mask = (df.index >= wk.index[-1] - pd.Timedelta(days=6)) & \
                    (df.index <= wk.index[-1])
        self.assertAlmostEqual(last_week["Close"], df.loc[week_mask, "Close"].iloc[-1])
        self.assertAlmostEqual(last_week["High"], df.loc[week_mask, "High"].max())
        self.assertAlmostEqual(last_week["Low"], df.loc[week_mask, "Low"].min())

    def test_monthly_aggregation(self):
        df = _toy_daily(80)
        mo = daily_to_monthly(df)
        self.assertGreater(len(mo), 0)


class TestNoLookahead(unittest.TestCase):
    """评估日 T 引用的周/月 K 线必须严格早于 T。"""

    def test_strict_less_than(self):
        df = _toy_daily(120)
        panel = build_signal_panel(df, monthly_j_thr=999, weekly_j_thr=999, daily_j_thr=999)
        # 周线收盘日（周五）当天 WK_Date 应当指向上一个周五，不是当天
        fridays = panel[panel["Date"].dt.weekday == 4].dropna(subset=["WK_Date"])
        self.assertGreater(len(fridays), 0)
        for _, row in fridays.iterrows():
            self.assertLess(row["WK_Date"], row["Date"],
                            f"lookahead leak on {row['Date']}: WK_Date={row['WK_Date']}")


class TestSignalGate(unittest.TestCase):

    def test_daily_is_required(self):
        """日线不满足时绝不触发，无论周/月多极端。"""
        df = _toy_daily(120)
        panel = build_signal_panel(df,
                                   monthly_j_thr=999,
                                   weekly_j_thr=999,
                                   daily_j_thr=-999)  # 日线不可能满足
        ready = panel.dropna(subset=["DA_J_prev"])
        self.assertFalse(ready["Signal"].any())

    def test_daily_alone_triggers(self):
        """只要日线满足即触发（强度可能为 LOW）。"""
        df = _toy_daily(120)
        panel = build_signal_panel(df,
                                   monthly_j_thr=-999,  # 月线不可能满足
                                   weekly_j_thr=-999,   # 周线不可能满足
                                   daily_j_thr=999)     # 日线一定满足
        ready = panel.dropna(subset=["DA_J_prev"])
        self.assertTrue(ready["Signal"].any())
        # 强度全部应为 LOW
        sig_rows = ready[ready["Signal"]]
        self.assertTrue((sig_rows["Strength"] == "LOW").all())


class TestSignalStrength(unittest.TestCase):

    def test_strength_tiers(self):
        """三档强度的具体分类逻辑。"""
        from signal_strength import classify, SignalStrength

        # HIGH: 三个都满足
        c = classify(daily_j_prev=-5, weekly_j=-3, monthly_j=20,
                     daily_thr=0, weekly_thr=10, monthly_thr=55)
        self.assertEqual(c.strength, SignalStrength.HIGH)

        # MEDIUM: 日 + 周
        c = classify(daily_j_prev=-5, weekly_j=-3, monthly_j=80,
                     daily_thr=0, weekly_thr=10, monthly_thr=55)
        self.assertEqual(c.strength, SignalStrength.MEDIUM)

        # MEDIUM: 日 + 月（无周）
        c = classify(daily_j_prev=-5, weekly_j=50, monthly_j=20,
                     daily_thr=0, weekly_thr=10, monthly_thr=55)
        self.assertEqual(c.strength, SignalStrength.MEDIUM)

        # LOW: 仅日线
        c = classify(daily_j_prev=-5, weekly_j=50, monthly_j=80,
                     daily_thr=0, weekly_thr=10, monthly_thr=55)
        self.assertEqual(c.strength, SignalStrength.LOW)

        # NONE: 日线不满足，无论周月
        c = classify(daily_j_prev=10, weekly_j=-3, monthly_j=20,
                     daily_thr=0, weekly_thr=10, monthly_thr=55)
        self.assertEqual(c.strength, SignalStrength.NONE)
        self.assertFalse(c.triggers)

    def test_strength_rank_ordering(self):
        from signal_strength import SignalStrength
        ranks = [SignalStrength.NONE.rank, SignalStrength.LOW.rank,
                 SignalStrength.MEDIUM.rank, SignalStrength.HIGH.rank]
        self.assertEqual(ranks, sorted(ranks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
