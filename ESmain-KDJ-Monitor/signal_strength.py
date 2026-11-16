"""
信号强度分级
============

触发门槛：日线 J 必须满足（J < daily_j 阈值）。
日线满足之后，再看周线、月线是否也满足，分三档：

    HIGH   月 + 周 + 日 全部满足
    MEDIUM 日 + (周 或 月) 满足，但不全
    LOW    仅日线满足

注：用户原话只提了 "周+日=次之"，这里把"月+日"等价并入 MEDIUM
（理由：高时间框架确认价值相当，缺一档算 MEDIUM）。
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class SignalStrength(Enum):
    HIGH = "HIGH"      # 月+周+日 三确认
    MEDIUM = "MEDIUM"  # 日+周 或 日+月（两确认）
    LOW = "LOW"        # 仅日线（单确认）
    NONE = "NONE"      # 不触发

    @property
    def label_cn(self) -> str:
        return {
            "HIGH":   "强 (月+周+日 三确认)",
            "MEDIUM": "中 (日+周 或 日+月 两确认)",
            "LOW":    "弱 (仅日线)",
            "NONE":   "无信号",
        }[self.value]

    @property
    def color_hex(self) -> str:
        return {
            "HIGH":   "#16a34a",  # 绿
            "MEDIUM": "#d97706",  # 橙
            "LOW":    "#6b7280",  # 灰
            "NONE":   "#9ca3af",
        }[self.value]

    @property
    def rank(self) -> int:
        return {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}[self.value]


@dataclass(frozen=True)
class Conditions:
    daily_ok: bool
    weekly_ok: bool
    monthly_ok: bool

    @property
    def strength(self) -> SignalStrength:
        if not self.daily_ok:
            return SignalStrength.NONE
        if self.weekly_ok and self.monthly_ok:
            return SignalStrength.HIGH
        if self.weekly_ok or self.monthly_ok:
            return SignalStrength.MEDIUM
        return SignalStrength.LOW

    @property
    def triggers(self) -> bool:
        """触发邮件的最低门槛 = 日线满足。"""
        return self.daily_ok


def classify(daily_j_prev, weekly_j, monthly_j,
             daily_thr: float, weekly_thr: float, monthly_thr: float) -> Conditions:
    """从三条 J 值 + 三条阈值算出 Conditions（NaN 当作不满足）。"""
    import math

    def _lt(v, t):
        try:
            return (v is not None) and (not math.isnan(float(v))) and float(v) < float(t)
        except (TypeError, ValueError):
            return False

    return Conditions(
        daily_ok=_lt(daily_j_prev, daily_thr),
        weekly_ok=_lt(weekly_j, weekly_thr),
        monthly_ok=_lt(monthly_j, monthly_thr),
    )
