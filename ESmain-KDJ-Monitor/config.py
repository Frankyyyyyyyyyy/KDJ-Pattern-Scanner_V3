"""
集中配置 —— 所有可调参数从环境变量读取，便于多环境部署且不泄露密钥到 GitHub。

加载顺序：
    1. 进程环境变量
    2. 项目根目录 .env（若存在）
    3. 本文件中的硬编码默认值
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------- 简易 .env 加载 ----------
def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(PROJECT_ROOT / ".env")


# ---------- 日志 ----------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("esmain_kdj")


# ---------- 工具函数 ----------
def _read_secret_file(name: str) -> Optional[str]:
    """从仓库根 / 父目录读取明文密钥文件（兼容旧用法）。"""
    for cand in (PROJECT_ROOT / name, PROJECT_ROOT.parent / name):
        if cand.exists():
            try:
                v = cand.read_text(encoding="utf-8").strip()
                if v:
                    return v
            except Exception:
                pass
    return None


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    return v.strip() if isinstance(v, str) and v.strip() else default


def _env_float(key: str, default: float) -> float:
    v = _env(key)
    try:
        return float(v) if v is not None else default
    except ValueError:
        logger.warning(f"环境变量 {key}={v!r} 不是数字，使用默认 {default}")
        return default


def _env_int(key: str, default: int) -> int:
    return int(_env_float(key, default))


def _env_list(key: str, default: List[str]) -> List[str]:
    v = _env(key)
    if not v:
        return default
    return [s.strip() for s in v.split(",") if s.strip()]


# ---------- 数据源 ----------
POLYGON_API_KEY = _env("POLYGON_API_KEY") or _read_secret_file("POLYGON_API_KEY.txt")
DATA_SOURCE_PRIORITY = _env_list("DATA_SOURCE_PRIORITY", ["polygon", "yahoo", "offline"])

# Polygon ESmain 的 ticker；不同账户可能不同
POLYGON_TICKER = _env("POLYGON_TICKER", "ESmain")
YAHOO_TICKER = _env("YAHOO_TICKER", "ES=F")


# ---------- 策略阈值 ----------
@dataclass(frozen=True)
class Thresholds:
    monthly_j: float = 55.0
    weekly_j: float = 10.0
    daily_j: float = 0.0

    def __str__(self) -> str:
        return (f"月J<{self.monthly_j} 且 周J<{self.weekly_j} "
                f"且 日J<{self.daily_j}")


THRESHOLDS = Thresholds(
    monthly_j=_env_float("MONTHLY_J_THRESHOLD", 55.0),
    weekly_j=_env_float("WEEKLY_J_THRESHOLD", 10.0),
    daily_j=_env_float("DAILY_J_THRESHOLD", 0.0),
)


# ---------- 邮件 ----------
@dataclass(frozen=True)
class EmailConfig:
    sender: Optional[str] = None
    sender_name: str = "ESmain KDJ Monitor"
    receiver: Optional[str] = None
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    password: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.sender and self.receiver and self.password)


EMAIL = EmailConfig(
    sender=_env("SENDER_EMAIL"),
    sender_name=_env("SENDER_NAME", "ESmain KDJ Monitor"),
    receiver=_env("RECEIVER_EMAIL"),
    smtp_host=_env("SMTP_HOST", "smtp.gmail.com"),
    smtp_port=_env_int("SMTP_PORT", 465),
    password=_env("GMAIL_APP_PASSWORD") or _read_secret_file("GMAIL_APP_PASSWORD.txt"),
)


# ---------- 告警日志 ----------
# 不再做"去重"——只要日线满足就发邮件。每次告警仅追加一行到 logs/alerts.csv。
ALERT_LOG_FILE = LOG_DIR / "alerts.csv"
