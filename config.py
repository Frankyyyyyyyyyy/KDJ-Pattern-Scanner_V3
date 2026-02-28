import os
import logging

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _read_secret_file(path):
    try:
        if not path:
            return None
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            val = f.read().strip()
        return val or None
    except Exception:
        return None

# ==========================================
# Data Source Configuration
# ==========================================
# 优先使用 Polygon.io（国内可直接访问，不需要 VPN）
# Yahoo Finance 作为备选
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    POLYGON_API_KEY = _read_secret_file(os.environ.get("POLYGON_API_KEY_FILE") or "POLYGON_API_KEY.txt")
if not POLYGON_API_KEY:
    logger.warning("POLYGON_API_KEY not found in environment variables. Polygon data source will be unavailable.")

DATA_SOURCE_PRIORITY = ['polygon', 'yahoo']  # Polygon 优先，Yahoo 备选（解决 Polygon 免费版数据延迟问题）

# ==========================================
# File & Email Configuration
# ==========================================
STOCK_LIST_FILE = 'stock_list.csv'
ARCHIVE_DIR = 'archive'
PRICE_CACHE_DIR = 'price_cache'
INDEX_TICKER = 'QQQ'
SENDER_EMAIL = "wrobincool@gmail.com"
RECEIVER_EMAIL = "wfancool@outlook.com"
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
if not GMAIL_APP_PASSWORD:
    GMAIL_APP_PASSWORD = _read_secret_file(os.environ.get("GMAIL_APP_PASSWORD_FILE") or "GMAIL_APP_PASSWORD.txt")

# ==========================================
# Strategy Configuration (集中管理所有阈值)
# ==========================================
STRATEGY_CONFIG = {
    # Short 方向 J 值阈值
    'short_j_high': 75,
    'short_j_falling': 60,
    'short_wk_j_high': 80,
    # Long 方向 J 值阈值
    'long_j_low': 25,
    'long_j_rising': 40,
    'long_wk_j_low': 20,
    # Index 过滤 J 值
    'idx_j_overbought': 80,
    'idx_j_overbought_falling': 50,
    'idx_j_oversold': 20,
    'idx_j_oversold_rising': 30,
    # K线形态参数
    'doji_body_ratio': 0.008,       # 放宽十字星判定 (0.3% -> 0.8%)
    'candle_body_ratio': 0.02,      # 略微放宽形态实体限制 (1.5% -> 2.0%)
    'shadow_body_multiplier': 1.8,  # 放宽影线与实体倍数 (2.0 -> 1.8)
    'shadow_other_multiplier': 1.5, # 放宽单侧影线主导性 (3.0 -> 1.5, 允许有一定反向影线)
    # 其他
    'volume_ratio': 0.6,
    'ma_short': 10,
    'ma_long': 20,
    'lookback_days': 5,
    'ma_cross_lookback': 10,
    'performance_days': 5,
    # 止损
    'stop_loss_buffer': 0.01,
    # 前置趋势检查
    'prior_trend_days': 10,         # 形态前回看天数
    'prior_trend_min_pct': 0.03,    # 前置趋势最低涨跌幅 3%
    # ATR 相关
    'atr_period': 14,               # ATR 计算周期
    'atr_effective_multiplier': 1.0, # Effective 阈值 = ATR 日均 × 此倍数
    # 背离检测
    'divergence_lookback': 10,      # 背离回看天数
    # 交易成本
    'round_trip_cost': 0.002,       # 往返交易成本 0.2%
    # 仓位管理
    'risk_per_trade': 0.01,         # 每笔交易风险占总资金 1%
    'default_capital': 100000,      # 默认总资金（用于仓位建议）
}

# ==========================================
# Pattern Weights (信号强度权重)
# ==========================================
PATTERN_WEIGHTS = {
    # Bearish
    "Bearish Engulfing (看跌吞没)": 5,
    "Dark Cloud Cover (乌云盖顶)": 4,
    "Shooting Star (射击之星)": 3,
    "Hanging Man (吊颈线)": 2,
    # Bullish
    "Bullish Engulfing (看涨吞没)": 5,
    "Piercing Line (刺透形态)": 4,
    "Hammer (锤子线)": 3,
    "Inverted Hammer (倒锤子线)": 2,
    # Common
    "Doji (十字星)": 1,
}
