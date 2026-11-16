"""
ESmain（标普 500 E-mini 期货连续主力合约）数据获取
====================================================

数据源（按 config.DATA_SOURCE_PRIORITY 顺序尝试）:
    1. polygon   Polygon.io 期货 API   —— 首选；需付费 Futures 订阅
    2. yahoo     yfinance ES=F         —— 免费；中国大陆需 VPN
    3. offline   本地 SPX 现货代理     —— 离线/无网演示，仅适合验证算法

返回统一的 DataFrame:
    index   = DatetimeIndex (US/Eastern, normalized, tz-naive)
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
"""

from __future__ import annotations
import time
from typing import Optional, List

import pandas as pd

from config import (
    POLYGON_API_KEY, POLYGON_TICKER, YAHOO_TICKER,
    DATA_SOURCE_PRIORITY, DATA_DIR, logger,
)

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# ES / CME equity-index futures: daily bar is treated as complete only after
# this Eastern-time cutoff on a weekday. This is a conservative heuristic,
# not a full CME calendar (holidays, early closes, and the Sunday evening
# open are not modeled). We do not use the NYSE cash-session calendar here.
ES_SESSION_CLOSE_HOUR = 17
ES_SESSION_CLOSE_MINUTE = 0
ES_TZ = "America/New_York"


def yahoo_exclusive_end(end_date):
    """yfinance `end` is right-open; convert an inclusive end date."""
    if end_date is None:
        return None
    return (pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def last_trusted_bar_date(now_et=None, close_hour=ES_SESSION_CLOSE_HOUR,
                          close_minute=ES_SESSION_CLOSE_MINUTE):
    """Last calendar date in US/Eastern whose daily bar we are willing to trust.

    Weekdays: trust 'today' only if now_et is at or after the session-close cutoff.
    Weekends: trust through the preceding Friday (drop in-progress Sunday night bars).
    """
    if now_et is None:
        now_et = pd.Timestamp.now(tz=ES_TZ)
    else:
        now_et = pd.Timestamp(now_et)
        if now_et.tzinfo is None:
            now_et = now_et.tz_localize(ES_TZ)
        else:
            now_et = now_et.tz_convert(ES_TZ)

    # Compare in ET, return a tz-naive calendar date.
    today_naive = pd.Timestamp(year=now_et.year, month=now_et.month, day=now_et.day)
    weekday = int(now_et.weekday())  # Mon=0
    if weekday >= 5:
        # Saturday=5 -> Friday is 1 day back; Sunday=6 -> Friday is 2 days back
        return today_naive - pd.Timedelta(days=(weekday - 4))
    close_dt = now_et.normalize() + pd.Timedelta(hours=close_hour, minutes=close_minute)
    if now_et >= close_dt:
        return today_naive
    return today_naive - pd.Timedelta(days=1)


def drop_untrusted_trailing_bars(df, now_et=None, close_hour=ES_SESSION_CLOSE_HOUR,
                                 close_minute=ES_SESSION_CLOSE_MINUTE):
    """Drop bars dated after last_trusted_bar_date. Empty input is unchanged."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(ES_TZ).tz_localize(None)
    out.index = out.index.normalize()
    cutoff = last_trusted_bar_date(now_et=now_et, close_hour=close_hour, close_minute=close_minute)
    return out[out.index <= cutoff]


# ----------------------------------------------------------------------
# Polygon.io 期货
# ----------------------------------------------------------------------
_LAST_POLYGON_REQ = 0.0


def _polygon_get(url: str, timeout: int = 20):
    """带速率限制的 Polygon GET（免费档约 5 RPM，futures 档另算）。"""
    import requests
    global _LAST_POLYGON_REQ
    import os
    min_interval = float(os.getenv("POLYGON_MIN_INTERVAL_SEC", "12"))
    wait = (_LAST_POLYGON_REQ + min_interval) - time.time()
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, timeout=timeout)
    _LAST_POLYGON_REQ = time.time()
    return resp


def fetch_polygon(start_date: str,
                  end_date: Optional[str] = None,
                  ticker: str = None) -> pd.DataFrame:
    if not POLYGON_API_KEY:
        logger.warning("POLYGON_API_KEY 未配置；跳过 Polygon")
        return pd.DataFrame()

    ticker = ticker or POLYGON_TICKER
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000"
        f"&apiKey={POLYGON_API_KEY}"
    )
    try:
        resp = _polygon_get(url)
        if resp.status_code != 200:
            logger.warning(f"Polygon HTTP {resp.status_code} for {ticker}")
            return pd.DataFrame()
        payload = resp.json()
        if payload.get("status") == "ERROR":
            logger.warning(f"Polygon API error: {payload.get('error') or payload.get('message')}")
            return pd.DataFrame()

        results = payload.get("results") or []
        if not results:
            logger.warning(f"Polygon 返回空 results for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(results).rename(columns={
            "o": "Open", "h": "High", "l": "Low",
            "c": "Close", "v": "Volume", "t": "Date",
        })
        df["Date"] = (pd.to_datetime(df["Date"], unit="ms", utc=True)
                        .dt.tz_convert("US/Eastern")
                        .dt.normalize().dt.tz_localize(None))
        return df.set_index("Date")[OHLCV_COLS].sort_index()
    except Exception as e:
        logger.error(f"Polygon 拉取异常: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# Yahoo Finance
# ----------------------------------------------------------------------
def fetch_yahoo(start_date: str,
                end_date: Optional[str] = None,
                ticker: str = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance 未安装；跳过 Yahoo")
        return pd.DataFrame()

    ticker = ticker or YAHOO_TICKER
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    try:
        # end_date is inclusive for this project; yfinance end is exclusive.
        yf_end = yahoo_exclusive_end(end_date) if end_date else None
        df = yf.download(ticker, start=start_date, end=yf_end,
                         progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[OHLCV_COLS].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "Date"
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date).normalize()]
        return df.sort_index()
    except Exception as e:
        logger.error(f"Yahoo 拉取异常: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# 离线代理 —— 仅用于沙箱 / 演示
# ----------------------------------------------------------------------
def _load_fred_spx(path) -> pd.DataFrame:
    """
    FRED 每日 SPX 收盘价；只有 Close。用前一日 Close 作 Open 构造保守 OHLC：
        Open=prev Close, High=max(O,C), Low=min(O,C)
    日内范围被压缩，但聚合到周/月时由"周内多日 Close 的 max/min"决定，与真实
    周/月范围相比仍属于"略偏窄"的保守近似。
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    val_col = "SP500" if "SP500" in df.columns else df.columns[1]
    df = df.rename(columns={date_col: "Date", val_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).set_index("Date").sort_index()

    df["Open"] = df["Close"].shift(1).fillna(df["Close"])
    df["High"] = df[["Open", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Close"]].min(axis=1)
    df["Volume"] = pd.NA  # FRED 不含成交量，明确标 NA 以免后续误用
    return df[OHLCV_COLS]


def fetch_offline(start_date: str,
                  end_date: Optional[str] = None) -> pd.DataFrame:
    """
    离线代理：用 FRED 每日 SPX 现货指数代替 ES 期货。两者趋势 99% 一致，
    不同处仅是基差（cost of carry，年化 < 1%），对 KDJ 信号判定无实质影响。
    """
    fred_path = DATA_DIR / "fred_sp500.csv"
    if not fred_path.exists():
        logger.warning(f"离线数据不存在：{fred_path}")
        return pd.DataFrame()

    df = _load_fred_spx(fred_path)
    start_ts = pd.Timestamp(start_date)
    df = df.loc[df.index >= start_ts]
    if end_date:
        df = df.loc[df.index <= pd.Timestamp(end_date)]
    return df


# ----------------------------------------------------------------------
# 总入口
# ----------------------------------------------------------------------
SOURCE_FUNCS = {
    "polygon": fetch_polygon,
    "yahoo": fetch_yahoo,
    "offline": fetch_offline,
}


def fetch_esmain_daily(start_date: str,
                       end_date: Optional[str] = None,
                       source_priority: Optional[List[str]] = None,
                       now_et=None) -> pd.DataFrame:
    """按优先级依次尝试，返回首个非空且看起来合理的结果。"""
    priority = source_priority or DATA_SOURCE_PRIORITY

    for src in priority:
        fn = SOURCE_FUNCS.get(src)
        if fn is None:
            logger.warning(f"未知数据源 {src!r}")
            continue
        df = fn(start_date, end_date)
        if df.empty:
            logger.warning(f"数据源 {src} 不可用，尝试下一个")
            continue
        if df["Close"].isna().all():
            logger.warning(f"数据源 {src} 返回全 NaN Close，跳过")
            continue
        df = drop_untrusted_trailing_bars(df, now_et=now_et)
        if df.empty:
            logger.warning(f"数据源 {src} 在排除未完成 bar 后为空，尝试下一个")
            continue
        logger.info(
            f"数据源 {src} 拉取成功，{len(df)} 行 "
            f"({df.index.min().date()} -> {df.index.max().date()})"
        )
        df.attrs["source"] = src
        return df

    return pd.DataFrame()
