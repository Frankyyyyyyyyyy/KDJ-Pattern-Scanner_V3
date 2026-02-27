import logging
import time
import requests as req
import pandas as pd
import yfinance as yf
import os
from config import POLYGON_API_KEY, DATA_SOURCE_PRIORITY, STOCK_LIST_FILE
from utils import get_last_completed_nyse_session_date, get_now_et

logger = logging.getLogger(__name__)


_LAST_POLYGON_REQUEST_AT = 0.0


def _polygon_get(url, timeout=15):
    global _LAST_POLYGON_REQUEST_AT

    try:
        min_interval = float(os.getenv('POLYGON_MIN_INTERVAL_SEC', '12'))
    except Exception:
        min_interval = 12.0

    now = time.time()
    wait_s = (_LAST_POLYGON_REQUEST_AT + min_interval) - now
    if wait_s > 0:
        time.sleep(wait_s)

    resp = req.get(url, timeout=timeout)
    _LAST_POLYGON_REQUEST_AT = time.time()
    return resp


def _fetch_polygon_prev_close_bar(ticker):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
    try:
        resp = _polygon_get(url, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        if data.get('status') == 'ERROR':
            return pd.DataFrame()

        results = data.get('results') or []
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 't': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
        df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.normalize().dt.tz_localize(None)
        df = df.set_index('Date')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index.name = 'Date'
        return df
    except Exception:
        return pd.DataFrame()

def fetch_polygon_bars(ticker, start_date, end_date=None, timespan='day'):
    """
    从 Polygon.io 获取单只股票的 OHLCV 数据。

    Args:
        ticker: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期，默认为今天
        timespan: 'day' | 'week'

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume (index=DatetimeIndex)
    """
    if end_date is None:
        end_date = get_last_completed_nyse_session_date()

    if not POLYGON_API_KEY:
        logger.warning("POLYGON_API_KEY 未配置，无法使用 Polygon 数据源")
        return pd.DataFrame()

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}"
        f"/{start_date}/{end_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )

    try:
        resp = _polygon_get(url, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Polygon HTTP {resp.status_code} for {ticker}")
            return pd.DataFrame()
        data = resp.json()

        if data.get('status') == 'ERROR':
            err_msg = data.get('error') or data.get('message') or 'Unknown error'
            logger.warning(f"Polygon API error for {ticker}: {err_msg}")
            return pd.DataFrame()

        if data.get('resultsCount', 0) == 0 or 'results' not in data:
            return pd.DataFrame()

        bars = data['results']
        df = pd.DataFrame(bars)

        # Polygon 字段映射: o=Open, h=High, l=Low, c=Close, v=Volume, t=timestamp(ms)
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 't': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
        df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.normalize().dt.tz_localize(None)
        df = df.set_index('Date')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index.name = 'Date'

        if timespan == 'day':
            try:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date) if end_date else None
                need_prev = (df.empty) or (end_ts is not None and df.index.max() < end_ts)
                if need_prev:
                    prev_df = _fetch_polygon_prev_close_bar(ticker)
                    if not prev_df.empty:
                        prev_date = prev_df.index.max()
                        if prev_date not in df.index:
                            in_range = (prev_date >= start_ts) and (end_ts is None or prev_date <= end_ts)
                            if in_range:
                                df = pd.concat([df, prev_df]).sort_index()
            except Exception:
                pass

        return df

    except Exception as e:
        logger.warning(f"Polygon API error for {ticker}: {e}")
        return pd.DataFrame()


def fetch_polygon_batch(tickers, start_date, end_date=None):
    """
    从 Polygon.io 批量获取多只股票数据。
    Polygon 免费版不支持批量接口，需逐个请求但有 5次/分钟限制。
    免费版限速：5 requests/minute，需控制请求频率。

    Returns:
        dict: {ticker: DataFrame}
    """
    result = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        logger.info(f"  Polygon: [{i+1}/{total}] {ticker}")
        df = fetch_polygon_bars(ticker, start_date, end_date)
        if not df.empty:
            result[ticker] = df
        else:
            logger.warning(f"  Polygon: {ticker} 无数据")

    return result


def fetch_yahoo_batch(tickers, start_date, yf_session=None):
    """
    从 Yahoo Finance 批量获取数据 (yf.download)。

    Returns:
        dict: {ticker: DataFrame}
    """
    result = {}

    download_kwargs = dict(
        tickers=tickers,
        start=start_date,
        interval="1d",
        auto_adjust=True,
        group_by='ticker',
        threads=False,
        progress=True,
    )
    if yf_session is not None:
        download_kwargs['session'] = yf_session

    try:
        all_data = yf.download(**download_kwargs)

        if all_data is None or all_data.empty:
            return result

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df_t = all_data.copy()
                else:
                    df_t = all_data[ticker].copy()
                df_t = df_t.dropna(how='all')
                if not df_t.empty:
                    result[ticker] = df_t
            except (KeyError, Exception):
                continue

    except Exception as e:
        logger.error(f"Yahoo batch download failed: {e}")

    return result


def fetch_all_data(tickers, start_date, yf_session=None, end_date=None):
    """
    多数据源获取，按优先级尝试:
    1. Polygon.io（国内直接可用，无需 VPN）
    2. Yahoo Finance（需 VPN）
    3. 本地缓存（当天有效）

    Returns:
        dict: {ticker: DataFrame}
    """
    cache_file = f"_cache_{get_now_et().strftime('%Y%m%d')}.pkl"

    disable_cache = os.getenv('DISABLE_CACHE', '').strip().lower() in {'1', 'true', 'yes'}
    expected_end_date = get_last_completed_nyse_session_date()
    expected_end_ts = pd.Timestamp(expected_end_date)

    all_result = {}

    # 尝试本地缓存
    if (not disable_cache) and os.path.exists(cache_file):
        try:
            cached = pd.read_pickle(cache_file)
            if isinstance(cached, dict) and len(cached) > 0:
                filtered = {}
                stale = []
                for k, v in cached.items():
                    if not isinstance(v, pd.DataFrame) or v.empty:
                        continue
                    try:
                        if v.index.max() >= expected_end_ts:
                            filtered[k] = v
                        else:
                            stale.append(k)
                    except Exception:
                        continue

                cached_keys = set(filtered.keys())
                requested_keys = set(tickers)
                if stale:
                    logger.info(f"本地缓存数据较旧，将重新拉取: {stale}")

                if requested_keys.issubset(cached_keys):
                    logger.info(f"从本地缓存加载数据: {cache_file} ({len(filtered)} tickers)")
                    return filtered

                all_result.update(filtered)
                logger.info(
                    f"从本地缓存加载部分数据: {cache_file} ({len(filtered)} tickers), 缺失 {len(requested_keys - cached_keys)}"
                )
        except Exception:
            pass

    for source in DATA_SOURCE_PRIORITY:
        missing = [t for t in tickers if t not in all_result]
        if not missing:
            break

        if source == 'polygon':
            logger.info(f"数据源 [Polygon.io]: 获取 {len(missing)} 只股票...")
            polygon_data = fetch_polygon_batch(missing, start_date, end_date=end_date)
            all_result.update(polygon_data)
            logger.info(f"  Polygon 获取成功: {len(polygon_data)}/{len(missing)}")

        elif source == 'yahoo':
            missing_after = [t for t in tickers if t not in all_result]
            if missing_after:
                logger.info(f"数据源 [Yahoo Finance]: 获取剩余 {len(missing_after)} 只股票...")
                yahoo_data = fetch_yahoo_batch(missing_after, start_date, yf_session)
                all_result.update(yahoo_data)
                logger.info(f"  Yahoo 获取成功: {len(yahoo_data)}/{len(missing_after)}")

    # 保存缓存
    if all_result and (not disable_cache):
        try:
            pd.to_pickle(all_result, cache_file)
            logger.info(f"数据已缓存: {cache_file} ({len(all_result)} tickers)")
        except Exception:
            pass

    # 汇总
    loaded = len(all_result)
    failed = [t for t in tickers if t not in all_result]
    logger.info(f"数据加载完成: {loaded}/{len(tickers)} 成功" +
                 (f", 失败: {failed}" if failed else ""))

    return all_result


def update_stock_metadata():
    """
    Updates stock_list.csv with Sector, Type and Average Volume.
    Returns the loaded DataFrame.
    """
    logger.info(f"Loading and updating metadata from {STOCK_LIST_FILE}...")

    try:
        try:
            df = pd.read_csv(STOCK_LIST_FILE, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(STOCK_LIST_FILE, encoding='gbk')
    except Exception as e:
        logger.error(f"Error reading {STOCK_LIST_FILE}: {e}")
        data = {
            'Ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'QQQ', 'SPY', 'IWM'],
            'Name_CN': ['苹果', '微软', '谷歌', '亚马逊', '英伟达', 'Meta', '特斯拉', '纳指100', '标普500', '罗素2000'],
            'Sector': ['Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Index ETF', 'Index ETF', 'Index ETF']
        }
        df = pd.DataFrame(data)

    if 'Sector' not in df.columns:
        df['Sector'] = 'Unknown'

    df['Type'] = df['Sector'].apply(
        lambda x: 'Index ETF' if 'Index' in str(x) or 'ETF' in str(x) else 'Stock'
    )

    # 保留已有 Avg_Volume（如果有），跳过逐个 API 调用获取 volume，节省 API 额度
    if 'Avg_Volume' not in df.columns:
        df['Avg_Volume'] = 0
    df = df.sort_values(by=['Type', 'Avg_Volume'], ascending=[True, False])
    df.to_csv(STOCK_LIST_FILE, index=False, encoding='utf-8-sig')
    logger.info("Metadata updated (skipped volume fetch to avoid rate limit).")
    return df


def validate_ohlc(df, ticker):
    """
    校验 OHLC 数据质量，清洗脏数据。

    检查项：
    - 移除 OHLCV 中包含 NaN、0、负值的行
    - 移除不满足 Low <= Open/Close <= High 的行
    - 移除成交量为 0 的行（非交易日数据残留）

    返回清洗后的 DataFrame 和被移除的行数。
    """
    original_len = len(df)

    # 1. 移除 NaN
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

    # 2. 移除 0 值和负值
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        df = df[df[col] > 0]
    df = df[df['Volume'] > 0]

    # 3. 移除不满足 Low <= min(Open,Close) 且 max(Open,Close) <= High 的行
    valid_mask = (
        (df['Low'] <= df['Open']) &
        (df['Low'] <= df['Close']) &
        (df['High'] >= df['Open']) &
        (df['High'] >= df['Close'])
    )
    df = df[valid_mask]

    removed = original_len - len(df)
    if removed > 0:
        logger.warning(f"  {ticker}: Removed {removed} invalid OHLC rows ({removed/original_len*100:.1f}%)")

    return df, removed
