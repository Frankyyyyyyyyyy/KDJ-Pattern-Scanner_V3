import logging
import time
import requests as req
import pandas as pd
import yfinance as yf
import os
from config import POLYGON_API_KEY, DATA_SOURCE_PRIORITY, STOCK_LIST_FILE, PRICE_CACHE_DIR
from utils import get_last_completed_nyse_session_date

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

    expected_end_ts = None
    try:
        expected_end_ts = pd.Timestamp(end_date).normalize()
        last_completed_ts = pd.Timestamp(get_last_completed_nyse_session_date()).normalize()
        if expected_end_ts > last_completed_ts:
            expected_end_ts = last_completed_ts
    except Exception:
        expected_end_ts = None

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

            if expected_end_ts is not None and not df.empty:
                try:
                    last_ts = pd.Timestamp(df.index.max()).normalize()
                    if last_ts < expected_end_ts:
                        logger.warning(
                            f"Polygon 数据滞后: {ticker} last={last_ts.date()} < expected={expected_end_ts.date()}，将回退到 Yahoo"
                        )
                        return pd.DataFrame()
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
    2. Yahoo Finance（备选）

    Returns:
        dict: {ticker: DataFrame}
    """
    all_result = {}

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

    if 'yahoo' in DATA_SOURCE_PRIORITY and len(all_result) == len(tickers):
        logger.info("数据源 [Yahoo Finance]: 无需回退（Polygon 数据已就绪）")

    # 汇总
    loaded = len(all_result)
    failed = [t for t in tickers if t not in all_result]
    logger.info(f"数据加载完成: {loaded}/{len(tickers)} 成功" +
                 (f", 失败: {failed}" if failed else ""))

    return all_result


def _normalize_ohlcv_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = [c[-1] for c in df.columns]
        except Exception:
            pass
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = 'Date'

    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.DataFrame()
    df = df[cols].dropna(how='all')
    return df


def _price_cache_path(year: int) -> str:
    return os.path.join(PRICE_CACHE_DIR, f"prices_{year}.csv")


def _read_price_cache(year: int) -> pd.DataFrame:
    path = _price_cache_path(year)
    if not os.path.exists(path):
        return pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source'])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source'])

    if df.empty:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source'])

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


def _write_price_cache(year: int, df: pd.DataFrame) -> str:
    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)
    path = _price_cache_path(year)
    df_out = df.copy()
    if 'Date' in df_out.columns:
        df_out['Date'] = pd.to_datetime(df_out['Date'], errors='coerce')
        df_out = df_out.dropna(subset=['Date'])
        df_out['Date'] = df_out['Date'].dt.strftime('%Y-%m-%d')
    df_out.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def update_price_cache_year(tickers, start_date, end_date):
    if end_date is None:
        end_date = get_last_completed_nyse_session_date()

    end_ts = pd.Timestamp(end_date).normalize()
    year = int(end_ts.year)
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    start_ts = pd.Timestamp(start_date).normalize()
    start_ts = max(start_ts, year_start)
    end_ts = min(end_ts, year_end)

    if start_ts > end_ts:
        return _price_cache_path(year)

    os.makedirs(PRICE_CACHE_DIR, exist_ok=True)
    cache_df = _read_price_cache(year)
    if cache_df.empty:
        cache_df = pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source'])

    try:
        if 'Date' in cache_df.columns:
            cache_df['Date'] = pd.to_datetime(cache_df['Date'], errors='coerce')
    except Exception:
        pass

    last_by_ticker = {}
    try:
        if not cache_df.empty and 'Ticker' in cache_df.columns and 'Date' in cache_df.columns:
            tmp = cache_df.dropna(subset=['Date', 'Ticker'])
            if not tmp.empty:
                last_by_ticker = tmp.groupby('Ticker')['Date'].max().to_dict()
    except Exception:
        last_by_ticker = {}

    polygon_ok = {}
    yahoo_needed = []
    yahoo_start_by_ticker = {}
    for ticker in tickers:
        last_dt = last_by_ticker.get(ticker)
        fetch_start = start_ts
        if last_dt is not None and pd.notna(last_dt):
            try:
                last_norm = pd.Timestamp(last_dt).normalize()
                fetch_start = max(fetch_start, last_norm + pd.Timedelta(days=1))
            except Exception:
                fetch_start = start_ts

        if fetch_start > end_ts:
            continue

        df_poly = fetch_polygon_bars(ticker, fetch_start.strftime('%Y-%m-%d'), end_ts.strftime('%Y-%m-%d'))
        df_poly = _normalize_ohlcv_df(df_poly)
        if not df_poly.empty:
            polygon_ok[ticker] = (fetch_start, df_poly)
        else:
            yahoo_needed.append(ticker)
            yahoo_start_by_ticker[ticker] = fetch_start

    new_rows = []
    for ticker, (fetch_start, df_poly) in polygon_ok.items():
        df_add = df_poly[(df_poly.index >= fetch_start) & (df_poly.index <= end_ts)].copy()
        if df_add.empty:
            continue
        df_add = df_add.reset_index()
        df_add.insert(1, 'Ticker', ticker)
        df_add['Source'] = 'polygon'
        new_rows.append(df_add)

    if yahoo_needed:
        yahoo_min_start = min(yahoo_start_by_ticker.values())
        yahoo_data = fetch_yahoo_batch(yahoo_needed, yahoo_min_start.strftime('%Y-%m-%d'), yf_session=None)
        for ticker in yahoo_needed:
            df_y = _normalize_ohlcv_df(yahoo_data.get(ticker, pd.DataFrame()))
            if df_y.empty:
                continue
            fetch_start = yahoo_start_by_ticker.get(ticker, start_ts)
            df_add = df_y[(df_y.index >= fetch_start) & (df_y.index <= end_ts)].copy()
            if df_add.empty:
                continue
            df_add = df_add.reset_index()
            df_add.insert(1, 'Ticker', ticker)
            df_add['Source'] = 'yahoo'
            new_rows.append(df_add)

    if new_rows:
        df_new = pd.concat(new_rows, ignore_index=True)
        cache_df = pd.concat([cache_df, df_new], ignore_index=True)

    if not cache_df.empty:
        if 'Date' in cache_df.columns:
            cache_df['Date'] = pd.to_datetime(cache_df['Date'], errors='coerce')
        cache_df = cache_df.dropna(subset=['Date', 'Ticker'])
        cache_df['Date'] = cache_df['Date'].dt.normalize()
        cache_df = cache_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
        cache_df = cache_df.sort_values(by=['Date', 'Ticker'])

    return _write_price_cache(year, cache_df)


def update_price_cache(tickers, start_date, end_date):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    years = list(range(int(start_ts.year), int(end_ts.year) + 1))
    paths = []
    for y in years:
        year_start = pd.Timestamp(f"{y}-01-01")
        year_end = pd.Timestamp(f"{y}-12-31")
        seg_start = max(start_ts, year_start)
        seg_end = min(end_ts, year_end)
        if seg_start <= seg_end:
            paths.append(update_price_cache_year(tickers, seg_start.strftime('%Y-%m-%d'), seg_end.strftime('%Y-%m-%d')))
    return paths


def load_cached_data(tickers, start_date, end_date):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    years = list(range(int(start_ts.year), int(end_ts.year) + 1))
    dfs = []
    for y in years:
        df_y = _read_price_cache(y)
        if df_y is None or df_y.empty:
            continue
        dfs.append(df_y)

    if not dfs:
        return {t: pd.DataFrame() for t in tickers}

    df_all = pd.concat(dfs, ignore_index=True)
    df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
    df_all = df_all.dropna(subset=['Date', 'Ticker'])
    df_all['Date'] = df_all['Date'].dt.normalize()
    df_all = df_all[(df_all['Date'] >= start_ts) & (df_all['Date'] <= end_ts)]
    df_all = df_all[df_all['Ticker'].isin(set(tickers))]

    result = {}
    for t in tickers:
        df_t = df_all[df_all['Ticker'] == t]
        if df_t.empty:
            result[t] = pd.DataFrame()
            continue
        df_t = df_t.sort_values(by='Date')
        df_t = df_t.set_index('Date')
        df_t.index.name = 'Date'
        df_t = df_t[['Open', 'High', 'Low', 'Close', 'Volume']]
        result[t] = df_t

    return result


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
