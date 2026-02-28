import logging
import platform
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from config import STRATEGY_CONFIG, INDEX_TICKER
from utils import is_trading_day, manage_csv_archive, get_now_et, get_last_completed_nyse_session_date
from data import update_stock_metadata, update_price_cache, load_cached_data, validate_ohlc
from indicators import calculate_kdj, calculate_atr, get_weekly_kdj_snapshot
from patterns import identify_patterns
from filters import check_j_filter, check_prior_trend, check_divergence, check_pattern_filter
from reporting import send_email_report

# Import new modules for V2
from ml_predictor import train_and_predict

# ==========================================
# Logging Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# Worker Function for Parallel Processing
# ==========================================
def process_ticker(ticker, ticker_map, ticker_type_map, df, df_index, scan_start_dt, end_date_dt, cfg):
    """
    Worker function to process a single ticker.
    Returns a list of result dicts.
    """
    local_results = []
    try:
        # Data validation already done in main, but double check
        if df.empty:
            logger.warning(f"Worker received empty DF for {ticker}")
            return []

        # Calc Indicators
        df = calculate_kdj(df)
        df = calculate_atr(df, period=cfg['atr_period'])
        df['Bearish_Patterns'], df['Bullish_Patterns'] = identify_patterns(df, j_values=df['J'])

        # Pre-calc MA
        cfg = cfg
        # DEBUG: Check config
        # if ticker == 'AAPL':
        #    logger.info(f"Worker Config Check for AAPL: MA Short={cfg['ma_short']}, Long={cfg['ma_long']}")

        df['MA10'] = df['Close'].rolling(cfg['ma_short']).mean()
        df['MA20'] = df['Close'].rolling(cfg['ma_long']).mean()

        # Weekly KDJ
        df_weekly = df.resample('W-FRI').agg(
            {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        ).dropna()
        df_weekly = calculate_kdj(df_weekly)

        # Filter Scan Window
        if df.index.tz is not None:
            local_scan_start = scan_start_dt.tz_convert(df.index.tz)
        else:
            local_scan_start = scan_start_dt.tz_localize(None)

        scan_window = df[df.index >= local_scan_start]
        
        # DEBUG: Log scan window size
        if ticker == 'AAPL' and scan_window.empty:
             logger.warning(f"AAPL Scan Window Empty! ScanStart: {local_scan_start}, DF End: {df.index[-1]}")
        
        ml_mode = os.getenv("ML_MODE", "per_signal").strip().lower()
        ml_cache = {}
        retrain_days_raw = os.getenv('ML_PER_SIGNAL_RETRAIN_DAYS', '0').strip()
        retrain_days = 0
        if retrain_days_raw.isdigit():
            retrain_days = max(0, int(retrain_days_raw))
        last_train_date_by_dir = {}

        for date, row in scan_window.iterrows():
            loc_idx = df.index.get_loc(date)
            daily_j = row['J']

            wk_k, wk_d, wk_j = get_weekly_kdj_snapshot(df, date, df_weekly)

            # --- Index Data ---
            idx_daily_j = 50
            idx_bearish = []
            idx_bullish = []

            if date in df_index.index:
                idx_row = df_index.loc[date]
                idx_daily_j = idx_row['J']
                idx_bearish = idx_row['Bearish_Patterns']
                idx_bullish = idx_row['Bullish_Patterns']

            # --- Prev J values ---
            prev_daily_j = df.iloc[loc_idx - 1]['J'] if loc_idx > 0 else None
            prev_wk_idx = df_weekly.index[df_weekly.index < date]
            prev_wk_j = df_weekly.loc[prev_wk_idx[-1]]['J'] if not prev_wk_idx.empty else None

            # ==========================
            # CHECK SHORT & LONG SIGNALS
            # ==========================
            for direction in ("Short", "Long"):
                # Divergence Check
                divergence = check_divergence(direction, df, df_index, date, loc_idx)

                # J Filter
                if not check_j_filter(direction, daily_j, prev_daily_j, wk_j, prev_wk_j):
                    continue

                # Prior Trend Check
                if not check_prior_trend(direction, df, loc_idx):
                    continue

                # Pattern Check
                valid_patterns = check_pattern_filter(direction, df, row, loc_idx)
                if not valid_patterns:
                    continue

                # FOUND SIGNAL - Enrich with V2 features
                current_ml_prob = None

                if ml_mode == 'per_signal':
                    ml_lookback_days_raw = os.getenv("ML_LOOKBACK_DAYS", "120").strip()
                    ml_lookback_days = 120
                    if ml_lookback_days_raw.isdigit():
                        ml_lookback_days = int(ml_lookback_days_raw)

                    check_ai_threshold = end_date_dt - pd.DateOffset(days=ml_lookback_days)
                    date_naive = date.tz_localize(None) if date.tzinfo else date
                    check_naive = check_ai_threshold.tz_localize(None) if check_ai_threshold.tzinfo else check_ai_threshold
                    should_run_ml = (date_naive >= check_naive)

                    if should_run_ml:
                        reuse_ok = False
                        if retrain_days > 0:
                            last_dt = last_train_date_by_dir.get(direction)
                            if last_dt is not None:
                                try:
                                    reuse_ok = (date - last_dt) < pd.Timedelta(days=retrain_days)
                                except Exception:
                                    reuse_ok = False

                        if reuse_ok:
                            current_ml_prob = ml_cache.get(direction)
                        else:
                            ml_train_start = date - pd.DateOffset(years=3)
                            if not df.empty and ml_train_start < df.index[0]:
                                ml_train_start = df.index[0]

                            current_ml_prob, _ = train_and_predict(
                                ticker,
                                start_date=ml_train_start.strftime("%Y-%m-%d"),
                                end_date=date.strftime("%Y-%m-%d"),
                                df=df,
                                direction=direction
                            )
                            ml_cache[direction] = current_ml_prob
                            last_train_date_by_dir[direction] = date
                else:
                    if direction not in ml_cache:
                        ml_train_start = df.index[0]
                        ml_train_end = df.index[-1]
                        probs, _ = train_and_predict(
                            ticker,
                            start_date=str(ml_train_start.date()),
                            end_date=str(ml_train_end.date()),
                            df=df,
                            direction=direction
                        )
                        ml_cache[direction] = probs

                    current_ml_prob = ml_cache.get(direction)
                
                # Construct Result Item
                # We need to simulate add_result behavior but return dict
                res_item = {
                    'Date (日期)': date,
                    'Ticker (股票代码)': ticker,
                    'Name (名称)': ticker_map.get(ticker, ticker),
                    'Type (类型)': ticker_type_map.get(ticker, 'Unknown'),
                    'Volume (成交量)': int(row['Volume']),
                    'Direction (方向)': direction,
                    'Signal Strength': "Strong" if valid_patterns else "Weak", # Simplified logic, refined below
                    'Pattern (形态)': ", ".join(valid_patterns),
                    'Daily_J': round(daily_j, 2),
                    'Weekly_J': round(wk_j, 2),
                    'Price': round(row['Close'], 2),
                    'ATR': round(df.iloc[loc_idx]['ATR'], 2) if 'ATR' in df.columns else None,
                    # Stop Loss/Shares logic simplified for worker return
                    'Stop_Loss (止损)': 0, # Calculated below or later
                    # 'Fund_Score': fund_score,
                    'ML_Prob': 'N/A',
                    'ML_Detail': None,
                    # 'Sentiment': f"{news_sentiment:.2f}" if news_sentiment is not None else "N/A",
                    
                }
                
                # Refine Signal Strength
                strength = "Strong" if len(valid_patterns) > 0 else "Weak"
                if divergence: strength += " + Div"
                res_item['Signal Strength'] = strength

                # ML Enrichment
                if isinstance(current_ml_prob, dict) and current_ml_prob:
                    xgb_p = current_ml_prob.get('XGB', 0)
                    lgbm_p = current_ml_prob.get('LGBM', 0)
                    lr_p = current_ml_prob.get('LR', 0)

                    final_p = current_ml_prob.get('FINAL', None)
                    base_rate = current_ml_prob.get('BASE', None)
                    n_train = current_ml_prob.get('N_TRAIN', None)

                    if isinstance(final_p, (float, int)):
                        prob_val = float(final_p)
                        res_item['ML_Prob'] = f"{prob_val:.1%}"
                    else:
                        prob_val = 0
                        res_item['ML_Prob'] = "N/A"

                    detail_parts = [
                        f"XGB:{float(xgb_p):.2f}",
                        f"LGB:{float(lgbm_p):.2f}",
                        f"LR:{float(lr_p):.2f}",
                    ]
                    if isinstance(n_train, (int, float)):
                        detail_parts.append(f"N:{int(n_train)}")
                    if isinstance(base_rate, (int, float)):
                        detail_parts.append(f"Base:{float(base_rate):.2f}")
                    res_item['ML_Detail'] = "|".join(detail_parts)
                else:
                    prob_val = current_ml_prob if isinstance(current_ml_prob, (float, int)) else 0
                    res_item['ML_Prob'] = f"{prob_val:.1%}" if isinstance(current_ml_prob, (float, int)) else "N/A"
                    res_item['ML_Detail'] = "Single Model" if isinstance(current_ml_prob, (float, int)) else "N/A"
                    
                # Signal Strength Adjustments
                # if fund_score is not None and fund_score >= 6:
                #    res_item['Signal Strength'] += " + Fund"
                if prob_val > 0.5:
                     if direction == 'Long': res_item['Signal Strength'] += " + AI_Bull"
                     elif direction == 'Short': res_item['Signal Strength'] += " + AI_Bear"
                # if news_sentiment is not None:
                #    if direction == 'Long' and news_sentiment > 0.2: res_item['Signal Strength'] += " + News_Pos"
                #    elif direction == 'Short' and news_sentiment < -0.2: res_item['Signal Strength'] += " + News_Neg"
                    
                # Calculate Stop Loss / Shares (Replicate reporting.py logic briefly)
                atr_val = res_item['ATR']
                price = res_item['Price']
                if direction == 'Long':
                    stop_loss = price - (atr_val * 2.0)
                else:
                    stop_loss = price + (atr_val * 2.0)
                res_item['Stop_Loss (止损)'] = round(stop_loss, 2)
                
                risk_per_trade = 500
                risk_per_share = abs(price - stop_loss)
                if risk_per_share > 0:
                    shares = int(risk_per_trade / risk_per_share)
                    res_item['Suggested_Shares (建议股数)'] = shares
                    res_item['Position_Size (建议仓位$)'] = int(shares * price)
                else:
                    res_item['Suggested_Shares (建议股数)'] = 0
                    res_item['Position_Size (建议仓位$)'] = 0

                local_results.append(res_item)
                
    except Exception:
        logger.exception(f"Worker crashed while processing {ticker}")

    return local_results

def run_strategy():
    try:
        # 0. Market Calendar Check
        if not is_trading_day():
            logger.info("今天美股休市，跳过扫描任务。")
            return

        # 0. IP Geolocation & Regional Config (Removed for V2 Pure Polygon Mode)
        # country_code, country_name, ip_addr, yahoo_domain = detect_ip_country()
        # logger.info(f"当前网络环境: {country_name} ({country_code}), IP: {ip_addr}")
        # logger.info(f"使用 Yahoo Finance 区域: {yahoo_domain}")
        # yf_session = configure_yf_session(yahoo_domain)
        
        cfg_run = STRATEGY_CONFIG.copy()

        # 1. Update Metadata
        df_stocks = update_stock_metadata()
        ticker_map = dict(zip(df_stocks['Ticker'], df_stocks['Name_CN']))
        ticker_type_map = dict(zip(df_stocks['Ticker'], df_stocks['Type']))

        limit_tickers_raw = os.getenv("LIMIT_TICKERS", "").strip()
        if limit_tickers_raw:
            limit_tickers = [t.strip().upper() for t in limit_tickers_raw.split(",") if t.strip()]
            tickers_to_scan = [t for t in df_stocks['Ticker'].tolist() if t.upper() in set(limit_tickers)]
            logger.info(f"LIMIT_TICKERS enabled: {tickers_to_scan}")
            os.environ['EMAIL_UNIVERSE'] = ','.join(tickers_to_scan)
        else:
            tickers_to_scan = df_stocks['Ticker'].tolist()
            os.environ['EMAIL_UNIVERSE'] = 'ALL'

        target_end_date_str = get_last_completed_nyse_session_date()
        logger.info(f"本次扫描目标交易日: {target_end_date_str}")

        # 2. Prepare Date Range
        end_date_dt = pd.Timestamp(target_end_date_str)
        start_date_dt = end_date_dt - pd.DateOffset(years=1, months=6)
        start_date_str = start_date_dt.strftime("%Y-%m-%d")
        
        # 3. 年度价格缓存：先增量拉取价格到 CSV，再从 CSV 读取计算
        all_tickers = list(dict.fromkeys(tickers_to_scan + [INDEX_TICKER]))

        end_date_str = target_end_date_str
        update_price_cache(all_tickers, start_date_str, end_date_str)
        ticker_data = load_cached_data(all_tickers, start_date_str, end_date_str)

        if not ticker_data:
            logger.critical("无法获取任何数据。请检查网络或 API Key。")
            return
        
        # DEBUG: Check AAPL data specifically
        if 'AAPL' in tickers_to_scan:
            if 'AAPL' in ticker_data:
                aapl_df = ticker_data['AAPL']
                logger.info(f"AAPL Data Loaded: {len(aapl_df)} rows. Last Date: {aapl_df.index[-1]}")
            else:
                logger.warning("AAPL Data NOT FOUND in fetch_all_data result!")

        # 提取单只 ticker 数据的辅助函数
        def extract_ticker_data(ticker):
            return ticker_data.get(ticker, pd.DataFrame())

        # 4. Process Index Data (QQQ)
        logger.info(f"Processing Index {INDEX_TICKER}...")
        try:
            df_index = extract_ticker_data(INDEX_TICKER)
            if df_index.empty:
                logger.critical(f"No data for index {INDEX_TICKER}. Strategy cannot run.")
                return

            # 数据质量校验
            df_index, _ = validate_ohlc(df_index, INDEX_TICKER)
            df_index = calculate_kdj(df_index)
            df_index = calculate_atr(df_index, period=cfg_run['atr_period'])
            df_index['Bearish_Patterns'], df_index['Bullish_Patterns'] = identify_patterns(df_index)
        except Exception as e:
            logger.critical(f"Failed to process index data. Strategy cannot run. {e}")
            return

        results = []

        # Scan Range: Last 1 Year
        scan_start_dt = end_date_dt - pd.DateOffset(years=1)
        if df_index.index.tz is not None:
            scan_start_dt = scan_start_dt.tz_localize(df_index.index.tz)

        # 5. Parallel Processing Loop
        
        # Detect Hardware Platform
        system_platform = platform.system()
        machine_arch = platform.machine()
        cpu_count = multiprocessing.cpu_count()
        
        # Determine Parallel Strategy
        # Mac M1/M2/M3 (ARM64) works best with 'fork' (default on Unix) or 'spawn'.
        # For dataframes, 'fork' is faster but can be unstable with some libraries.
        # We will use ProcessPoolExecutor which defaults to a safe method.
        
        max_workers = max(1, cpu_count - 1) # Leave 1 core for OS
        max_workers_env = os.getenv("MAX_WORKERS", "").strip()
        if max_workers_env.isdigit():
            max_workers = max(1, int(max_workers_env))
        if system_platform == 'Darwin' and 'arm64' in machine_arch:
            logger.info(f"🚀 Detected Apple Silicon ({machine_arch}). Optimizing for {cpu_count} cores.")
            # M3 allows high parallelism
        else:
            logger.info(f"💻 Detected {system_platform} ({machine_arch}). Using {max_workers} workers.")

        max_workers = min(max_workers, max(1, len(tickers_to_scan)))
        logger.info(f"Starting parallel scanning with {max_workers} workers...")
        
        # Prepare arguments for workers
        futures = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ticker in tickers_to_scan:
                df = extract_ticker_data(ticker)
                
                if df.empty:
                    continue
                    
                # Validate here to save worker overhead
                df, removed = validate_ohlc(df, ticker)
                if df.empty:
                    continue
                    
                # Submit task
                futures.append(
                    executor.submit(
                        process_ticker, 
                        ticker, ticker_map, ticker_type_map, 
                        df, df_index, scan_start_dt, end_date_dt, cfg_run
                    )
                )
                
            # Collect results
            total_tasks = len(futures)
            completed_tasks = 0
            
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        results.extend(res)
                except Exception as e:
                    logger.error(f"Worker exception: {e}")
                
                completed_tasks += 1
                if completed_tasks % 5 == 0:
                    logger.info(f"Progress: {completed_tasks}/{total_tasks} tickers scanned.")

        # 5. Output with CSV archival
        target_end_date = pd.Timestamp(target_end_date_str).date()

        cols = ['Date (日期)', 'Ticker (股票代码)', 'Name (名称)', 'Type (类型)', 'Volume (成交量)',
                'Signal Strength', 'Direction (方向)', 'ML_Prob', 'ML_Detail',
                'Pattern (形态)', 'Daily_J', 'Weekly_J', 'Price', 'ATR',
                'Stop_Loss (止损)', 'Suggested_Shares (建议股数)', 'Position_Size (建议仓位$)']

        def build_no_signal_row(date_str: str):
            return {
                'Date (日期)': date_str,
                'Ticker (股票代码)': 'NO_SIGNAL',
                'Name (名称)': '无信号',
                'Type (类型)': 'Info',
                'Volume (成交量)': 0,
                'Signal Strength': 'No Signal',
                'Direction (方向)': 'N/A',
                'ML_Prob': 'N/A',
                'ML_Detail': 'N/A',
                'Pattern (形态)': '无信号',
                'Daily_J': None,
                'Weekly_J': None,
                'Price': None,
                'ATR': None,
                'Stop_Loss (止损)': None,
                'Suggested_Shares (建议股数)': None,
                'Position_Size (建议仓位$)': None,
            }

        if results:
            df_res = pd.DataFrame(results)
            
            # Ensure Date column is proper datetime
            df_res['Date (日期)'] = pd.to_datetime(df_res['Date (日期)'])
            
            df_res = df_res.sort_values(
                by=['Date (日期)', 'Direction (方向)', 'Type (类型)', 'Ticker (股票代码)'],
                ascending=[False, False, True, True]
            )
            
            # Ensure all columns exist
            for col in cols:
                if col not in df_res.columns:
                    df_res[col] = None

            df_res = df_res[cols]

            logger.info("=" * 60)
            logger.info("UNIFIED TRADING SIGNALS (V2 PARALLEL)")
            logger.info("=" * 60)
            print(df_res.head(20).to_string(index=False))

            count_target = 0
            try:
                df_res_dates = pd.to_datetime(df_res['Date (日期)'], errors='coerce')
                count_target = int((df_res_dates.dt.date == target_end_date).sum())
                logger.info(f"目标交易日 {target_end_date_str} 信号数: {count_target}")
            except Exception:
                pass

            if count_target == 0:
                df_res = pd.concat([df_res, pd.DataFrame([build_no_signal_row(target_end_date_str)])], ignore_index=True)
                df_res['Date (日期)'] = pd.to_datetime(df_res['Date (日期)'], errors='coerce')
                df_res = df_res.sort_values(
                    by=['Date (日期)', 'Direction (方向)', 'Type (类型)', 'Ticker (股票代码)'],
                    ascending=[False, False, True, True]
                )
                df_res = df_res[cols]

            # --- CSV 归档逻辑 ---
            csv_file = manage_csv_archive(df_res)

            # 邮件功能已禁用，信号通过 Web Dashboard 查看
            # Email disabled — view signals via Web Dashboard (python server.py)
            logger.info(f"扫描完成，共 {len(df_res)} 条信号已保存至 {csv_file}")
            logger.info(f"Scan complete. {len(df_res)} signals saved to {csv_file}")
        else:
            logger.info(f"目标交易日 {target_end_date_str} 信号数: 0")
            logger.info("No signals found.")

            df_res = pd.DataFrame([build_no_signal_row(target_end_date_str)])
            for col in cols:
                if col not in df_res.columns:
                    df_res[col] = None
            df_res = df_res[cols]
            csv_file = manage_csv_archive(df_res)
            logger.info(f"当前季度 CSV 已写入无信号占位: {csv_file}")

    except Exception as e:
        logger.critical(f"Strategy execution failed: {e}", exc_info=True)
        # Re-raise so that GitHub Action knows it failed, but we get logs
        raise e

if __name__ == "__main__":
    # On macOS, 'spawn' is the default start method for Python 3.8+.
    # We need to ensure this is guarded by if __name__ == "__main__"
    multiprocessing.set_start_method("spawn", force=True) # Switch to 'spawn' for stability
    run_strategy()
