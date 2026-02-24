import logging
import pandas_market_calendars as mcal
import pytz
from datetime import datetime
import pandas as pd
import os
import shutil
import glob
from config import ARCHIVE_DIR

logger = logging.getLogger(__name__)


_EASTERN_TZ = pytz.timezone('US/Eastern')


def get_now_et():
    return datetime.now(_EASTERN_TZ)


def get_last_completed_nyse_session_date(now_et=None):
    try:
        if now_et is None:
            now_et = get_now_et()
        now_et_ts = pd.Timestamp(now_et)

        nyse = mcal.get_calendar('NYSE')
        start_date = (now_et_ts - pd.Timedelta(days=14)).date()
        end_date = now_et_ts.date()
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)

        if schedule.empty:
            return end_date.isoformat()

        schedule = schedule.copy()
        schedule['market_close'] = schedule['market_close'].dt.tz_convert(_EASTERN_TZ)

        last_session = schedule.index[-1]
        last_close = schedule.loc[last_session, 'market_close']
        if now_et_ts >= last_close:
            return pd.Timestamp(last_session).date().isoformat()

        if len(schedule.index) >= 2:
            prev_session = schedule.index[-2]
            return pd.Timestamp(prev_session).date().isoformat()

        return pd.Timestamp(last_session).date().isoformat()
    except Exception as e:
        logger.warning(f"无法计算最近已收盘交易日，回退到美东日期: {e}")
        try:
            return (now_et or get_now_et()).date().isoformat()
        except Exception:
            return pd.Timestamp.now().date().isoformat()


def is_trading_day():
    """
    检查今天是否为美股交易日（基于 NYSE 日历）。
    使用美东时间 (US/Eastern) 判断。
    """
    if os.getenv('FORCE_RUN', '').strip().lower() in {'1', 'true', 'yes'}:
        return True
    try:
        nyse = mcal.get_calendar('NYSE')
        now_et = get_now_et()
        today_date = now_et.date()
        
        # 获取包含今天的日程安排
        # schedule 接受 start, end。我们取今天前后各一天以防万一
        schedule = nyse.schedule(start_date=today_date, end_date=today_date)
        
        if schedule.empty:
            logger.info(f"📅 非交易日 (Holiday/Weekend): {today_date}")
            return False
            
        logger.info(f"📅 交易日确认: {today_date} (Market Open)")
        return True
    except Exception as e:
        logger.warning(f"无法获取市场日历，默认继续运行: {e}")
        return True  # 如果日历获取失败，为了安全起见，还是运行吧


def get_current_quarter():
    """返回当前季度标识，如 '2026Q1'"""
    now = pd.Timestamp(get_now_et().replace(tzinfo=None))
    quarter = (now.month - 1) // 3 + 1
    return f"{now.year}Q{quarter}"


def get_quarter_csv_filename():
    """返回当前季度的 CSV 文件名，如 'signals_2026Q1.csv'"""
    return f"signals_{get_current_quarter()}.csv"


def _archive_old_quarter_files(current_csv):
    """将非当前季度的 signals_*.csv 文件移入 archive/ 目录"""
    for f in glob.glob('signals_*Q*.csv'):
        if f != current_csv:
            dest = os.path.join(ARCHIVE_DIR, f)
            if not os.path.exists(dest):
                shutil.move(f, dest)
                logger.info(f"已归档历史文件: {f} → {dest}")


def manage_csv_archive(df_new):
    """
    CSV 归档管理:
    - 当前季度的信号追加到 signals_YYYYQN.csv
    - CSV 只保留最近 3 个月的信号
    - 每个季度自动创建新文件
    - 上一季度的文件移入 archive/ 目录

    Args:
        df_new: 本次扫描的新信号 DataFrame

    Returns:
        当前季度的 CSV 文件路径
    """
    # 确保归档目录存在
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    current_csv = get_quarter_csv_filename()
    now = pd.Timestamp(get_now_et().replace(tzinfo=None))
    three_months_ago = (now - pd.DateOffset(months=3)).date()

    # 读取现有当前季度文件（如果有）
    if os.path.exists(current_csv):
        try:
            df_existing = pd.read_csv(current_csv, encoding='utf-8-sig')
            df_existing['Date (日期)'] = pd.to_datetime(df_existing['Date (日期)']).dt.date
            if 'ML_Prob' not in df_existing.columns:
                df_existing['ML_Prob'] = 'N/A'
            if 'ML_Detail' not in df_existing.columns:
                df_existing['ML_Detail'] = 'N/A'
        except Exception:
            df_existing = pd.DataFrame()
    else:
        df_existing = pd.DataFrame()

    target_cols = list(df_new.columns)

    # 合并新旧数据，去重
    if not df_existing.empty:
        # 统一日期格式为字符串，避免 Timestamp vs date 类型不一致导致去重失败
        df_existing['Date (日期)'] = pd.to_datetime(df_existing['Date (日期)']).dt.strftime('%Y-%m-%d')
        df_new_copy = df_new.copy()
        df_new_copy['Date (日期)'] = pd.to_datetime(df_new_copy['Date (日期)']).dt.strftime('%Y-%m-%d')
        df_combined = pd.concat([df_existing, df_new_copy], ignore_index=True)
        # 按 Date + Ticker + Direction 去重，保留最新的
        df_combined = df_combined.drop_duplicates(
            subset=['Date (日期)', 'Ticker (股票代码)', 'Direction (方向)'],
            keep='last'
        )
    else:
        df_combined = df_new.copy()
        df_combined['Date (日期)'] = pd.to_datetime(df_combined['Date (日期)']).dt.strftime('%Y-%m-%d')

    for col in target_cols:
        if col not in df_combined.columns:
            df_combined[col] = None
    df_combined = df_combined[target_cols]

    for col in ['ML_Prob', 'ML_Detail']:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna('N/A')

    # 只保留最近 3 个月
    df_combined = df_combined[pd.to_datetime(df_combined['Date (日期)']).dt.date >= three_months_ago]

    # 排序并保存
    df_combined = df_combined.sort_values(
        by=['Direction (方向)', 'Type (类型)', 'Ticker (股票代码)', 'Date (日期)'],
        ascending=[False, True, True, False]
    )
    df_combined.to_csv(current_csv, index=False, encoding='utf-8-sig')
    logger.info(f"当前季度 CSV 已更新: {current_csv} ({len(df_combined)} 条信号)")

    # 不再写出 unified_signals.csv，避免重复产物

    # 归档上一季度文件
    _archive_old_quarter_files(current_csv)

    return current_csv
