import os
import pandas as pd

from data import fetch_polygon_bars
from ml_predictor import train_and_predict, get_triple_barrier_labels
from utils import get_now_et


def _env_int(name, default):
    raw = os.getenv(name, str(default)).strip()
    if raw.isdigit():
        return int(raw)
    return default


def _env_float(name, default):
    raw = os.getenv(name, str(default)).strip()
    try:
        return float(raw)
    except Exception:
        return default


def _calibration_table(df_eval, bin_size=0.1):
    out = []
    bins = [i * bin_size for i in range(int(1 / bin_size) + 1)]
    df_eval = df_eval.copy()
    df_eval['bin'] = pd.cut(df_eval['p'], bins=bins, include_lowest=True, right=True)

    for b, g in df_eval.groupby('bin', observed=True):
        if g.empty:
            continue
        out.append(
            {
                'bin': str(b),
                'n': int(len(g)),
                'p_mean': float(g['p'].mean()),
                'hit_rate': float(g['y'].mean()),
            }
        )

    return pd.DataFrame(out)


def walkforward_audit_from_df(
    ticker,
    direction,
    df,
    eval_days=30,
    train_years=3,
    horizon=10,
    profit_take=0.03,
    stop_loss=0.015,
):
    df = df.copy().sort_index()
    labels = get_triple_barrier_labels(
        df,
        profit_take=profit_take,
        stop_loss=stop_loss,
        horizon=horizon,
        direction=direction,
    )
    valid_dates = labels.dropna().index
    if len(valid_dates) == 0:
        return None

    eval_dates = valid_dates[-eval_days:]
    rows = []
    for dt in eval_dates:
        train_start = dt - pd.DateOffset(years=train_years)
        if train_start < df.index[0]:
            train_start = df.index[0]

        probs, _ = train_and_predict(
            ticker,
            start_date=str(train_start.date()),
            end_date=str(dt.date()),
            df=df,
            direction=direction,
        )
        if not probs or not isinstance(probs, dict):
            continue

        p = probs.get('FINAL', None)
        if not isinstance(p, (float, int)):
            continue

        rows.append(
            {
                'date': dt,
                'p': float(p),
                'y': float(labels.loc[dt]),
                'n_train': int(probs.get('N_TRAIN', 0)),
                'base': float(probs.get('BASE', 0.5)),
            }
        )

    df_eval = pd.DataFrame(rows)
    if df_eval.empty:
        return None

    brier = float(((df_eval['p'] - df_eval['y']) ** 2).mean())
    hit_rate = float(df_eval['y'].mean())
    p_mean = float(df_eval['p'].mean())

    return {
        'ticker': ticker,
        'direction': direction,
        'eval_rows': int(len(df_eval)),
        'avg_p': p_mean,
        'hit_rate': hit_rate,
        'brier': brier,
        'calibration': _calibration_table(df_eval, bin_size=0.1),
    }


def build_ml_email_section(df_results, ticker_data):
    recent_days_raw = os.getenv('EMAIL_LOOKBACK_DAYS', '14')
    recent_days = 14
    if str(recent_days_raw).strip().isdigit():
        recent_days = int(str(recent_days_raw).strip())

    max_pairs_raw = os.getenv('EMAIL_ML_AUDIT_MAX_PAIRS', '1').strip()
    max_pairs = 1
    if max_pairs_raw.isdigit():
        max_pairs = max(0, int(max_pairs_raw))

    eval_days_raw = os.getenv('EMAIL_ML_AUDIT_EVAL_DAYS', '10').strip()
    eval_days = 10
    if eval_days_raw.isdigit():
        eval_days = max(10, int(eval_days_raw))

    train_years_raw = os.getenv('EMAIL_ML_AUDIT_TRAIN_YEARS', '3').strip()
    train_years = 3
    if train_years_raw.isdigit():
        train_years = max(1, int(train_years_raw))

    enable_walkforward = os.getenv('EMAIL_ML_WALKFORWARD', '1').strip().lower() in {'1', 'true', 'yes'}

    df = df_results.copy()
    df['Date (日期)'] = pd.to_datetime(df['Date (日期)']).dt.normalize()
    cutoff = pd.Timestamp(get_now_et().replace(tzinfo=None)).normalize() - pd.Timedelta(days=recent_days)
    df_recent = df[df['Date (日期)'] >= cutoff].copy()

    p = pd.to_numeric(df_recent.get('ML_Prob', pd.Series(dtype=str)).astype(str).str.rstrip('%'), errors='coerce')
    ml_count = int(p.notna().sum())
    total = int(len(df_recent))
    na_ratio = float(1 - (ml_count / total)) if total else 1.0
    p_valid = p.dropna()

    stats_line = 'N/A'
    if len(p_valid):
        stats_line = f"count={len(p_valid)} min={p_valid.min():.1f}% median={p_valid.median():.1f}% max={p_valid.max():.1f}%"

    html = f'''
    <div style="margin-top:30px;padding-top:15px;border-top:1px dashed #eee;">
      <h3 style="color:#2c3e50;margin:0 0 10px 0;">ML 诊断</h3>
      <div style="font-size:12px;color:#7f8c8d;line-height:1.6;">
        <div>Recent({recent_days}d) 信号数: <strong>{total}</strong>，含 ML 概率: <strong>{ml_count}</strong>，N/A 比例: <strong>{na_ratio:.0%}</strong></div>
        <div>ML_Prob 分布: {stats_line}</div>
      </div>
    '''

    if not enable_walkforward or max_pairs <= 0:
        html += '</div>'
        return html

    pairs = (
        df_recent[['Ticker (股票代码)', 'Direction (方向)', 'ML_Prob']]
        .dropna(subset=['Ticker (股票代码)', 'Direction (方向)'])
        .copy()
    )
    pairs['p'] = pd.to_numeric(pairs['ML_Prob'].astype(str).str.rstrip('%'), errors='coerce')
    pairs = pairs[pairs['p'].notna()]
    if pairs.empty:
        html += '</div>'
        return html

    pair_counts = (
        pairs.groupby(['Ticker (股票代码)', 'Direction (方向)'])
        .size()
        .sort_values(ascending=False)
        .head(max_pairs)
        .reset_index()
    )

    html += '<div style="margin-top:12px;font-size:12px;color:#7f8c8d;">Walk-forward 校准(样本较少，仅作健康检查)：</div>'
    html += '<table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:12px;">'
    html += '<thead><tr><th style="background:#34495e;color:#fff;padding:8px;">Ticker</th><th style="background:#34495e;color:#fff;padding:8px;">Dir</th><th style="background:#34495e;color:#fff;padding:8px;">Rows</th><th style="background:#34495e;color:#fff;padding:8px;">AvgP</th><th style="background:#34495e;color:#fff;padding:8px;">Hit</th><th style="background:#34495e;color:#fff;padding:8px;">Brier</th></tr></thead><tbody>'

    for _, r in pair_counts.iterrows():
        t = str(r['Ticker (股票代码)']).strip()
        d = str(r['Direction (方向)']).strip()
        df_t = ticker_data.get(t)
        if df_t is None or df_t.empty:
            continue
        audit = walkforward_audit_from_df(t, d, df_t, eval_days=eval_days, train_years=train_years)
        if not audit:
            continue

        html += (
            f"<tr>"
            f"<td style=\"padding:8px;border-bottom:1px solid #eee;\">{t}</td>"
            f"<td style=\"padding:8px;border-bottom:1px solid #eee;\">{d}</td>"
            f"<td style=\"padding:8px;border-bottom:1px solid #eee;\">{audit['eval_rows']}</td>"
            f"<td style=\"padding:8px;border-bottom:1px solid #eee;\">{audit['avg_p']:.3f}</td>"
            f"<td style=\"padding:8px;border-bottom:1px solid #eee;\">{audit['hit_rate']:.3f}</td>"
            f"<td style=\"padding:8px;border-bottom:1px solid #eee;\">{audit['brier']:.4f}</td>"
            f"</tr>"
        )

    html += '</tbody></table></div>'
    return html


def main():
    ticker = os.getenv('TICKER', 'SPY').strip().upper()
    direction = os.getenv('DIRECTION', 'Short').strip().title()
    if direction not in {'Long', 'Short'}:
        raise ValueError('DIRECTION must be Long or Short')

    eval_days = _env_int('EVAL_DAYS', 60)
    train_years = _env_int('TRAIN_YEARS', 3)
    horizon = _env_int('HORIZON', 10)
    profit_take = _env_float('PROFIT_TAKE', 0.03)
    stop_loss = _env_float('STOP_LOSS', 0.015)

    end_date = os.getenv('END_DATE', '').strip() or None
    start_date = os.getenv('START_DATE', '').strip() or None

    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    end_ts = pd.Timestamp(end_date)

    if start_date is None:
        start_ts = end_ts - pd.DateOffset(years=train_years + 1)
        start_date = start_ts.strftime('%Y-%m-%d')

    df = fetch_polygon_bars(ticker, start_date, end_date)
    if df.empty:
        raise RuntimeError(f'No data for {ticker} between {start_date} and {end_date}')

    df = df.sort_index()

    labels = get_triple_barrier_labels(
        df,
        profit_take=profit_take,
        stop_loss=stop_loss,
        horizon=horizon,
        direction=direction,
    )

    valid_dates = labels.dropna().index
    if len(valid_dates) == 0:
        raise RuntimeError('No valid labels to evaluate')

    eval_dates = valid_dates[-eval_days:]
    rows = []

    for dt in eval_dates:
        train_start = dt - pd.DateOffset(years=train_years)
        if train_start < df.index[0]:
            train_start = df.index[0]

        probs, _ = train_and_predict(
            ticker,
            start_date=str(train_start.date()),
            end_date=str(dt.date()),
            df=df,
            direction=direction,
        )

        if not probs or not isinstance(probs, dict):
            continue

        p = probs.get('FINAL', None)
        if not isinstance(p, (float, int)):
            continue

        y = float(labels.loc[dt])
        rows.append(
            {
                'date': dt,
                'p': float(p),
                'y': y,
                'n_train': int(probs.get('N_TRAIN', 0)),
                'base': float(probs.get('BASE', 0.5)),
            }
        )

    df_eval = pd.DataFrame(rows)
    if df_eval.empty:
        raise RuntimeError('No evaluation rows produced (ML returned empty too often)')

    brier = float(((df_eval['p'] - df_eval['y']) ** 2).mean())
    hit_rate = float(df_eval['y'].mean())
    p_mean = float(df_eval['p'].mean())

    print(f'TICKER={ticker} DIRECTION={direction} EVAL_ROWS={len(df_eval)}')
    print(f'AVG_P={p_mean:.3f} HIT_RATE={hit_rate:.3f} BRIER={brier:.4f}')
    print('Calibration (bin -> mean(p) vs empirical hit rate):')
    print(_calibration_table(df_eval, bin_size=0.1).to_string(index=False))

    out_csv = os.getenv('OUT_CSV', '').strip()
    if out_csv:
        df_eval.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
