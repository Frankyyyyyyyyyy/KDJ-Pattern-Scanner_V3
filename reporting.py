import logging
import smtplib
import os
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from config import STRATEGY_CONFIG, PATTERN_WEIGHTS, SENDER_EMAIL, RECEIVER_EMAIL, GMAIL_APP_PASSWORD
from utils import get_now_et

logger = logging.getLogger(__name__)

def add_result(results, date, ticker, t_map, t_type_map, row, daily_j, wk_j,
               direction, patterns, df, loc_idx):
    """构建信号结果，包含强度评分、止损建议、仓位建议、回测收益。"""
    cfg = STRATEGY_CONFIG

    # --- 信号强度评分 ---
    score = 0
    for p in patterns:
        score += PATTERN_WEIGHTS.get(p, 1)

    # J 值极端程度加分
    if direction == "Short":
        if daily_j > 90:
            score += 2
        elif daily_j > 80:
            score += 1
    else:
        if daily_j < 10:
            score += 2
        elif daily_j < 20:
            score += 1

    if score >= 10:
        s_desc = f"极强 ({score})"
    elif score >= 6:
        s_desc = f"强 ({score})"
    elif score >= 3:
        s_desc = f"中 ({score})"
    else:
        s_desc = f"弱 ({score})"

    p_str = ", ".join(list(set(patterns)))

    # --- 止损建议 ---
    lookback = cfg['lookback_days']
    sl_start = max(0, loc_idx - (lookback - 1))
    recent = df.iloc[sl_start:loc_idx + 1]

    if direction == "Short":
        stop_loss = recent['High'].max() * (1 + cfg['stop_loss_buffer'])
    else:
        stop_loss = recent['Low'].min() * (1 - cfg['stop_loss_buffer'])

    # --- ATR 仓位建议 ---
    current_atr = row.get('ATR', None)
    stop_distance = abs(row['Close'] - stop_loss)
    position_size = None
    suggested_shares = None

    if current_atr is not None and not pd.isna(current_atr) and stop_distance > 0:
        capital = cfg['default_capital']
        risk_amount = capital * cfg['risk_per_trade']
        suggested_shares = int(risk_amount / stop_distance)
        position_size = round(suggested_shares * row['Close'], 0)

    # --- 回测收益（扣除交易成本） ---
    perf_days = cfg['performance_days']
    perf_max = None
    perf_ret = None
    is_eff = None

    # ATR 归一化 Effective 阈值
    atr_pct = current_atr / row['Close'] if (current_atr and row['Close'] > 0) else 0.01
    effective_threshold = atr_pct * cfg['atr_effective_multiplier']

    cost = cfg['round_trip_cost']

    if loc_idx + perf_days < len(df):
        fut = df.iloc[loc_idx + 1: loc_idx + perf_days + 1]
        c_nd = fut['Close'].iloc[-1]

        if direction == "Short":
            min_low = fut['Low'].min()
            max_p = (row['Close'] - min_low) / row['Close'] - cost
            act_ret = (row['Close'] - c_nd) / row['Close'] - cost
        else:
            max_high = fut['High'].max()
            max_p = (max_high - row['Close']) / row['Close'] - cost
            act_ret = (c_nd - row['Close']) / row['Close'] - cost

        perf_max = f"{max_p * 100:.2f}%"
        perf_ret = f"{act_ret * 100:.2f}%"
        is_eff = "Yes" if max_p > effective_threshold else "No"
    else:
        is_eff = "N/A"

    results.append({
        'Date (日期)': date.date(),
        'Ticker (股票代码)': ticker,
        'Name (名称)': t_map.get(ticker, ticker),
        'Type (类型)': t_type_map.get(ticker, 'Unknown'),
        'Volume (成交量)': int(row['Volume']),
        'Direction (方向)': direction,
        'Signal Strength': s_desc,
        'Pattern (形态)': p_str,
        'Daily_J': round(daily_j, 2),
        'Weekly_J': round(wk_j, 2),
        'Price': round(row['Close'], 2),
        'ATR': round(current_atr, 2) if current_atr and not pd.isna(current_atr) else None,
        'Stop_Loss (止损)': round(stop_loss, 2),
        'Suggested_Shares (建议股数)': suggested_shares,
        'Position_Size (建议仓位$)': int(position_size) if position_size else None,
        'Max_Profit/Gain': perf_max,
        '5D_Return': perf_ret,
        'Effective': is_eff,
        'ML_Detail': None, # Placeholder
    })


def _build_signal_row(row):
    """为单条信号生成一个表格行 HTML (v3.0 - 极简风格)。"""
    direction = row['Direction (方向)']
    is_short = direction == 'Short'
    
    # 颜色定义
    dir_color = '#e74c3c' if is_short else '#27ae60'
    arrow = '▼' if is_short else '▲'
    
    # 信号强度
    strength = str(row.get('Signal Strength', ''))
    if '极强' in strength:
        s_badge = f'<span style="color:#c0392b;font-weight:bold;">{strength}</span>'
    elif '强' in strength:
        s_badge = f'<span style="color:#d35400;font-weight:bold;">{strength}</span>'
    elif '中' in strength:
        s_badge = f'<span style="color:#2980b9;">{strength}</span>'
    else:
        s_badge = f'<span style="color:#7f8c8d;">{strength}</span>'

    # J值高亮
    daily_j = row.get('Daily_J', 0)
    wk_j = row.get('Weekly_J', 0)
    
    j_style = ""
    if is_short and (daily_j > 90 or wk_j > 90):
        j_style = "color:#c0392b;font-weight:bold;"
    elif not is_short and (daily_j < 10 or wk_j < 10):
        j_style = "color:#27ae60;font-weight:bold;"
        
    # 形态简化
    patterns = str(row.get('Pattern (形态)', '')).split(',')
    short_patterns = []
    for p in patterns:
        p = p.strip()
        if '(' in p:
            p = p.split('(')[0].strip()
        short_patterns.append(p)
    pattern_str = ', '.join(short_patterns)

    # 格式化数字
    vol = pd.to_numeric(row.get('Volume (成交量)', 0), errors='coerce')
    vol = 0 if pd.isna(vol) else float(vol)
    vol_str = f"{int(vol)/1000000:.1f}M" if vol > 1000000 else f"{int(vol)/1000:.0f}K"
    stop_loss = row.get('Stop_Loss (止损)', 0)
    price = row.get('Price', 0)
    date_str = str(row.get('Date (日期)', ''))
    
    ml_prob = str(row.get('ML_Prob', 'N/A'))
    ml_detail = str(row.get('ML_Detail', '') or '')
    if not ml_prob or ml_prob == 'nan':
        ml_prob = 'N/A'

    ai_cell = '<span style="color:#bdc3c7;">—</span>'
    if ml_prob != 'N/A':
        try:
            p = float(ml_prob.strip('%'))
        except Exception:
            p = None

        if p is None:
            ai_cell = f'<span style="font-size:11px;color:#7f8c8d;">{ml_prob}</span>'
        else:
            ai_color = "#27ae60" if p >= 55 else ("#d35400" if p >= 50 else "#7f8c8d")
            ai_cell = f'<span style="font-size:11px;color:{ai_color};font-weight:600;">{ml_prob}</span>'

        if ml_detail and ml_detail != 'N/A' and ml_detail != 'nan':
            ai_cell += f'<br><span style="font-size:9px;color:#95a5a6;">{ml_detail}</span>'

    # 使用简洁的 td 样式，避免过多的内联样式干扰 CSS 类
    return f'''
    <tr>
        <td style="color:#7f8c8d;">{date_str}</td>
        <td class="ticker">{row.get('Ticker (股票代码)', '')}</td>
        <td style="color:{dir_color};font-weight:bold;">{arrow} {direction}</td>
        <td>{ai_cell}</td>
        <td style="font-family:monospace;">${price}</td>
        <td>{s_badge}</td>
        <td style="max-width:180px;font-size:12px;">{pattern_str}</td>
        <td style="{j_style}">D:{daily_j}<br>W:{wk_j}</td>
        <td style="font-size:11px;color:#7f8c8d;">SL: ${stop_loss}<br>Vol: {vol_str}</td>
    </tr>
    '''


def build_email_html(df_results, extra_html=None):
    """
    生成完整的邮件 HTML 内容 (v3.2 - 优化排序逻辑)。
    排序优先级: 
    1. 日期 (Date) 降序 - 最近的信号排最前
    2. KDJ 极值 (Highlight) 降序 - 红色/绿色加粗的排前面 (J>90 or J<10)
    3. 活跃度 (Volume) 降序 - 成交量大的排前面
    """
    now = pd.Timestamp(get_now_et().replace(tzinfo=None))
    current_date = now.strftime("%Y-%m-%d")
    universe = os.getenv('EMAIL_UNIVERSE', 'ALL')

    lookback_days_raw = os.getenv('EMAIL_LOOKBACK_DAYS', '14')
    lookback_days = 14
    if str(lookback_days_raw).strip().isdigit():
        lookback_days = int(str(lookback_days_raw).strip())

    two_weeks_ago = (now.normalize() - pd.Timedelta(days=lookback_days))
    # Ensure two_weeks_ago is a Timestamp (it already is, but make sure df date is comparable)
    # The dataframe 'Date (日期)' column might be Timestamp or datetime.date
    
    # We will normalize both to pandas Timestamp for comparison
    df_results['Date (日期)'] = pd.to_datetime(df_results['Date (日期)']).dt.normalize()
    df_recent = df_results[df_results['Date (日期)'] >= two_weeks_ago].copy()
    
    # --- 构建排序辅助列 ---
    # 1. KDJ 极值标识 (Short > 90, Long < 10)
    def is_kdj_highlight(row):
        direction = row['Direction (方向)']
        d_j = row.get('Daily_J', 50)
        w_j = row.get('Weekly_J', 50)
        
        if direction == 'Short':
            return 1 if (d_j > 90 or w_j > 90) else 0
        else:
            return 1 if (d_j < 10 or w_j < 10) else 0

    df_recent['Sort_Highlight'] = df_recent.apply(is_kdj_highlight, axis=1)
    
    # 2. AI 概率排序 (ML_Prob 越高越好)
    def parse_ml_prob(val):
        if not val or val == 'N/A': return 0
        try:
            return float(val.strip('%')) 
        except:
            return 0
            
    df_recent['Sort_AI'] = df_recent['ML_Prob'].apply(parse_ml_prob)

    # 2. 确保 Volume 是数值类型
    df_recent['Sort_Vol'] = pd.to_numeric(df_recent['Volume (成交量)'], errors='coerce').fillna(0)
    
    # 3. 执行排序: 日期(降) -> AI概率(降) -> 极值(降) -> 活跃度(降)
    df_recent = df_recent.sort_values(
        by=['Date (日期)', 'Sort_AI', 'Sort_Highlight', 'Sort_Vol'], 
        ascending=[False, False, False, False]
    )
    # --------------------

    short_signals = df_recent[df_recent['Direction (方向)'] == 'Short']
    long_signals = df_recent[df_recent['Direction (方向)'] == 'Long']
    
    short_count = len(short_signals)
    long_count = len(long_signals)
    total_count = short_count + long_count

    # 表格头
    table_header = '''
    <thead>
        <tr>
            <th width="80">Date</th>
            <th width="60">Ticker</th>
            <th width="60">Dir</th>
            <th width="110">AI</th>
            <th width="70">Price</th>
            <th width="60">Str</th>
            <th>Patterns</th>
            <th width="70">KDJ</th>
            <th width="90">Info</th>
        </tr>
    </thead>
    '''

    # 构建 Short 表格行
    short_rows = ''
    if short_count > 0:
        short_rows = ''.join(_build_signal_row(row) for _, row in short_signals.iterrows())

    # 构建 Long 表格行
    long_rows = ''
    if long_count > 0:
        long_rows = ''.join(_build_signal_row(row) for _, row in long_signals.iterrows())

    # 历史数据提示
    total_all = len(df_results)
    older_count = total_all - total_count

    extra_html = extra_html or ''

    html_content = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f5f5f5;
    }}
    .container {{
        max-width: 800px;
        margin: 20px auto;
        background-color: white;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    h2 {{
        color: #2c3e50;
        margin-top: 0;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 13px;
    }}
    th {{
        background-color: #3498db;
        color: white;
        padding: 10px 8px;
        text-align: left;
        font-weight: 600;
        font-size: 12px;
    }}
    td {{
        padding: 10px 8px;
        border-bottom: 1px solid #eee;
        color: #333;
    }}
    /* 隔行变色 */
    tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    /* 鼠标悬停高亮 - 关键 */
    tr:hover {{
        background-color: #e8f4f8 !important;
    }}
    .ticker {{
        font-weight: bold;
        color: #2980b9;
    }}
</style>
</head>
<body>
  <div class="container">
    <h2>Trading Signals Report</h2>
    <p style="color:#7f8c8d;font-size:14px;margin-bottom:20px;">
        {current_date} | Recent: <strong>{total_count}</strong> signals | Lookback: {lookback_days} days | Universe: {universe}
    </p>

    <!-- 摘要统计卡片 (纯展示，无跳转链接) -->
    <div style="display:flex;gap:15px;margin-bottom:25px;">
        <div style="flex:1;text-align:center;padding:12px;border-radius:6px;background:#fdf2f2;border:1px solid #f8d7da;">
            <div style="font-size:20px;font-weight:bold;color:#c0392b;">▼ {short_count}</div>
            <div style="font-size:12px;color:#7f8c8d;">Short (做空)</div>
        </div>
        <div style="flex:1;text-align:center;padding:12px;border-radius:6px;background:#f0f9eb;border:1px solid #d4edda;">
            <div style="font-size:20px;font-weight:bold;color:#27ae60;">▲ {long_count}</div>
            <div style="font-size:12px;color:#7f8c8d;">Long (做多)</div>
        </div>
    </div>

    <!-- Short Section -->
    <h3 id="short-section" style="color:#c0392b;margin-top:30px;padding-left:10px;border-left:4px solid #c0392b;">
        ▼ Short Signals ({short_count})
    </h3>
    <table>
        {table_header}
        <tbody>
            {short_rows}
        </tbody>
    </table>

    <!-- Long Section -->
    <h3 style="color:#27ae60;margin-top:40px;padding-left:10px;border-left:4px solid #27ae60;">
        ▲ Long Signals ({long_count})
    </h3>
    <table>
        {table_header}
        <tbody>
            {long_rows}
        </tbody>
    </table>

    {extra_html}

    <!-- 底部提示 -->
    <p style="color:#bdc3c7;font-size:12px;margin-top:40px;text-align:center;border-top:1px solid #eee;padding-top:15px;">
        显示最近 2 周信号。{"另有 " + str(older_count) + " 条历史信号见 CSV 附件" if older_count > 0 else "完整数据见 CSV 附件"}
    </p>
  </div>
</body>
</html>'''
    return html_content


def send_email_report(df_results, output_file, extra_html=None):
    """Sends an email with the results and CSV attachment."""
    if os.getenv('SKIP_EMAIL', '').strip().lower() in {'1', 'true', 'yes'}:
        logger.info("SKIP_EMAIL 已启用，跳过发送邮件")
        return

    password = GMAIL_APP_PASSWORD
    if not password:
        logger.warning("GMAIL_APP_PASSWORD not found. Email will NOT be sent.")
        return

    logger.info("Preparing email report...")

    html_content = build_email_html(df_results, extra_html=extra_html)

    now = pd.Timestamp(get_now_et().replace(tzinfo=None))
    current_date = now.strftime("%Y-%m-%d")
    short_count = len(df_results[df_results['Direction (方向)'] == 'Short'])
    long_count = len(df_results[df_results['Direction (方向)'] == 'Long'])

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = f"Trading Signals {current_date} | {short_count}S {long_count}L"

    msg.attach(MIMEText(html_content, 'html'))

    if os.path.exists(output_file):
        with open(output_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {output_file}",
        )
        msg.attach(part)

    try:
        smtp_timeout_raw = os.getenv('SMTP_TIMEOUT', '90').strip()
        smtp_timeout = 90
        if smtp_timeout_raw.isdigit():
            smtp_timeout = int(smtp_timeout_raw)

        text = msg.as_string()

        primary_err = None
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587, timeout=smtp_timeout)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SENDER_EMAIL, password)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
            server.quit()
            logger.info(f"Email sent successfully to {RECEIVER_EMAIL}")
            return
        except Exception as e:
            primary_err = e

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=smtp_timeout)
        server.login(SENDER_EMAIL, password)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, text)
        server.quit()
        logger.info(f"Email sent successfully to {RECEIVER_EMAIL} (SSL)")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
