# ESmain KDJ Monitor

监控标普 500 E-mini 期货（ESmain，连续主力合约）KDJ 超卖买入机会的轻量级算法工具。
纯命令行 + 邮件提醒，可挂 cron 或 GitHub Actions。

## 策略

每个交易日扫描一次，**只要日线 J 满足阈值即触发邮件**，邮件标题和内容里
按"高时间框架确认数"分三档强度：

| 强度 | 触发条件 | 含义 |
|------|---------|------|
| **HIGH 强**   | 日J<0 且 周J<10 且 月J<55 | 三个时间框架同时超卖（最优买点） |
| **MEDIUM 中** | 日J<0 且（周J<10 或 月J<55） | 两个时间框架确认 |
| **LOW 弱**    | 仅 日J<0                   | 单时间框架，需结合其他工具确认 |

阈值默认 日J<0 / 周J<10 / 月J<55，可在 `.env` 或命令行覆盖。

KDJ 参数：N = 9，K/D 平滑因子 1/3（标准 9-3-3）。

防 lookahead-bias：评估日 T 只引用截止日**严格早于** T 的已完成 K 线
（`pd.merge_asof(..., allow_exact_matches=False)`）。

回测按 **Episode_Start 当日可知的 Start_Strength** 分组，不使用区间事后 Peak_Strength。
入场为该日收盘价。数据源日界：yfinance 的 `end` 按右开区间处理，本项目把它转换为包容结束日；
拉取后会按美东时间保守排除可能未完成的最后一根日线（工作日 17:00 ET 前不信任“今天”，
周末不信任周日夜盘 bar）。这不是完整的 CME 日历，也不套用 NYSE 股票休市表。
KDJ 暖机：K/D 初值 50，RSV 未定义时保持 50；与主项目约定一致，不是唯一市场标准。

## 项目结构

```
ESmain-KDJ-Monitor/
├── README.md
├── .gitignore                # 排除密钥 / output / logs / data
├── .env.example              # 配置模板
├── requirements.txt
├── config.py                 # 集中读取环境变量
├── indicators.py             # KDJ + 周/月线重采样
├── data_fetcher.py           # polygon / yahoo / offline 三种数据源
├── signal_strength.py        # 信号强度三档分级逻辑
├── notifier.py               # SMTP 邮件（HTML 强度色条）
├── scanner.py                # 历史扫描入口（CLI）
├── monitor.py                # 每日 cron 入口（含 --min-strength 过滤）
├── run_daily.sh              # cron 包装脚本
├── tests/test_indicators.py  # KDJ 公式 + 防 lookahead + 强度分级 单元测试
├── data/                     # 离线 CSV 数据（不上传）
├── output/                   # 扫描结果（不上传）
├── logs/alerts.csv           # 历次告警追加日志（不上传）
└── .github/workflows/daily.yml  # GitHub Actions 模板
```

## 快速上手

### 1) 安装

```bash
git clone <your-repo>
cd ESmain-KDJ-Monitor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env       # 填入 Polygon key / Gmail 应用专用密码 / 收件人
```

> Gmail 应用专用密码生成入口：<https://myaccount.google.com/apppasswords>

### 2) 一次性扫描过去 10 年

```bash
python scanner.py
# 自定义阈值 / 时间窗
python scanner.py --start 2014-01-01 --monthly-j 50 --weekly-j 5 --daily-j -10
# 只看强信号
python scanner.py --min-strength HIGH
```

输出在 `output/`：

| 文件 | 内容 |
|------|------|
| `esmain_kdj_signals.csv`     | 信号事件（连续天合并 + 区段内峰值强度） |
| `esmain_kdj_panel_daily.csv` | 每日诊断面板（含三条 J、三个 OK 标志、Strength、Signal） |
| `esmain_kdj_summary.txt`     | 文本汇总（按强度分档统计） |

### 3) 每日运行 + 邮件提醒

```bash
python monitor.py                          # 默认：日线满足就发（含 LOW）
python monitor.py --min-strength MEDIUM    # 弱信号不发，只发 MEDIUM/HIGH
python monitor.py --min-strength HIGH      # 只发三确认强信号
python monitor.py --dry-run                # 只判定不发邮件
```

邮件主题示例：
```
[ESmain KDJ] HIGH 信号 — 2025-04-07
[ESmain KDJ] MEDIUM 信号 — 2024-04-16
[ESmain KDJ] LOW 信号 — 2025-08-22
```

邮件正文表格里逐条列出三个时间框架 J 值、各自阈值、是否满足（✓/✗），
左侧带强度色条（HIGH 绿 / MEDIUM 橙 / LOW 灰），方便邮件列表里一眼分辨。

每发一封邮件追加一行到 `logs/alerts.csv`，方便日后回溯。

### 4) 挂 cron（本地）

```cron
# crontab -e   每天美东收盘后 30 分钟（北京时间次日 04:30）
30 4 * * 2-6 /path/to/ESmain-KDJ-Monitor/run_daily.sh >> logs/cron.log 2>&1
```

### 5) GitHub Actions（云端）

仓库 Settings → Secrets 添加：
`POLYGON_API_KEY` / `SENDER_EMAIL` / `GMAIL_APP_PASSWORD` / `RECEIVER_EMAIL`，
`.github/workflows/daily.yml` 已经写好，每个交易日自动跑。

## 数据源

| 来源 | 何时使用 | 备注 |
|------|---------|------|
| `polygon` | 首选 | 真正的 ES 期货数据；需付费 Futures 订阅 |
| `yahoo`   | 备选 | 免费 `ES=F`；中国大陆需 VPN |
| `offline` | 演示 | 用 SPX 现货指数代理；趋势 99% 一致，绝对值有微小基差 |

## 单元测试

```bash
python -m unittest tests/test_indicators.py -v
```

11 个用例覆盖：KDJ 公式、平价收敛、周/月聚合、严格无 lookahead、
日线触发门槛、强度三档分类、强度排序。

## License

MIT
