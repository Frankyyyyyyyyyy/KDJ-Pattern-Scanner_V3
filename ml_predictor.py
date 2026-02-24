import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import ta
import logging

logger = logging.getLogger(__name__)


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


def _tscv_eval_lr(X, y, n_splits=5):
    X = X.copy()
    y = y.copy()

    if len(X) < (n_splits + 1) * 20:
        return None

    if y.nunique() < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_true_all = []
    y_prob_all = []

    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
        model.fit(X_train_scaled, y_train)
        prob = model.predict_proba(X_test_scaled)[:, 1]

        y_true_all.append(y_test.to_numpy())
        y_prob_all.append(prob)

    if not y_true_all:
        return None

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None
    try:
        ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        ll = None

    brier = float(np.mean((y_prob - y_true) ** 2))
    base = float(np.mean(y_true))

    return {
        'CV_N': int(len(y_true)),
        'CV_BASE': base,
        'CV_BRIER': brier,
        'CV_LOGLOSS': ll,
        'CV_AUC': auc,
    }

# ==========================================
# 1. Feature Engineering (Shared)
# ==========================================

def calculate_advanced_features(df):
    """
    Calculate comprehensive technical indicators using 'ta' library.
    """
    df = df.copy()
    
    # Ensure no zero volume
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill()

    # Log Returns
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility
    atr = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR_Norm'] = atr.average_true_range() / df['Close']
    
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Width'] = bb.bollinger_wband()
    df['BB_P'] = bb.bollinger_pband() 
    
    # Momentum
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Diff'] = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_K'] = stoch.stoch()
    
    # Trend
    adx = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
    df['ADX'] = adx.adx()
    
    # Volume
    df['OBV_ROC'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume().pct_change(5)
    df['MFI'] = ta.volume.MFIIndicator(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume']).money_flow_index()

    # Reversal / Mean Reversion Features
    # 1. Bias (乖离率): (Price - MA) / MA
    df['Bias_20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    
    # 2. RSI Extremes (超买超卖): Distance from 50
    df['RSI_Dist'] = df['RSI'] - 50
    
    # 3. KDJ J-Value Extremes
    kdj = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['KDJ_J'] = 3 * kdj.stoch() - 2 * kdj.stoch_signal()
    df['J_Dist'] = df['KDJ_J'] - 50

    return df

def get_triple_barrier_labels(df, profit_take=0.03, stop_loss=0.015, horizon=10, direction='Long'):
    """
    Triple Barrier Method for Labeling.
    """
    labels = []
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    for i in range(len(df)):
        end_idx = min(i + horizon, len(df))
        future_highs = highs[i+1:end_idx+1]
        future_lows = lows[i+1:end_idx+1]
        
        if len(future_highs) == 0:
            labels.append(np.nan)
            continue
            
        current_close = closes[i]
        
        if direction == 'Long':
            upper = current_close * (1 + profit_take)
            lower = current_close * (1 - stop_loss)
            
            hit_upper = np.where(future_highs >= upper)[0]
            hit_lower = np.where(future_lows <= lower)[0]
            
            first_upper = hit_upper[0] if len(hit_upper) > 0 else horizon + 1
            first_lower = hit_lower[0] if len(hit_lower) > 0 else horizon + 1
            
            if first_upper < first_lower:
                labels.append(1)
            else:
                labels.append(0)
                
        elif direction == 'Short':
            lower_target = current_close * (1 - profit_take)
            upper_stop = current_close * (1 + stop_loss)
            
            hit_lower_target = np.where(future_lows <= lower_target)[0]
            hit_upper_stop = np.where(future_highs >= upper_stop)[0]
            
            first_lower_target = hit_lower_target[0] if len(hit_lower_target) > 0 else horizon + 1
            first_upper_stop = hit_upper_stop[0] if len(hit_upper_stop) > 0 else horizon + 1
            
            if first_lower_target < first_upper_stop:
                labels.append(1)
            else:
                labels.append(0)
            
    return pd.Series(labels, index=df.index)

# ==========================================
# 2. Ensemble Prediction System
# ==========================================

def run_ensemble_prediction(ticker, start_date, end_date, df_input, direction='Long'):
    """
    Runs 3 different models and returns their probabilities.
    Models:
    1. XGBoost (Gradient Boosting)
    2. LightGBM (Gradient Boosting - Faster/Different splits)
    3. Logistic Regression (Linear Baseline)
    
    Returns: Dict of probabilities
    """
    try:
        # --- Data Prep ---
        if df_input is None:
            return {}

        df_input = df_input.copy().sort_index()
        if start_date:
            df_input = df_input[df_input.index >= start_date]
        if end_date:
            df_input = df_input[df_input.index <= end_date]

        df_features = calculate_advanced_features(df_input)
        
        # Target
        df_features['Target'] = get_triple_barrier_labels(
            df_features, profit_take=0.03, stop_loss=0.015, horizon=10, direction=direction
        )
        
        feature_cols = [
            'Log_Ret', 'ATR_Norm', 'BB_Width', 'BB_P', 'RSI', 'MACD', 'MACD_Diff', 
            'Stoch_K', 'ADX', 'OBV_ROC', 'MFI',
            'Bias_20', 'RSI_Dist', 'J_Dist'
        ]
        valid_features = [c for c in feature_cols if c in df_features.columns]
        
        train_df = df_features.dropna(subset=['Target'] + valid_features)
        
        min_train = _env_int('ML_MIN_TRAIN_SAMPLES', 100)
        if len(train_df) < min_train:
            return {}

        X_train = train_df[valid_features]
        y_train = train_df['Target']
        base_rate = float(y_train.mean())
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        if y_train.nunique() < 2:
            base_rate = float(y_train.mean())
            return {
                'XGB': base_rate,
                'LGBM': base_rate,
                'LR': base_rate,
                'HMM_Bull': 0.0,
            }
        
        predict_row = df_features.dropna(subset=valid_features).tail(1)
        if predict_row.empty:
            return {}

        X_predict = predict_row[valid_features]

        results = {}
        results['BASE'] = base_rate
        results['N_TRAIN'] = int(len(train_df))

        # --- Model 1: XGBoost ---
        xgb_params = {
            'n_estimators': 150,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': 1,
            'device': 'cpu',
            'scale_pos_weight': scale_pos_weight,
        }

        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        results['XGB'] = xgb_model.predict_proba(X_predict)[0][1]

        # --- Model 2: LightGBM ---
        lgb_model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, 
            n_jobs=1, verbosity=-1,
            scale_pos_weight=scale_pos_weight,
        )
        lgb_model.fit(X_train, y_train)
        results['LGBM'] = lgb_model.predict_proba(X_predict)[0][1]

        # --- Model 3: Logistic Regression (Robust Baseline) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_predict_scaled = scaler.transform(X_predict)
        
        lr_model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        results['LR'] = lr_model.predict_proba(X_predict_scaled)[0][1]

        enable_cv = os.getenv('ML_ENABLE_CV', '').strip().lower() in {'1', 'true', 'yes'}
        if enable_cv:
            n_splits = _env_int('ML_CV_SPLITS', 5)
            cv_max_rows = _env_int('ML_CV_MAX_ROWS', 1200)
            X_cv = X_train.tail(cv_max_rows)
            y_cv = y_train.tail(cv_max_rows)
            cv = _tscv_eval_lr(X_cv, y_cv, n_splits=n_splits)
            if cv:
                results.update(cv)

        return results

    except Exception as e:
        logger.error(f"Error in ensemble prediction for {ticker}: {e}")
        return {}

def train_and_predict(ticker, start_date=None, end_date=None, df=None, direction='Long'):
    """
    Wrapper for robust training using Ensemble.
    Returns: (probs_dict, importances)
    """
    probs = run_ensemble_prediction(ticker, start_date, end_date, df, direction=direction)
    if not probs:
        return None, None
        
    xgb_p = float(probs.get('XGB', 0))
    lgbm_p = float(probs.get('LGBM', 0))
    lr_p = float(probs.get('LR', 0))
    base_rate = float(probs.get('BASE', 0.5))
    n_train = int(probs.get('N_TRAIN', 0))

    tree_avg = (xgb_p + lgbm_p) / 2
    divergence = abs(tree_avg - lr_p)

    w_tree = 0.8
    w_lr = 0.2
    if divergence > 0.3:
        w_tree = 0.4
        w_lr = 0.6

    final_p = (tree_avg * w_tree) + (lr_p * w_lr)

    shrink_k = _env_int('ML_SHRINK_K', 800)
    if n_train > 0 and shrink_k > 0:
        strength = n_train / (n_train + shrink_k)
        final_p = base_rate + (final_p - base_rate) * strength

    final_p = max(0.0, min(1.0, float(final_p)))

    probs['W_TREE'] = float(w_tree)
    probs['W_LR'] = float(w_lr)
    probs['FINAL'] = final_p

    return probs, None
