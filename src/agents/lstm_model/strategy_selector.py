"""strategy_selector.py
Offline backtesting and strategy selection for a single ticker.

Produces a small JSON file `models/{ticker}_strategy.json` with the chosen
strategy and tuned parameters. Strategies currently implemented:
 - moving_average_crossover
 - momentum_threshold
 - bollinger_breakout
 - buy_and_hold

Selection metric: Sharpe ratio (annualized). If Sharpe ties, prefers higher CAGR.
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from pathlib import Path
from pathlib import Path


@dataclass
class StrategyResult:
    name: str
    params: Dict[str, Any]
    cagr: float
    sharpe: float
    max_drawdown: float
    trades: int


def download_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df = df[['Date', 'Close']].copy()
    df = df.dropna()
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0)


def cagr(returns: pd.Series) -> float:
    compounded = (1 + returns).prod()
    years = len(returns) / 252.0
    if years <= 0:
        return 0.0
    return compounded ** (1 / years) - 1


def sharpe_ratio(returns: pd.Series, rf=0.0) -> float:
    excess = returns - rf / 252.0
    if returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / (excess.std() + 1e-12)


def max_drawdown(cum_returns: pd.Series) -> float:
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / (peak + 1e-12)
    return float(dd.min())


def backtest_ma_crossover(df: pd.DataFrame, short=5, long=20) -> StrategyResult:
    prices = df['Close']
    sma_short = prices.rolling(window=short).mean()
    sma_long = prices.rolling(window=long).mean()
    signal = (sma_short > sma_long).astype(int)
    # signal shifts: buy on cross from 0->1, hold when 1
    positions = signal.shift(1).fillna(0)
    returns = compute_returns(prices) * positions
    cum = (1 + returns).cumprod() - 1
    res = StrategyResult(
        name='moving_average_crossover',
        params={'short': short, 'long': long},
        cagr=cagr(returns),
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown(cum),
        trades=int(positions.diff().abs().sum())
    )
    return res


def backtest_momentum(df: pd.DataFrame, window=5, thr=0.0) -> StrategyResult:
    prices = df['Close']
    mom = prices.pct_change(periods=window)
    signal = (mom > thr).astype(int)
    positions = signal.shift(1).fillna(0)
    returns = compute_returns(prices) * positions
    cum = (1 + returns).cumprod() - 1
    res = StrategyResult(
        name='momentum_threshold',
        params={'window': window, 'threshold': thr},
        cagr=cagr(returns),
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown(cum),
        trades=int(positions.diff().abs().sum())
    )
    return res


def backtest_bollinger(df: pd.DataFrame, window=20, k=2.0) -> StrategyResult:
    prices = df['Close']
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + k * std
    lower = ma - k * std
    # breakout: buy when price crosses above upper, sell when below lower
    signal = pd.Series(0, index=prices.index)
    signal[prices > upper] = 1
    signal[prices < lower] = -1
    positions = (signal.replace(-1, 0)).shift(1).fillna(0)
    returns = compute_returns(prices) * positions
    cum = (1 + returns).cumprod() - 1
    res = StrategyResult(
        name='bollinger_breakout',
        params={'window': window, 'k': k},
        cagr=cagr(returns),
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown(cum),
        trades=int(positions.diff().abs().sum())
    )
    return res


def backtest_buy_and_hold(df: pd.DataFrame) -> StrategyResult:
    prices = df['Close']
    returns = compute_returns(prices)
    cum = (1 + returns).cumprod() - 1
    res = StrategyResult(
        name='buy_and_hold',
        params={},
        cagr=cagr(returns),
        sharpe=sharpe_ratio(returns),
        max_drawdown=max_drawdown(cum),
        trades=1
    )
    return res


def backtest_lstm(df: pd.DataFrame, ticker: str, model_dir: str = 'models') -> StrategyResult | None:
    """Backtest using saved LSTM model predictions if model/scaler exist.

    Approach:
    - Load `{model_dir}/{ticker}_lstm.pt` and scaler `{model_dir}/{ticker}_scaler.pkl`.
    - Recreate sequences using the same features ordering expected by scaler.
    - Run the model in a sliding-window fashion to predict next-step Close.
    - Generate simple threshold trading signal: buy when predicted > current*(1+thr), sell when below.
    - Compute returns from those signals.
    """
    model_path = Path(model_dir) / f"{ticker}_lstm.pt"
    scaler_path = Path(model_dir) / f"{ticker}_scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None

    # Load PyTorch lazily so script can run without torch installed for rule-based backtests
    try:
        import torch
        import torch.nn as nn
    except Exception:
        # PyTorch not installed; skip LSTM backtest
        return None

    # Lightweight LSTM matching training architecture for loading checkpoint
    class StockLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(StockLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :]
            prediction = self.fc(last_out)
            return prediction

    # Load checkpoint
    try:
        checkpoint = torch.load(str(model_path), map_location='cpu')
    except Exception:
        return None

    input_size = checkpoint.get('input_size')
    lookback = checkpoint.get('lookback')
    feature_columns = checkpoint.get('feature_columns', None)

    # Use default feature order if not present
    if feature_columns is None:
        feature_columns = ['Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Middle', 'Momentum', 'Volume_Ratio']

    # Ensure df has all feature columns by computing approximations if missing
    # For simplicity here, we only require 'Close' which always exists; others will be filled with zeros if missing
    data = df.copy()
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0.0

    # Load scaler
    try:
        scaler = joblib.load(str(scaler_path))
    except Exception:
        return None

    # Prepare feature matrix
    features = data[feature_columns].values

    # Scale entire dataset
    try:
        scaled = scaler.transform(features)
    except Exception:
        # scaler may expect 2D; attempt reshape
        try:
            scaled = scaler.transform(features.reshape(-1, len(feature_columns)))
        except Exception:
            return None

    # Create sequences
    sequences = []
    indices = []
    for i in range(lookback, len(scaled)):
        seq = scaled[i-lookback:i]
        sequences.append(seq)
        indices.append(i)

    if not sequences:
        return None

    X = torch.FloatTensor(np.array(sequences))

    # Recreate the model and load state
    model = StockLSTM(input_size=input_size or X.shape[2], hidden_size=checkpoint.get('hidden_size',64), num_layers=checkpoint.get('num_layers',2), dropout=checkpoint.get('dropout',0.2))
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        # Incompatible checkpoint
        return None
    model.eval()

    with torch.no_grad():
        preds = model(X).numpy().reshape(-1)

    # Inverse transform predicted feature vector assuming scaler first column is Close
    predicted_prices = []
    for p in preds:
        arr = np.zeros((1, scaler.n_features_in_))
        arr[0, 0] = p
        try:
            inv = scaler.inverse_transform(arr)[0, 0]
        except Exception:
            inv = float(p)
        predicted_prices.append(inv)

    # Map predictions to indices (predicted for index i corresponds to actual index indices[i])
    # Create trading signals: buy when predicted > current*(1+thr), sell when predicted < current*(1-thr)
    thr = 0.005
    prices = df['Close'].values
    positions = []
    for idx, pred_price in zip(indices, predicted_prices):
        current_price = prices[idx]
        if pred_price > current_price * (1 + thr):
            positions.append(1)
        else:
            positions.append(0)

    # compute returns aligned with positions (positions applies to next step)
    returns_series = pd.Series(prices).pct_change().fillna(0).values
    strat_returns = returns_series[indices] * np.array(positions)

    cum = (1 + pd.Series(strat_returns)).cumprod() - 1
    res = StrategyResult(
        name='lstm_prediction_strategy',
        params={'threshold': thr, 'model': str(model_path.name)},
        cagr=cagr(pd.Series(strat_returns)),
        sharpe=sharpe_ratio(pd.Series(strat_returns)),
        max_drawdown=max_drawdown(cum),
        trades=int(np.abs(np.diff(positions)).sum())
    )
    return res


def select_best_strategy(ticker: str, period: str = '1y', output_dir: str = 'models') -> Dict[str, Any]:
    df = download_history(ticker, period=period)

    results = []
    # try some candidate parameters
    for short, long in [(5,20),(10,50),(5,50)]:
        try:
            results.append(backtest_ma_crossover(df, short=short, long=long))
        except Exception:
            pass

    for window in [3,5,10]:
        for thr in [0.0, 0.002, 0.005]:
            try:
                results.append(backtest_momentum(df, window=window, thr=thr))
            except Exception:
                pass

    for window in [10,20]:
        for k in [1.5,2.0,2.5]:
            try:
                results.append(backtest_bollinger(df, window=window, k=k))
            except Exception:
                pass

    results.append(backtest_buy_and_hold(df))

    # LSTM model based backtest (if model exists)
    try:
        lstm_res = backtest_lstm(df, ticker)
        if lstm_res is not None:
            results.append(lstm_res)
    except Exception:
        pass

    # Choose best by sharpe, tie-breaker by cagr
    best = sorted(results, key=lambda r: (r.sharpe, r.cagr), reverse=True)[0]

    # Convert to JSON-serializable types (convert numpy types to native Python)
    def make_serializable(obj):
        # numpy scalar
        if hasattr(obj, 'item') and not isinstance(obj, (dict, list, tuple, str)):
            try:
                return obj.item()
            except Exception:
                pass
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(make_serializable(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    best_dict = asdict(best)
    serializable = make_serializable(best_dict)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{ticker}_strategy.json")
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    return serializable


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--period', default='1y')
    args = parser.parse_args()
    result = select_best_strategy(args.ticker, period=args.period)
    # Pretty-print JSON-safe result
    print(json.dumps(result, indent=2))
