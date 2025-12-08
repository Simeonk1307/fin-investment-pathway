"""live_signals.py
Demonstration integration that (A) loads per-ticker strategy, (B) initializes
the `lstm_shadow` predictor manager, and (C) runs a CSV-simulated streaming loop
to produce `buy`/`hold`/`sell` signals. This mirrors patterns in `lstm_test.py`
but focuses on signal generation and integration.

Usage (CSV simulation):
    python live_signals.py --tickers AAPL MSFT --period 1y

Outputs are written to `outputs/lstm/signals/`.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
import pandas as pd

# Predictor (shadow model) from repo will be imported lazily inside run_demo
from src.agents.lstm_model.signal_generator import generate_signal, load_strategy
from datetime import datetime
import threading


OUTPUT_DIR = 'outputs/lstm/signals'
DATA_DIR = 'outputs/lstm/test/data'


def prepare_streaming_csv(tickers, period='1y') -> tuple[str, str]:
    # Local implementation to avoid importing `lstm_test` (which pulls in Pathway)
    import yfinance as yf
    import numpy as np

    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)

        df['Momentum'] = df['Close'].pct_change(periods=5)
        df['Momentum5'] = df['Close'].pct_change(periods=5)
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        df = df.ffill().bfill()
        return df

    def download_ticker_data(ticker: str, period: str = '1y') -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df = compute_technical_indicators(df)
        df['ticker'] = ticker
        required_cols = [
            'Date', 'ticker', 'Close', 'Volume', 'Return',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_30', 'SMA_50',
            'RSI', 'MACD', 'MACD_Signal', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Momentum', 'Momentum5', 'Volume_Ratio'
        ]
        df = df[required_cols].copy()
        df = df.dropna()
        return df

    # Download and combine
    all_data = []
    for t in tickers:
        try:
            all_data.append(download_ticker_data(t, period))
        except Exception as e:
            print(f"  ❌ Error downloading {t}: {e}")

    if not all_data:
        raise ValueError("No data downloaded for any ticker")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['Date', 'ticker']).reset_index(drop=True)

    streaming = []
    unique_dates = sorted(combined['Date'].unique())

    # Split by date into 80% train / 20% stream
    n_dates = len(unique_dates)
    split_at = max(1, int(n_dates * 0.8))
    train_dates = set(unique_dates[:split_at])
    stream_dates = set(unique_dates[split_at:])

    train_rows = []
    stream_rows = []

    # Assign monotonic time indices separately for train and stream
    train_time_map = {d: i+1 for i, d in enumerate(sorted(train_dates))}
    stream_time_map = {d: i+1 for i, d in enumerate(sorted(stream_dates))}

    for date in sorted(unique_dates):
        date_data = combined[combined['Date'] == date].copy()
        if date in train_dates:
            date_data['time'] = train_time_map[date]
            train_rows.append(date_data)
        else:
            # For streaming we keep time starting after train block when used standalone
            date_data['time'] = stream_time_map[date]
            stream_rows.append(date_data)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    streaming_df = pd.concat(stream_rows, ignore_index=True) if stream_rows else pd.DataFrame()

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    train_path = f"{DATA_DIR}/stocks_train.csv"
    stream_path = f"{DATA_DIR}/stocks_streaming.csv"

    if not train_df.empty:
        train_df.to_csv(train_path, index=False)
    # streaming_df may be empty if not enough dates; still write it
    streaming_df.to_csv(stream_path, index=False)

    return train_path, stream_path


def run_demo(tickers, period='1y', quiet: bool = False, force_train: bool = False):
    # Try to import predictor manager lazily. If torch or the predictor
    # implementation isn't available, fall back to indicator-only signals.
    predictor_available = True
    try:
        from src.agents.lstm_model.lstm_shadow import initialize_manager, predict_stock, save_all_models, _manager, PredictorManager
    except Exception as e:
        predictor_available = False
        if not quiet:
            print(f"[WARN] Predictor module unavailable: {e}")

    if predictor_available:
        if not quiet:
            print("Initializing predictor manager...")
        try:
            initialize_manager(enable_training=True, prefill_buffers=True)
        except Exception as e:
            predictor_available = False
            if not quiet:
                print(f"[WARN] Predictor manager failed to initialize: {e}")

    # Prepare train + streaming CSVs (train: first 80% of dates, stream: last 20%)
    train_path, streaming_path = prepare_streaming_csv(tickers, period)

    # If predictor available, feed training data to the predictor manager
    if predictor_available and os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path)
            # Feed training rows in chronological order to the predictor to prefill buffers
            for time_step in sorted(train_df['time'].unique()):
                time_data = train_df[train_df['time'] == time_step]
                for _, row in time_data.iterrows():
                    ticker = row['ticker']
                    data_point = {
                        'Close': row['Close'],
                        'Volume': row['Volume'],
                        'Return': row.get('Return', 0.0),
                        'SMA_5': row.get('SMA_5'),
                        'SMA_10': row.get('SMA_10'),
                        'SMA_20': row.get('SMA_20'),
                        'SMA_30': row.get('SMA_30'),
                        'SMA_50': row.get('SMA_50'),
                        'RSI': row.get('RSI'),
                        'MACD': row.get('MACD'),
                        'MACD_Signal': row.get('MACD_Signal'),
                        'BB_Middle': row.get('BB_Middle'),
                        'BB_Upper': row.get('BB_Upper'),
                        'BB_Lower': row.get('BB_Lower'),
                        'Momentum': row.get('Momentum'),
                        'Momentum5': row.get('Momentum5'),
                        'Volume_Ratio': row.get('Volume_Ratio')
                    }
                    try:
                        # Use predict_stock to both buffer and possibly trigger retraining
                        predict_stock(ticker, data_point)
                    except Exception:
                        # Ignore errors during training feed; predictor may be partial
                        pass
        except Exception as e:
            print(f"[WARN] Failed to feed training data to predictor: {e}")

    df = pd.read_csv(streaming_path)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_signals = []

    # Keep track of which tickers we've already warned about (to reduce spam)
    warned_tickers = set()

    for time_step in sorted(df['time'].unique()):
        time_data = df[df['time'] == time_step]
        for _, row in time_data.iterrows():
            ticker = row['ticker']
            data_point = {
                'Close': row['Close'],
                'Volume': row['Volume'],
                'Return': row.get('Return', 0.0),
                'SMA_5': row.get('SMA_5'),
                'SMA_10': row.get('SMA_10'),
                'SMA_20': row.get('SMA_20'),
                'SMA_30': row.get('SMA_30'),
                'SMA_50': row.get('SMA_50'),
                'RSI': row.get('RSI'),
                'MACD': row.get('MACD'),
                'MACD_Signal': row.get('MACD_Signal'),
                'BB_Middle': row.get('BB_Middle'),
                'BB_Upper': row.get('BB_Upper'),
                'BB_Lower': row.get('BB_Lower'),
                'Momentum': row.get('Momentum'),
                'Momentum5': row.get('Momentum5'),
                'Volume_Ratio': row.get('Volume_Ratio')
            }

            try:
                if predictor_available:
                    prediction = predict_stock(ticker, data_point)
                else:
                    # Fallback naive prediction when LSTM predictor is unavailable.
                    current_price = float(row['Close'])
                    momentum = data_point.get('Momentum') or data_point.get('Return') or 0.0
                    try:
                        fallback_pred = float(current_price * (1.0 + float(momentum)))
                    except Exception:
                        fallback_pred = float(current_price)
                    prediction = {
                        'ticker': ticker,
                        'predicted_price': fallback_pred,
                        'current_price': current_price,
                        'ready': True,
                        'rmse': 9999.0,
                        'fallback': True
                    }
            except Exception as e:
                # Predictor/model attempted but raised; provide fallback prediction
                if not quiet and ticker not in warned_tickers:
                    print(f"[WARN] Predictor unavailable for {ticker}: {e}")
                    warned_tickers.add(ticker)
                current_price = float(row['Close'])
                momentum = data_point.get('Momentum') or data_point.get('Return') or 0.0
                try:
                    fallback_pred = float(current_price * (1.0 + float(momentum)))
                except Exception:
                    fallback_pred = float(current_price)
                prediction = {
                    'ticker': ticker,
                    'predicted_price': fallback_pred,
                    'current_price': current_price,
                    'ready': False,
                    'rmse': 9999.0,
                    'fallback': True
                }
            strategy = load_strategy(ticker)
            indicators = {k: data_point.get(k) for k in data_point.keys()}
            signal = generate_signal(ticker, prediction, indicators, strategy)

            out = {
                'time': time_step,
                'date': row['Date'],
                'ticker': ticker,
                'current_price': row['Close'],
                'predicted_price': prediction.get('predicted_price'),
                'rmse': prediction.get('rmse'),
                'prediction_timestamp': datetime.utcnow().isoformat() + 'Z',
                'finalized': bool(prediction.get('ready', False) or prediction.get('fallback', False)),
                'signal': signal['signal'],
                'reason': signal.get('reason'),
                'confidence': signal.get('confidence')
            }
            all_signals.append(out)

    out_df = pd.DataFrame(all_signals)
    out_df.to_csv(f"{OUTPUT_DIR}/signals.csv", index=False)
    print(f"Saved signals to {OUTPUT_DIR}/signals.csv")

    # Save modified models (if any) after streaming
    if predictor_available:
        try:
            save_all_models()
        except Exception:
            if not quiet:
                print("[WARN] save_all_models() failed or was unavailable.")

    # If force_train requested, trigger an explicit offline retrain per-ticker
    if force_train and predictor_available:
        try:
            # _manager was imported above when predictor_available
            mgr = None
            try:
                from src.agents.lstm_model.lstm_shadow import _manager
                mgr = _manager
            except Exception:
                mgr = None

            if mgr is None:
                if not quiet:
                    print("[WARN] Predictor manager not accessible for force-train.")
            else:
                for t in sorted({r['ticker'] for r in all_signals}):
                    pred = mgr.get_predictor(t)
                    if pred is None:
                        if not quiet:
                            print(f"[WARN] No predictor for {t}, skipping force-train.")
                        continue
                    try:
                        # Directly invoke shadow retrain (runs in background)
                        retrain_thread = threading.Thread(target=pred._retrain_shadow_model, daemon=True)
                        retrain_thread.start()
                        if not quiet:
                            print(f"Triggered offline retrain for {t}")
                    except Exception as e:
                        if not quiet:
                            print(f"[WARN] Force-train failed for {t}: {e}")
        except Exception as e:
            if not quiet:
                print(f"[WARN] force-train process failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--period', default='1y')
    args = parser.parse_args()
    run_demo(args.tickers, args.period)


if __name__ == '__main__':
    main()
