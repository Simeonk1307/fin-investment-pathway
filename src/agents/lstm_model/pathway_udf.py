"""pathway_udf.py
Defines a Pathway-compatible UDF `predict_and_signal` which calls the
shadow LSTM predictor (`lstm_shadow.predict_stock`) and `signal_generator.generate_signal`.

Import this module in your Pathway pipeline and call `predict_and_signal` as a UDF.
Example usage is shown in comments below.
"""
from __future__ import annotations
import pathway as pw
from typing import Any

from src.agents.lstm_model.lstm_shadow import predict_stock, initialize_manager
from src.agents.lstm_model.signal_generator import generate_signal, load_strategy

import logging 

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)

@pw.udf
def predict_and_signal(
    ticker: str,
    Close: float,
    Volume: float,
    Return: float,
    SMA_5: float = 0.0,
    SMA_10: float = 0.0,
    SMA_20: float = 0.0,
    SMA_30: float = 0.0,
    SMA_50: float = 0.0,
    RSI: float = 0.0,
    MACD: float = 0.0,
    MACD_Signal: float = 0.0,
    BB_Middle: float = 0.0,
    BB_Upper: float = 0.0,
    BB_Lower: float = 0.0,
    Momentum: float = 0.0,
    Momentum5: float = 0.0,
    Volume_Ratio: float = 0.0,
) -> pw.Json:
    """Pathway UDF wrapper.

    Returns a JSON-like dict with predicted_price, signal, reason, confidence, rmse, ready.
    """
    data_point = {
        'Close': Close,
        'Volume': Volume,
        'Return': Return,
        'SMA_5': SMA_5,
        'SMA_10': SMA_10,
        'SMA_20': SMA_20,
        'SMA_30': SMA_30,
        'SMA_50': SMA_50,
        'RSI': RSI,
        'MACD': MACD,
        'MACD_Signal': MACD_Signal,
        'BB_Middle': BB_Middle,
        'BB_Upper': BB_Upper,
        'BB_Lower': BB_Lower,
        'Momentum': Momentum,
        'Momentum5': Momentum5,
        'Volume_Ratio': Volume_Ratio
    }

    # Ensure manager initialized (no-op if already)
    try:
        initialize_manager()
    except Exception:
        pass

    # Get model prediction
    logger.info("[Stocks prediction] : Predicting signal")
    prediction = predict_stock(ticker, data_point)
    logger.info(f"[Stocks prediction] : Data point - {prediction}")
    # import time
    # time.sleep(5)

    # Load strategy metadata (cached by load_strategy internally)
    strategy = load_strategy(ticker)

    # Generate signal
    signal = generate_signal(ticker, prediction, data_point, strategy)

    out = {
        'predicted_price': prediction.get('predicted_price'),
        'current_price': prediction.get('current_price'),
        'rmse': prediction.get('rmse'),
        'ready': prediction.get('ready'),
        'signal': signal.get('signal'),
        'reason': signal.get('reason'),
        'confidence': signal.get('confidence')
    }

    logger.info(f"[Stocks prediction] : Output - {out}")
    return out
