"""signal_generator.py
Convert prediction results + indicators into discrete actions: 'buy', 'hold', 'sell'.

This module loads per-ticker strategy metadata saved by `strategy_selector.py` and
applies the chosen rules. It's intentionally simple and deterministic.
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any


def load_strategy(ticker: str, model_dir: str = 'models') -> Dict[str, Any]:
    path = os.path.join(model_dir, f"{ticker}_strategy.json")
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def generate_signal(ticker: str, prediction_result: Dict[str, Any], indicators: Dict[str, Any], strategy: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a dict with keys: signal, reason, confidence

    - `prediction_result` is the dict returned by `predict_stock()` (may include `predicted_price`, `rmse`, `ready`).
    - `indicators` contains current indicator values like SMA_5, SMA_20, Momentum, etc.
    - `strategy` is the previously selected strategy (if None, will be loaded from models folder).
    """
    if strategy is None:
        strategy = load_strategy(ticker)

    pred = prediction_result.get('predicted_price')
    current = float(prediction_result.get('current_price', 0.0))
    ready = bool(prediction_result.get('ready', False))
    rmse = float(prediction_result.get('rmse', 0.0))

    # Basic safety: if model not ready or RMSE too high, return hold
    if not ready or rmse > 9999:
        return {'signal': 'hold', 'reason': 'model_not_ready_or_unreliable', 'confidence': 0.0}

    name = strategy.get('name') if strategy else None
    params = strategy.get('params', {}) if strategy else {}

    # Threshold strategy when LSTM provided predicted price
    if pred is not None:
        # default thresholds
        up_thr = params.get('up_threshold', 0.005)
        down_thr = params.get('down_threshold', 0.005)
        # allow strategy to define thresholds
        if name == 'momentum_threshold' and 'threshold' in params:
            # increase buy threshold for momentum
            up_thr = max(up_thr, params['threshold'])

        pct = (pred - current) / (current + 1e-12)
        if pct > up_thr:
            return {'signal': 'buy', 'reason': 'predicted_up', 'confidence': float(min(1.0, pct / 0.1))}
        if pct < -down_thr:
            return {'signal': 'sell', 'reason': 'predicted_down', 'confidence': float(min(1.0, abs(pct) / 0.1))}

    # Fallback: use rule defined in strategy
    if name == 'moving_average_crossover':
        short = params.get('short', 5)
        long = params.get('long', 20)
        sma_short = indicators.get(f'SMA_{short}')
        sma_long = indicators.get(f'SMA_{long}')
        if sma_short is None or sma_long is None:
            return {'signal': 'hold', 'reason': 'missing_indicators', 'confidence': 0.0}
        if sma_short > sma_long:
            return {'signal': 'buy', 'reason': 'ma_short_above_long', 'confidence': 0.6}
        if sma_short < sma_long:
            return {'signal': 'sell', 'reason': 'ma_short_below_long', 'confidence': 0.6}

    if name == 'bollinger_breakout':
        close = indicators.get('Close')
        upper = indicators.get('BB_Upper')
        lower = indicators.get('BB_Lower')
        if close is None or upper is None or lower is None:
            return {'signal': 'hold', 'reason': 'missing_indicators', 'confidence': 0.0}
        if close > upper:
            return {'signal': 'buy', 'reason': 'price_above_upper', 'confidence': 0.7}
        if close < lower:
            return {'signal': 'sell', 'reason': 'price_below_lower', 'confidence': 0.7}

    if name == 'momentum_threshold':
        mom = indicators.get('Momentum', 0.0)
        thr = params.get('threshold', 0.0)
        if mom > thr:
            return {'signal': 'buy', 'reason': 'momentum_positive', 'confidence': 0.6}
        if mom < -thr:
            return {'signal': 'sell', 'reason': 'momentum_negative', 'confidence': 0.6}

    # Default fallback
    return {'signal': 'hold', 'reason': 'no_rule_triggered', 'confidence': 0.0}


if __name__ == '__main__':
    # simple manual test
    sample = {'predicted_price': 105.0, 'current_price': 100.0, 'ready': True, 'rmse': 0.5}
    print(generate_signal('AAPL', sample, {'Close':100.0, 'SMA_5':101.0, 'SMA_20':99.0}))
