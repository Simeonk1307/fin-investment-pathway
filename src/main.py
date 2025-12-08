import os
from typing import Dict, Any, List, Tuple
import math
import pandas as pd

import yfinance as yf
from datetime import datetime

from src.sentiment_integration import SentimentIntegrator

# Lightweight orchestrator that combines LSTM (or heuristic) + sentiment to return BUY/SELL/HOLD


def fetch_hist_data(ticker: str, period: str = "180d", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker.replace("-","."))
        df = t.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
        # ensure lowercase cols
        df.columns = [c.lower() for c in df.columns]
        # keep close, open, high, low, volume
        return df
    except Exception:
        return pd.DataFrame()


def _simple_technical_predict(df: pd.DataFrame) -> Tuple[str, float]:
    # returns action, confidence
    if df is None or df.empty:
        return "HOLD", 0.2
    close = df['close']
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    latest = close.iloc[-1]
    # momentum
    mom = (latest - close.iloc[-2]) / close.iloc[-2] if len(close) >= 2 else 0.0
    if latest > sma20 and latest > sma50 and mom > 0:
        return "BUY", min(0.95, 0.5 + abs(mom) * 5)
    if latest < sma20 and latest < sma50 and mom < 0:
        return "SELL", min(0.95, 0.5 + abs(mom) * 5)
    return "HOLD", 0.4


def compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute simple technical indicators: SMA20, SMA50, MACD, RSI(14), Bollinger bands."""
    out: Dict[str, Any] = {}
    if df is None or df.empty:
        return out
    close = df['close'].astype(float)
    out['sma20'] = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.mean())
    out['sma50'] = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else out['sma20']
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out['macd'] = float(macd.iloc[-1])
    out['macd_signal'] = float(signal.iloc[-1])
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0).fillna(0)
    down = -1 * delta.clip(upper=0).fillna(0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    out['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else 50.0
    # Bollinger
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out['bb_upper'] = float((ma20 + 2 * std20).iloc[-1]) if len(close) >= 20 else float(close.max())
    out['bb_lower'] = float((ma20 - 2 * std20).iloc[-1]) if len(close) >= 20 else float(close.min())
    out['latest'] = float(close.iloc[-1])
    return out


def predict_lstm_like(ticker: str, hist: pd.DataFrame) -> Dict[str, Any]:
    # Try to call existing LSTM model components if present; otherwise fallback to technical rule
    try:
        # attempt to import project LSTM predictor
        from src.agents.lstm_model.signal_generator import predict_signal

        action, conf = predict_signal(ticker, hist)
        return {"action": action, "confidence": conf}
    except Exception:
        action, conf = _simple_technical_predict(hist)
        return {"action": action, "confidence": conf}


def combine_lstm_and_sentiment(lstm_out: Dict[str, Any], sentiment_score: float) -> Dict[str, Any]:
    # sentiment_score in [-1,1]; lstm_out confidence in [0,1]
    mapping = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}
    lstm_val = mapping.get(lstm_out.get("action","HOLD"), 0.0)
    lstm_conf = float(lstm_out.get("confidence", 0.5))
    strength = abs(sentiment_score)
    # LSTM weight decays slightly as sentiment strength grows (keeps preference to LSTM)
    lstm_weight = max(0.6, 0.9 - 0.3 * strength)
    sent_weight = 1.0 - lstm_weight
    final_score = lstm_weight * lstm_val * lstm_conf + sent_weight * sentiment_score
    # map to action
    if final_score > 0.2:
        action = "BUY"
    elif final_score < -0.2:
        action = "SELL"
    else:
        action = "HOLD"
    confidence = min(0.995, 0.2 + 0.8 * abs(final_score))
    return {
        "action": action,
        "confidence": float(confidence),
        "score": float(final_score),
        "lstm_weight": lstm_weight,
        "sent_weight": sent_weight,
    }


sent_int = SentimentIntegrator()


def orchestrate_signal(ticker: str, hist_df: pd.DataFrame) -> Dict[str, Any]:
    # Build a short text summary to score
    head_text = "".join([str(x) for x in hist_df['close'].tail(5).tolist()])
    lstm_out = predict_lstm_like(ticker, hist_df)
    indicators = compute_indicators(hist_df)
    # Try to collect a representative news text from kg
    # Use kgn json lines if available
    kg_news = sent_int.kg_news_for_ticker(ticker)
    sample_text = ""
    sources = []
    if kg_news:
        n = kg_news[0]
        sample_text = (n.get('title') or '') + '\n' + (n.get('content') or '')
        url = n.get('url') or n.get('link')
        if url:
            sources.append(url)
    # fallback use head_text
    if not sample_text:
        sample_text = f"Recent prices: {head_text}"

    sent_score = sent_int.combined_score(sample_text, ticker)
    combined = combine_lstm_and_sentiment(lstm_out, sent_score)

    # Guardrail heuristics
    guardrail = []
    if indicators.get('latest') and indicators.get('bb_upper') and indicators.get('bb_lower'):
        if indicators['latest'] > indicators['bb_upper'] * 1.02:
            guardrail.append('Price far above upper Bollinger band — possible overbought')
        if indicators['latest'] < indicators['bb_lower'] * 0.98:
            guardrail.append('Price far below lower Bollinger band — possible oversold')
    if abs(sent_score) > 0.9:
        guardrail.append('Extremely strong sentiment detected — verify sources')

    # Compose reasoning from components
    parts = []
    parts.append(f"LSTM predicted {lstm_out.get('action')} with confidence {lstm_out.get('confidence'):.2f}")
    parts.append(f"Sentiment score {sent_score:.3f} (kg-influenced)")
    if indicators:
        parts.append(f"Price {indicators.get('latest'):.2f}, SMA20 {indicators.get('sma20'):.2f}, SMA50 {indicators.get('sma50'):.2f}")
        parts.append(f"MACD {indicators.get('macd'):.4f} (signal {indicators.get('macd_signal'):.4f}), RSI {indicators.get('rsi'):.1f}")
    reason = '; '.join(parts)

    return {
        "ticker": ticker,
        "action": combined['action'],
        "confidence": combined['confidence'],
        "score": combined['score'],
        "lstm_weight": combined['lstm_weight'],
        "sent_weight": combined['sent_weight'],
        "reason": reason,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "indicators": indicators,
        "guardrail": guardrail,
    }


def backtest_strategy(ticker: str, hist: pd.DataFrame, capital: float = 1000.0) -> Dict[str, Any]:
    # Simple backtest: apply daily predicted action and hold until opposite action
    df = hist.copy().reset_index()
    if df.empty:
        return {"equity_curve": pd.DataFrame(), "metrics": {}}
    equity = []
    cash = capital
    position = 0.0
    last_price = None
    equities = []
    preds = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        price = float(row['close'])
        window = hist.iloc[: max(1, idx+1)].copy()
        # predict
        out = predict_lstm_like(ticker, window)
        preds.append(out.get('confidence', 0.5) * (1 if out.get('action')=='BUY' else -1 if out.get('action')=='SELL' else 0))
        # trade logic: if BUY and no position, buy; if SELL and have position sell
        if out.get('action') == 'BUY' and position == 0:
            position = cash / price
            cash = 0.0
        elif out.get('action') == 'SELL' and position > 0:
            cash = position * price
            position = 0.0
        total = cash + position * price
        equities.append(total)
        last_price = price
    eq_df = pd.DataFrame({"equity": equities}, index=df['Date'])
    # buy-hold
    bh_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1.0) * capital
    strat_return = eq_df['equity'].iloc[-1] - capital
    # metrics: returns, MAE between predicted sign and actual return sign, simple sharpe
    rets = df['close'].pct_change().fillna(0)
    strat_rets = pd.Series(equities).pct_change().fillna(0)
    sharpe = (strat_rets.mean() / (strat_rets.std() + 1e-9)) * (252 ** 0.5)
    drawdown = (pd.Series(equities).cummax() - pd.Series(equities)).max()
    metrics = {
        "strategy_return": float(strat_return),
        "buy_hold_return": float(bh_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown),
        "mae": float((pd.Series(preds).abs()).mean()),
    }
    return {"equity_curve": eq_df, "metrics": metrics}


def get_agent_insights(tickers: List[str]) -> List[Dict[str, Any]]:
    out = []
    for t in tickers:
        df = fetch_hist_data(t, period="7d", interval="1d")
        sig = orchestrate_signal(t, df) if not df.empty else {"action":"HOLD","confidence":0.1}
        # gather provenance from sentiment integrator
        kg_news = sent_int.kg_news_for_ticker(t)
        sources = []
        for n in kg_news[:5]:
            if n.get('url'):
                sources.append(n.get('url'))
        item = {
            "ticker": t,
            "action": sig.get('action'),
            "confidence": sig.get('confidence'),
            "timestamp": sig.get('timestamp'),
            "reason": sig.get('reason'),
            "sources": sources,
            "guardrail": [],
            "rejected": sig.get('confidence',0) < 0.25,
            "rejection_reason": "Low confidence" if sig.get('confidence',0) < 0.25 else "",
        }
        out.append(item)
    return out


if __name__ == "__main__":
    print("Main orchestrator module. Import functions from this file to use the API.")
