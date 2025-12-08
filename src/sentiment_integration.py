import os
import json
from datetime import datetime, timezone
import math
from typing import List, Dict, Optional, Tuple

try:
    from src.agents.finbert import FinBertSentimentAnalyzer
except Exception:
    FinBertSentimentAnalyzer = None


def _read_local_kg_news(path: str) -> List[Dict]:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def compute_age_hours(ts: int) -> float:
    try:
        then = datetime.fromtimestamp(ts, tz=timezone.utc)
        diff = datetime.now(tz=timezone.utc) - then
        return diff.total_seconds() / 3600.0
    except Exception:
        return 9999.0


class SentimentIntegrator:
    def __init__(self, kg_json_path: Optional[str] = None):
        self.kg_json_path = kg_json_path or os.path.join(os.getcwd(), "kg_news_output.jsonl")
        self.finbert = FinBertSentimentAnalyzer() if FinBertSentimentAnalyzer else None

    def score_text(self, text: str) -> Tuple[float, float, float]:
        """Return (neg, neutral, pos) probabilities. If FinBERT not available, return neutral."""
        if not self.finbert:
            return (0.0, 1.0, 0.0)
        try:
            return self.finbert.analyze_sentiment(text)
        except Exception:
            return (0.0, 1.0, 0.0)

    def kg_news_for_ticker(self, ticker: str, within_days: int = 7) -> List[Dict]:
        all_news = _read_local_kg_news(self.kg_json_path)
        cutoff = datetime.now().timestamp() - within_days * 24 * 3600
        out = []
        for n in all_news:
            sym = n.get("symbol") or n.get("symbols") or n.get("ticker")
            # symbol may be a comma-separated string
            found = False
            if isinstance(sym, str) and ticker.upper() in sym.upper():
                found = True
            if isinstance(sym, list) and ticker.upper() in [s.upper() for s in sym]:
                found = True
            if found:
                ts = int(n.get("timestamp") or n.get("ts") or 0)
                if ts >= cutoff:
                    out.append(n)
        return out

    def kg_effect_score(self, ticker: str, tau_hours: float = 48.0) -> float:
        """Return a KG-derived sentiment score in [-1,1], applying exponential decay by age.
        This implementation uses local kg jsonl as fallback.
        """
        news = self.kg_news_for_ticker(ticker, within_days=7)
        if not news:
            return 0.0
        weighted = 0.0
        weight_sum = 0.0
        for n in news:
            text = (n.get("title") or "") + "\n" + (n.get("content") or "")
            neg, neu, pos = self.score_text(text)
            # sentiment in [-1,1]
            s = pos - neg
            age_h = compute_age_hours(int(n.get("timestamp") or 0))
            decay = math.exp(-age_h / tau_hours)
            weighted += s * decay
            weight_sum += decay
        if weight_sum <= 0:
            return 0.0
        return weighted / weight_sum

    def combined_score(self, text: str, ticker: str) -> float:
        """Return final combined sentiment in [-1,1].
        We combine direct FinBERT score from `text` and KG effect for `ticker`.
        """
        neg, neu, pos = self.score_text(text)
        direct = pos - neg
        kg = self.kg_effect_score(ticker)
        # coefficient: KG contribution already decayed by age; combine additively
        final = direct + kg * 0.8
        # normalize to [-1,1]
        if final > 1:
            final = 1.0
        if final < -1:
            final = -1.0
        return final


__all__ = ["SentimentIntegrator"]
