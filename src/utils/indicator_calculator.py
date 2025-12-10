"""indicator_calculator.py
Compute technical indicators from streaming stock data using circular buffers.
Maintains state per ticker for real-time calculation.
"""
from collections import deque
from typing import Dict, Tuple
import numpy as np
import logging 

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger(__name__)


class StreamingIndicatorCalculator:
    """
    Maintains rolling windows per ticker to compute technical indicators.
    Optimized for streaming - no pandas overhead.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        # Per-ticker buffers: {ticker: deque([price, volume])}
        self.buffers: Dict[str, deque] = {}
        
    def update(self, ticker: str, price: float, volume: float) -> Dict[str, float]:
        """
        Update buffers and compute all indicators for a ticker.
        Returns dict with all computed values.
        """
        # Initialize buffer if needed
        if ticker not in self.buffers:
            self.buffers[ticker] = deque(maxlen=self.lookback)
        
        # Add new data point
        self.buffers[ticker].append({'price': price, 'volume': volume})
        
        # Need minimum data for indicators
        buffer = list(self.buffers[ticker])
        if len(buffer) < 5:
            return self._get_zero_indicators(price, volume)
        
        prices = np.array([d['price'] for d in buffer])
        volumes = np.array([d['volume'] for d in buffer])
        
        # Compute indicators
        indicators = {
            'Close': price,
            'Volume': volume,
            'Return': self._compute_return(prices),
        }
        
        # SMAs
        for window in [5, 10, 20, 30, 50]:
            indicators[f'SMA_{window}'] = self._compute_sma(prices, window)
        
        # RSI
        indicators['RSI'] = self._compute_rsi(prices)
        
        # MACD
        macd, signal = self._compute_macd(prices)
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = signal
        
        # Bollinger Bands
        bb_mid, bb_up, bb_low = self._compute_bollinger(prices)
        indicators['BB_Middle'] = bb_mid
        indicators['BB_Upper'] = bb_up
        indicators['BB_Lower'] = bb_low
        
        # Momentum
        mom = self._compute_momentum(prices, 5)
        indicators['Momentum'] = mom
        indicators['Momentum5'] = mom
        
        # Volume Ratio
        indicators['Volume_Ratio'] = self._compute_volume_ratio(volumes)
        # logger.info(f"indicators: {indicators}")
        
        return indicators
    
    def _get_zero_indicators(self, price: float, volume: float) -> Dict[str, float]:
        """Return zero indicators when insufficient data"""
        return {
            'Close': price,
            'Volume': volume,
            'Return': 0.0,
            'SMA_5': price, 'SMA_10': price, 'SMA_20': price, 
            'SMA_30': price, 'SMA_50': price,
            'RSI': 50.0,  # Neutral
            'MACD': 0.0,
            'MACD_Signal': 0.0,
            'BB_Middle': price,
            'BB_Upper': price,
            'BB_Lower': price,
            'Momentum': 0.0,
            'Momentum5': 0.0,
            'Volume_Ratio': 1.0
        }
    
    def _compute_return(self, prices: np.ndarray) -> float:
        if len(prices) < 2:
            return 0.0
        return float((prices[-1] - prices[-2]) / (prices[-2] + 1e-10))
    
    def _compute_sma(self, prices: np.ndarray, window: int) -> float:
        if len(prices) < window:
            return float(np.mean(prices))
        return float(np.mean(prices[-window:]))
    
    def _compute_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _compute_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        if len(prices) < 26:
            return 0.0, 0.0
        
        # Simple approximation using EMAs
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd = ema12 - ema26
        
        # Signal line (9-period EMA of MACD) - simplified
        signal = macd * 0.9  # Approximation
        
        return float(macd), float(signal)
    
    def _ema(self, prices: np.ndarray, span: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < span:
            return float(np.mean(prices))
        
        alpha = 2 / (span + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return float(ema)
    
    def _compute_bollinger(self, prices: np.ndarray, period: int = 20, k: float = 2.0) -> Tuple[float, float, float]:
        if len(prices) < period:
            mid = float(np.mean(prices))
            return mid, mid, mid
        
        recent = prices[-period:]
        mid = float(np.mean(recent))
        std = float(np.std(recent))
        
        upper = mid + k * std
        lower = mid - k * std
        
        return mid, upper, lower
    
    def _compute_momentum(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        
        return float((prices[-1] - prices[-(period+1)]) / (prices[-(period+1)] + 1e-10))
    
    def _compute_volume_ratio(self, volumes: np.ndarray, period: int = 20) -> float:
        if len(volumes) < 2:
            return 1.0
        
        if len(volumes) < period:
            avg_vol = float(np.mean(volumes[:-1]))
        else:
            avg_vol = float(np.mean(volumes[-period:-1]))
        
        if avg_vol == 0:
            return 1.0
        
        return float(volumes[-1] / avg_vol)


# Global instance
_calculator = None

def get_calculator() -> StreamingIndicatorCalculator:
    """Get or create global calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = StreamingIndicatorCalculator(lookback=50)
    return _calculator
