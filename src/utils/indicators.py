# src/agents/indicators.py

import numpy as np

class Indicators:
    @staticmethod
    def calc(closes, highs, lows, volumes):
        """
        INPUT: 
        - closes: [100.5, 101.2, 102.0, ...]  ← list of close prices
        - highs:  [101.0, 102.0, 103.0, ...]  ← list of high prices
        - lows:   [100.0, 100.5, 101.0, ...]  ← list of low prices
        - volumes:[50000, 60000, 45000, ...]  ← list of volumes
        
        OUTPUT: Dictionary with all 15 indicators
        """
        
        if len(closes) < 2:
            return {} 
        
        c = np.array(closes)
        h = np.array(highs)
        l = np.array(lows)
        v = np.array(volumes)
        
        sma5 = c[-5:].mean() if len(c) >= 5 else c.mean()
     
        sma20 = c[-20:].mean() if len(c) >= 20 else c.mean()
       
        ema12 = c[-12:].mean() if len(c) >= 12 else c.mean()
        
        ema26 = c[-26:].mean() if len(c) >= 26 else c.mean()
        
        deltas = np.diff(c[-15:]) if len(c) > 15 else np.diff(c)
        
        gains = np.maximum(deltas, 0).mean()
        losses = np.maximum(-deltas, 0).mean()
        
        rsi = 100 - (100 / (1 + gains / (losses + 1e-10)))
        
        macd = ema12 - ema26
        signal = macd * 0.8
        
        std = c[-20:].std() if len(c) >= 20 else c.std()
        bb_mid = sma20 
        bb_upper = bb_mid + 2 * std 
        bb_lower = bb_mid - 2 * std 
        
      
        tr = np.maximum(
            h - l,  
            np.maximum(
                np.abs(h - np.roll(c, 1)),
                np.abs(l - np.roll(c, 1))
            )
        )
        atr = tr[-14:].mean() if len(tr) >= 14 else tr.mean()
        
    
        obv = np.sum(v[1:] * np.sign(np.diff(c)))
        
        vol_avg = v.mean()
        
        current = c[-1]
        
        change_pct = (c[-1] - c[0]) / c[0] * 100 if c[0] != 0 else 0.0
        
        return {
            "current": float(current),
            "sma5": float(sma5),
            "sma20": float(sma20),
            "ema12": float(ema12),
            "ema26": float(ema26),
            "rsi": float(rsi),
            "macd": float(macd),
            "macd_signal": float(signal),
            "bb_upper": float(bb_upper),
            "bb_mid": float(bb_mid),
            "bb_lower": float(bb_lower),
            "atr": float(atr),
            "obv": float(obv),
            "vol_avg": float(vol_avg),
            "change_pct": float(change_pct)
        }