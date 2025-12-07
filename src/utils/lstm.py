import torch
import torch.nn as nn
import numpy as np


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, 2, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class Predictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = LSTM().eval()
        return cls._instance
    
    def predict(self, prices):
        """
        INPUT: [100.5, 101.2, 102.0, ..., 105.3] ← list of prices
        OUTPUT: 105.8  ← predicted next price
        """
        
        if len(prices) < 20:
            return prices[-1]
        
        x = np.array(prices[-20:]).reshape(1, -1, 1)
        
        mean = x.mean()
        std = x.std() + 1e-8
        x_normalized = (x - mean) / std
        
        with torch.no_grad():
            pred = self.model(torch.FloatTensor(x_normalized))
        
        predicted_price = pred.item() * std + mean
        
        return float(predicted_price)