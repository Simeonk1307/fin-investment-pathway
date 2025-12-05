#!/usr/bin/env python3
"""
LSTM Stock Prediction - Real-time Testing with Auto Buffer Fill
================================================================

Real-time stock prediction using trained LSTM models.
Automatically pre-fills buffers with historical data for immediate predictions.

FEATURES:
- Auto-loads trained models from models/ directory
- Pre-fills buffers with historical data (predictions start immediately!)
- Real-time prediction via any streaming source
- Tracks model performance (RMSE, prediction count)
- Online retraining based on RMSE thresholds
- Independent buffers per ticker

USAGE:
    python lstm_test.py

NO ARGUMENTS NEEDED - Everything configured below.

Author: Stock Prediction Team
"""

import os
import json
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import yfinance as yf

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Model directory
MODEL_DIR = "models"
OUTPUT_DIR = "outputs/lstm/config/lstm_predictions"

# Buffer configuration
PREFILL_BUFFER = True  # Auto-fill buffer with historical data (predictions start immediately!)

# Online learning configuration
ENABLE_ONLINE_LEARNING = True
RMSE_THRESHOLD = 5.0                    # Retrain if RMSE exceeds this
MIN_PREDICTIONS_BEFORE_RETRAIN = 20     # Minimum predictions before retraining
RETRAIN_BATCH_SIZE = 10                 # Samples to accumulate before batch retrain
RETRAIN_ITERATIONS = 5                  # Mini-batch iterations during retrain

# ═══════════════════════════════════════════════════════════════════════════
# LSTM MODEL (Same architecture as training)
# ═══════════════════════════════════════════════════════════════════════════

class StockLSTM(nn.Module):
    """LSTM model for stock price prediction"""
    
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


# ═══════════════════════════════════════════════════════════════════════════
# REAL-TIME PREDICTOR (Per Ticker)
# ═══════════════════════════════════════════════════════════════════════════

class RealtimeStockPredictor:
    """
    Real-time LSTM predictor for a single ticker.
    
    Manages:
    - Model loading and caching
    - Automatic buffer pre-filling with historical data
    - Sequential data buffering
    - Prediction generation
    - Performance tracking (RMSE)
    - Online retraining
    """
    
    def __init__(self, ticker: str, enable_training: bool = True, prefill_buffer: bool = True):
        self.ticker = ticker
        self.enable_training = enable_training
        self.prefill_buffer = prefill_buffer
        
        # Model components
        self.model = None
        self.scaler = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model metadata
        self.lookback = None
        self.input_size = None
        self.feature_columns = None
        
        # Data buffer for sequences
        self.buffer = deque()
        
        # Performance tracking
        self.prediction_count = 0
        self.training_count = 0
        self.recent_errors = deque(maxlen=50)
        self.current_rmse = 0.0
        
        # Retraining buffer
        self.retrain_buffer = deque(maxlen=RETRAIN_BATCH_SIZE * 2)
        
        # Load model
        self._load_model()
        
        # Pre-fill buffer with historical data if requested
        if self.prefill_buffer:
            self._prefill_buffer_with_history()
    
    def _load_model(self):
        """Load pre-trained model and scaler"""
        model_path = f"{MODEL_DIR}/{self.ticker}_lstm.pt"
        scaler_path = f"{MODEL_DIR}/{self.ticker}_scaler.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ Scaler not found: {scaler_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.input_size = checkpoint['input_size']
        self.lookback = checkpoint['lookback']
        self.feature_columns = checkpoint.get('feature_columns', None)
        
        # Initialize model
        self.model = StockLSTM(
            input_size=self.input_size,
            hidden_size=checkpoint.get('hidden_size', 64),
            num_layers=checkpoint.get('num_layers', 2),
            dropout=checkpoint.get('dropout', 0.2)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Setup optimizer for online learning
        if self.enable_training:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        # Initialize buffer
        self.buffer = deque(maxlen=self.lookback)
        
        print(f"✅ Loaded model for {self.ticker}")
        print(f"   Input size: {self.input_size}, Lookback: {self.lookback}")
    
    def _compute_indicators(self, df):
        """Compute technical indicators (same as training)"""
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
        
        # Momentum
        df['Momentum'] = df['Close'].pct_change(periods=5)
        
        # Volume Ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Fill NaN
        df = df.ffill().bfill()
        
        return df
    
    def _prefill_buffer_with_history(self):
        """
        Pre-fill buffer with recent historical data so predictions can start immediately.
        Downloads last 'lookback' days of data to warm up the buffer.
        """
        try:
            print(f"🔄 Pre-filling buffer for {self.ticker} with {self.lookback} days of history...")
            
            # Download recent historical data (extra days for indicator calculation)
            df = yf.download(
                self.ticker, 
                period=f"{self.lookback + 50}d",
                progress=False, 
                auto_adjust=True
            )
            
            if df.empty:
                print(f"⚠️  Could not download history for {self.ticker}, buffer will fill from stream")
                return
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            
            # Compute indicators (same as training)
            df = self._compute_indicators(df)
            
            # Extract features
            if self.feature_columns is None:
                feature_names = [
                    'Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50',
                    'RSI', 'MACD', 'MACD_Signal', 'BB_Middle',
                    'Momentum', 'Volume_Ratio'
                ]
            else:
                feature_names = self.feature_columns
            
            # Select required columns
            df = df[feature_names].copy()
            df = df.dropna()
            
            # Take last 'lookback' rows
            recent_data = df.tail(self.lookback)
            
            if len(recent_data) < self.lookback:
                print(f"⚠️  Only {len(recent_data)}/{self.lookback} history available for {self.ticker}")
            
            # Scale and add to buffer
            for _, row in recent_data.iterrows():
                features = row.values.reshape(1, -1)
                features_scaled = self.scaler.transform(features)[0]
                self.buffer.append(features_scaled)
            
            print(f"✅ Buffer pre-filled: {len(self.buffer)}/{self.lookback}")
            
        except Exception as e:
            print(f"⚠️  Buffer pre-fill failed for {self.ticker}: {e}")
            print(f"   Buffer will fill from streaming data instead")
    
    def _extract_features(self, data_point: Dict) -> Optional[np.ndarray]:
        """Extract features from data point"""
        if self.feature_columns is None:
            # Default feature order
            feature_names = [
                'Close', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50',
                'RSI', 'MACD', 'MACD_Signal', 'BB_Middle',
                'Momentum', 'Volume_Ratio'
            ]
        else:
            feature_names = self.feature_columns
        
        try:
            features = np.array([
                float(data_point.get(name, 0))
                for name in feature_names
            ])
            return features
        except Exception as e:
            print(f"❌ Feature extraction error for {self.ticker}: {e}")
            return None
    
    def predict(self, data_point: Dict) -> Dict:
        """
        Make prediction for next price using current data point.
        
        Args:
            data_point: Dict with feature values (Close, Volume, SMA_5, etc.)
        
        Returns:
            Dict with prediction results and metadata
        """
        # Extract features
        features = self._extract_features(data_point)
        if features is None:
            return {
                'ticker': self.ticker,
                'predicted_price': None,
                'current_price': data_point.get('Close', 0),
                'ready': False,
                'error': 'Feature extraction failed'
            }
        
        # Scale features
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))[0]
        except Exception as e:
            return {
                'ticker': self.ticker,
                'predicted_price': None,
                'current_price': data_point.get('Close', 0),
                'ready': False,
                'error': f'Scaling failed: {e}'
            }
        
        # Add to buffer
        self.buffer.append(features_scaled)
        
        # Store for potential retraining
        self.retrain_buffer.append({
            'features': features_scaled,
            'actual_price': data_point.get('Close', 0)
        })
        
        # Check if buffer is full
        if len(self.buffer) < self.lookback:
            return {
                'ticker': self.ticker,
                'predicted_price': None,
                'current_price': data_point.get('Close', 0),
                'ready': False,
                'buffer_size': len(self.buffer),
                'buffer_needed': self.lookback,
                'message': f'Buffering... {len(self.buffer)}/{self.lookback}'
            }
        
        # Make prediction
        sequence = np.array(list(self.buffer)).reshape(1, self.lookback, -1)
        sequence_tensor = torch.FloatTensor(sequence).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(sequence_tensor).cpu().item()
        
        # Denormalize prediction
        pred_full = np.zeros((1, self.scaler.n_features_in_))
        pred_full[0, 0] = pred_scaled
        predicted_price = self.scaler.inverse_transform(pred_full)[0, 0]
        
        current_price = float(data_point.get('Close', 0))
        self.prediction_count += 1
        
        # Calculate error
        error = abs(predicted_price - current_price)
        self.recent_errors.append(error)
        self.current_rmse = np.sqrt(np.mean([e**2 for e in self.recent_errors]))
        
        result = {
            'ticker': self.ticker,
            'predicted_price': float(predicted_price),
            'current_price': current_price,
            'ready': True,
            'prediction_count': self.prediction_count,
            'training_count': self.training_count,
            'rmse': float(self.current_rmse),
            'buffer_size': len(self.buffer),
            'retrained': False
        }
        
        # Check if retraining is needed
        if (self.enable_training and 
            self.prediction_count >= MIN_PREDICTIONS_BEFORE_RETRAIN and
            len(self.retrain_buffer) >= RETRAIN_BATCH_SIZE):
            
            should_retrain = self.current_rmse > RMSE_THRESHOLD
            
            if should_retrain:
                retrain_success = self._retrain_model()
                result['retrained'] = retrain_success
                if retrain_success:
                    print(f"🔄 [{self.ticker}] Model retrained. New RMSE: {self.current_rmse:.4f}")
        
        return result
    
    def _retrain_model(self) -> bool:
        """
        Retrain model using accumulated buffer data.
        NOTE: This creates a NEW model state in memory only.
        Does NOT modify the saved model file in models/ directory.
        """
        try:
            # Prepare batch from retrain buffer
            sequences = []
            targets = []
            
            buffer_list = list(self.retrain_buffer)
            
            for i in range(self.lookback, len(buffer_list)):
                seq = [buffer_list[j]['features'] for j in range(i-self.lookback, i)]
                target = buffer_list[i]['features'][0]  # Close price (index 0)
                sequences.append(seq)
                targets.append(target)
            
            if len(sequences) < 5:
                return False
            
            sequences = torch.FloatTensor(sequences).to(self.device)
            targets = torch.FloatTensor(targets).unsqueeze(1).to(self.device)
            
            # Train for a few iterations (in-memory update only)
            self.model.train()
            for _ in range(RETRAIN_ITERATIONS):
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            self.training_count += 1
            
            # NOTE: We do NOT save to disk here
            # The updated model exists only in memory during this session
            
            return True
        
        except Exception as e:
            print(f"❌ Retraining failed for {self.ticker}: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get current predictor status"""
        return {
            'ticker': self.ticker,
            'prediction_count': self.prediction_count,
            'training_count': self.training_count,
            'buffer_size': len(self.buffer),
            'buffer_needed': self.lookback,
            'rmse': float(self.current_rmse),
            'retrain_buffer_size': len(self.retrain_buffer)
        }


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTOR MANAGER (Multi-Ticker)
# ═══════════════════════════════════════════════════════════════════════════

class PredictorManager:
    """Manages predictors for multiple tickers"""
    
    def __init__(self, enable_training: bool = True, prefill_buffers: bool = True):
        self.predictors: Dict[str, RealtimeStockPredictor] = {}
        self.enable_training = enable_training
        self.prefill_buffers = prefill_buffers
    
    def get_predictor(self, ticker: str) -> Optional[RealtimeStockPredictor]:
        """Get or create predictor for ticker"""
        if ticker not in self.predictors:
            try:
                self.predictors[ticker] = RealtimeStockPredictor(
                    ticker,
                    enable_training=self.enable_training,
                    prefill_buffer=self.prefill_buffers
                )
            except Exception as e:
                print(f"❌ Failed to load predictor for {ticker}: {e}")
                return None
        
        return self.predictors[ticker]
    
    def predict(self, ticker: str, data_point: Dict) -> Dict:
        """Make prediction for ticker"""
        predictor = self.get_predictor(ticker)
        if predictor is None:
            return {
                'ticker': ticker,
                'ready': False,
                'error': 'Predictor not available'
            }
        
        return predictor.predict(data_point)
    
    def get_all_statuses(self) -> Dict:
        """Get status of all predictors"""
        return {
            ticker: predictor.get_status()
            for ticker, predictor in self.predictors.items()
        }


# Global manager instance
_manager = None


def initialize_manager(enable_training: bool = ENABLE_ONLINE_LEARNING, 
                      prefill_buffers: bool = PREFILL_BUFFER):
    """Initialize global predictor manager"""
    global _manager
    _manager = PredictorManager(
        enable_training=enable_training,
        prefill_buffers=prefill_buffers
    )
    print("✅ LSTM Predictor Manager initialized")


def predict_stock(ticker: str, data_point: Dict) -> Dict:
    """
    Main prediction function - call this from Pathway or any pipeline.
    
    Args:
        ticker: Stock ticker symbol
        data_point: Dict with features (Close, Volume, SMA_5, etc.)
    
    Returns:
        Dict with prediction results
    """
    global _manager
    if _manager is None:
        initialize_manager()
    
    return _manager.predict(ticker, data_point)


# ═══════════════════════════════════════════════════════════════════════════
# DEMO PIPELINE (For Testing Without Pathway)
# ═══════════════════════════════════════════════════════════════════════════

def demo_streaming_pipeline():
    """
    Demo pipeline to test predictions without Pathway.
    Replace this with actual Pathway integration.
    """
    print("\n" + "="*70)
    print("🚀 LSTM REAL-TIME PREDICTION - DEMO MODE")
    print("="*70)
    
    # Initialize manager with buffer pre-fill
    initialize_manager(
        enable_training=ENABLE_ONLINE_LEARNING,
        prefill_buffers=PREFILL_BUFFER
    )
    
    # Generate demo data (simulates streaming)
    print("\n📊 Generating demo streaming data...")
    
    demo_data = []
    
    # Generate 25 timesteps for AAPL
    for i in range(25):
        demo_data.append({
            'ticker': 'AAPL',
            'Close': 150.0 + i * 0.5,
            'Volume': 1000000 + i * 10000,
            'SMA_5': 150.0 + i * 0.45,
            'SMA_20': 150.0 + i * 0.40,
            'SMA_50': 150.0 + i * 0.35,
            'RSI': 65.0 + i * 0.1,
            'MACD': 0.35 + i * 0.01,
            'MACD_Signal': 0.30 + i * 0.01,
            'BB_Middle': 150.0 + i * 0.45,
            'Momentum': 0.005,
            'Volume_Ratio': 1.05
        })
    
    # Generate 25 timesteps for MSFT
    for i in range(25):
        demo_data.append({
            'ticker': 'MSFT',
            'Close': 340.0 + i * 0.3,
            'Volume': 800000 + i * 8000,
            'SMA_5': 340.0 + i * 0.28,
            'SMA_20': 340.0 + i * 0.25,
            'SMA_50': 340.0 + i * 0.22,
            'RSI': 62.0 + i * 0.1,
            'MACD': 0.45 + i * 0.01,
            'MACD_Signal': 0.40 + i * 0.01,
            'BB_Middle': 340.0 + i * 0.28,
            'Momentum': 0.004,
            'Volume_Ratio': 0.98
        })
    
    # Process data points
    predictions = []
    
    print("\n🔄 Processing streaming data...\n")
    
    for idx, data_point in enumerate(demo_data):
        result = predict_stock(data_point['ticker'], data_point)
        
        if result['ready']:
            print(f"✅ [{result['ticker']}] Prediction #{result['prediction_count']}")
            print(f"   Current: ${result['current_price']:.2f}")
            print(f"   Predicted: ${result['predicted_price']:.2f}")
            print(f"   RMSE: {result['rmse']:.4f}")
            if result.get('retrained'):
                print(f"   🔄 MODEL RETRAINED!")
            print()
            
            predictions.append(result)
        else:
            if idx < 5:  # Print first few buffering messages
                print(f"⏳ [{result['ticker']}] {result.get('message', 'Buffering...')}")
    
    # Save predictions
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
        
        with open(f"{OUTPUT_DIR}/predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print("="*70)
        print(f"✅ Demo complete! Generated {len(predictions)} predictions")
        print(f"📁 Saved to: {OUTPUT_DIR}/")
    else:
        print("⚠️  No predictions generated. Check buffer pre-fill or data.")
    
    print("="*70)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("📈 LSTM STOCK PREDICTION - TESTING & STREAMING")
    print("="*70)
    
    # Check if models exist
    if not os.path.exists(MODEL_DIR):
        print(f"\n❌ Model directory not found: {MODEL_DIR}")
        print("👉 Run train_lstm_stocks.py first!")
        return
    
    # Run demo pipeline
    demo_streaming_pipeline()
    
    # Show integration info
    print("\n💡 Integration with Pathway or other systems:")
    print("  1. Import: from lstm_test import initialize_manager, predict_stock")
    print("  2. Initialize: initialize_manager()")
    print("  3. Predict: result = predict_stock(ticker, data_point)")
    print("  4. Buffer auto-fills on first call per ticker")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()