#!/usr/bin/env python3
"""
LSTM Stock Prediction - WITH PROPER LOGGING FOR PATHWAY

KEY FIX: Use logging.getLogger() instead of print() for Pathway compatibility
"""

import os
import json
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, Optional
import threading
import time
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import yfinance as yf

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP - CRITICAL FOR PATHWAY
# ═══════════════════════════════════════════════════════════════════════════

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
if not logger.handlers:
    logger.addHandler(console_handler)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE_MODEL_DIR = "models"
MODIFIED_MODEL_DIR = "modified_models"
OUTPUT_DIR = "outputs/lstm/config/lstm_predictions"

PREFILL_BUFFER = True
ENABLE_ONLINE_LEARNING = True

# FIXED: No duplicates, clear thresholds
RMSE_THRESHOLD = 5.0  # Lowered to trigger more easily
MIN_PREDICTIONS_BEFORE_RETRAIN = 50
RETRAIN_COOLDOWN = 15
RETRAIN_BATCH_SIZE = 30
RETRAIN_ITERATIONS = 5

ENHANCED_FEATURES = [
    'Close', 'Volume', 'Return', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_30', 'SMA_50',
    'RSI', 'MACD', 'MACD_Signal', 'BB_Middle', 'BB_Upper', 'BB_Lower',
    'Momentum', 'Momentum5', 'Volume_Ratio'
]

# ═══════════════════════════════════════════════════════════════════════════
# LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def compute_enhanced_indicators(df):
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

# ═══════════════════════════════════════════════════════════════════════════
# RETRAINING CHECKER FUNCTION - ADD THIS
# ═══════════════════════════════════════════════════════════════════════════

def check_retrain_conditions(predictor) -> Dict:
    """
    Separate function to check all retraining conditions.
    Returns dict with condition checks and blocking reasons.
    """
    conditions = {
        'enable_training': predictor.enable_training,
        'is_retraining': predictor.is_retraining,
        'prediction_count': predictor.prediction_count,
        'min_needed': MIN_PREDICTIONS_BEFORE_RETRAIN,
        'predictions_since_last': predictor.prediction_count - predictor.last_retrain_at,
        'cooldown_needed': RETRAIN_COOLDOWN,
        'buffer_size': len(predictor.retrain_buffer),
        'batch_size_needed': RETRAIN_BATCH_SIZE,
        'current_rmse': predictor.current_rmse,
        'rmse_threshold': RMSE_THRESHOLD,
        'should_retrain': False,
        'blocks': []
    }
    
    # Check each condition
    if not predictor.enable_training:
        conditions['blocks'].append("Training disabled")
        return conditions
    
    if predictor.is_retraining:
        conditions['blocks'].append("Already retraining")
        return conditions
    
    if predictor.prediction_count < MIN_PREDICTIONS_BEFORE_RETRAIN:
        conditions['blocks'].append(
            f"Need {MIN_PREDICTIONS_BEFORE_RETRAIN - predictor.prediction_count} more predictions"
        )
        return conditions
    
    predictions_since = predictor.prediction_count - predictor.last_retrain_at
    if predictions_since < RETRAIN_COOLDOWN:
        conditions['blocks'].append(
            f"Cooldown: {RETRAIN_COOLDOWN - predictions_since} predictions remaining"
        )
        return conditions
    
    if len(predictor.retrain_buffer) < RETRAIN_BATCH_SIZE:
        conditions['blocks'].append(
            f"Buffer: need {RETRAIN_BATCH_SIZE - len(predictor.retrain_buffer)} more samples"
        )
        return conditions
    
    # All conditions met, check RMSE
    if predictor.current_rmse > RMSE_THRESHOLD:
        conditions['should_retrain'] = True
        conditions['blocks'].append("✅ ALL CONDITIONS MET")
    else:
        conditions['blocks'].append(
            f"RMSE OK: {predictor.current_rmse:.4f} <= {RMSE_THRESHOLD}"
        )
    
    return conditions

# ═══════════════════════════════════════════════════════════════════════════
# REAL-TIME PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════

class RealtimeStockPredictor:
    def __init__(self, ticker: str, enable_training: bool = True, prefill_buffer: bool = True):
        self.ticker = ticker
        self.enable_training = enable_training
        self.prefill_buffer = prefill_buffer
        
        self.model = None
        self.scaler = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lookback = None
        self.input_size = None
        self.feature_columns = None
        self.buffer = deque()
        
        self.prediction_count = 0
        self.training_count = 0
        self.last_retrain_at = 0
        self.recent_errors = deque(maxlen=50)
        self.current_rmse = 0.0
        
        self.retrain_buffer = deque(maxlen=RETRAIN_BATCH_SIZE * 2)
        self.retraining_lock = threading.Lock()
        self.prediction_lock = threading.Lock()
        self.is_retraining = False
        
        self.using_modified_model = False
        self.modified_model_path = None
        
        self._load_model()
        
        if self.prefill_buffer:
            self._prefill_buffer_with_history()
    
    def _load_model(self):
        modified_path = f"{MODIFIED_MODEL_DIR}/{self.ticker}_lstm.pt"
        base_path = f"{BASE_MODEL_DIR}/{self.ticker}_lstm.pt"
        
        if os.path.exists(modified_path):
            model_path = modified_path
            self.using_modified_model = True
            self.modified_model_path = modified_path
            logger.info(f"Loading MODIFIED model for {self.ticker}")
        elif os.path.exists(base_path):
            model_path = base_path
            self.using_modified_model = False
            logger.info(f"Loading BASE model for {self.ticker}")
        else:
            raise FileNotFoundError(f"No model found for {self.ticker}")
        
        scaler_path = f"{BASE_MODEL_DIR}/{self.ticker}_scaler.pkl"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.input_size = checkpoint['input_size']
        self.lookback = checkpoint['lookback']
        self.feature_columns = checkpoint.get('feature_columns', None)
        
        self.model = StockLSTM(
            input_size=self.input_size,
            hidden_size=checkpoint.get('hidden_size', 64),
            num_layers=checkpoint.get('num_layers', 2),
            dropout=checkpoint.get('dropout', 0.2)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        
        if self.enable_training:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        self.buffer = deque(maxlen=self.lookback)
        logger.info(f"Loaded model for {self.ticker}: input_size={self.input_size}, lookback={self.lookback}")
    
    def _compute_indicators(self, df):
        return compute_enhanced_indicators(df)
    
    def _prefill_buffer_with_history(self):
        try:
            logger.info(f"Pre-filling buffer for {self.ticker}...")
            
            df = yf.download(
                self.ticker, 
                period=f"{self.lookback + 50}d",
                progress=False, 
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"Could not download history for {self.ticker}")
                return
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df = self._compute_indicators(df)
            
            if self.feature_columns is None:
                feature_names = ENHANCED_FEATURES
            else:
                feature_names = self.feature_columns
            
            df = df[feature_names].copy()
            df = df.dropna()
            recent_data = df.tail(self.lookback)
            
            for _, row in recent_data.iterrows():
                features = row.values.reshape(1, -1)
                features_scaled = self.scaler.transform(features)[0]
                self.buffer.append(features_scaled)
            
            logger.info(f"Buffer pre-filled for {self.ticker}: {len(self.buffer)}/{self.lookback}")
            
        except Exception as e:
            logger.error(f"Buffer pre-fill failed for {self.ticker}: {e}")
    
    def _extract_features(self, data_point: Dict) -> Optional[np.ndarray]:
        if self.feature_columns is None:
            feature_names = ENHANCED_FEATURES
        else:
            feature_names = self.feature_columns
        
        try:
            features = np.array([
                float(data_point.get(name, 0))
                for name in feature_names
            ])
            return features
        except Exception as e:
            logger.error(f"Feature extraction error for {self.ticker}: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # PREDICT METHOD - RETRAINING LOGIC IS HERE (LINE ~330)
    # ═══════════════════════════════════════════════════════════════════════
    
    def predict(self, data_point: Dict) -> Dict:
        """
        RETRAINING LOGIC HAPPENS AT THE END OF THIS METHOD (around line 380)
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
        
        with self.prediction_lock:
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
            'retrained': False,
            'using_modified_model': self.using_modified_model
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # RETRAINING LOGIC STARTS HERE - THIS IS THE KEY SECTION
        # ═══════════════════════════════════════════════════════════════════
        
        # Log every 10 predictions for debugging
        if self.prediction_count % 10 == 0:
            conditions = check_retrain_conditions(self)
            logger.info(f"[{self.ticker}] Pred #{self.prediction_count}: "
                       f"RMSE={self.current_rmse:.4f}, "
                       f"Buffer={len(self.retrain_buffer)}, "
                       f"Training={self.training_count}, "
                       f"Status={conditions['blocks']}")
        
        # Check retraining conditions
        conditions = check_retrain_conditions(self)
        
        if conditions['should_retrain']:
            logger.info(f"🔥 [{self.ticker}] TRIGGERING RETRAINING at prediction #{self.prediction_count}")
            logger.info(f"   RMSE: {self.current_rmse:.4f} > {RMSE_THRESHOLD}")
            logger.info(f"   Buffer: {len(self.retrain_buffer)} samples")
            logger.info(f"   Predictions since last retrain: {conditions['predictions_since_last']}")
            
            # Trigger retraining in separate thread
            retrain_thread = threading.Thread(
                target=self._retrain_model_async,
                daemon=True,
                name=f"Retrain-{self.ticker}"
            )
            retrain_thread.start()
            result['retraining_triggered'] = True
            
            logger.info(f"   Thread started: {retrain_thread.name} (ID: {retrain_thread.ident})")
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # ASYNC RETRAINING METHOD - THIS RUNS IN SEPARATE THREAD
    # ═══════════════════════════════════════════════════════════════════════
    
    def _retrain_model_async(self):
        """
        THIS METHOD RUNS IN A SEPARATE THREAD.
        It performs the actual model retraining.
        """
        logger.info(f"🚀 [{self.ticker}] Retrain thread started (TID: {threading.current_thread().ident})")
        
        with self.retraining_lock:
            if self.is_retraining:
                logger.warning(f"[{self.ticker}] Already retraining, skipping")
                return
            self.is_retraining = True
        
        try:
                    # CRITICAL: Set training mode INSIDE the lock and keep it
            self.model.train()  # Move this BEFORE any other operations
            
            logger.info(f"🔄 [{self.ticker}] RETRAINING STARTED")
            logger.info(f"   RMSE: {self.current_rmse:.4f}")
            logger.info(f"   Buffer size: {len(self.retrain_buffer)}")
            
            # Prepare batch from retrain buffer
            sequences = []
            targets = []
            
            buffer_list = list(self.retrain_buffer)
            
            for i in range(self.lookback, len(buffer_list)):
                seq = [buffer_list[j]['features'] for j in range(i-self.lookback, i)]
                target = buffer_list[i]['features'][0]  # Close price
                sequences.append(seq)
                targets.append(target)
            
            logger.info(f"   Created {len(sequences)} training sequences")
            
            if len(sequences) < 5:
                logger.warning(f"[{self.ticker}] Not enough sequences: {len(sequences)} < 5")
                return
            
            sequences = torch.FloatTensor(sequences).to(self.device)
            targets = torch.FloatTensor(targets).unsqueeze(1).to(self.device)
            
            logger.info(f"   Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
            
            # Train for iterations
            self.model.train()
            losses = []
            
            for epoch in range(RETRAIN_ITERATIONS):
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                
                if (epoch + 1) % 2 == 0:
                    logger.info(f"     Epoch {epoch+1}/{RETRAIN_ITERATIONS}, Loss: {loss.item():.6f}")
            
            self.model.eval()
            self.training_count += 1
            self.last_retrain_at = self.prediction_count
            
            # Save modified model
            Path(MODIFIED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
            modified_path = f"{MODIFIED_MODEL_DIR}/{self.ticker}_lstm.pt"
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'input_size': self.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'dropout': 0.2,
                'lookback': self.lookback,
                'feature_columns': self.feature_columns or ENHANCED_FEATURES,
                'training_count': self.training_count,
                'rmse_before': self.current_rmse,
                'retrained_at': time.time()
            }
            
            torch.save(checkpoint, modified_path)
            self.using_modified_model = True
            self.modified_model_path = modified_path
            
            avg_loss = np.mean(losses)
            logger.info(f"✅ [{self.ticker}] RETRAINING COMPLETE!")
            logger.info(f"   Training count: {self.training_count}")
            logger.info(f"   Avg loss: {avg_loss:.6f}")
            logger.info(f"   Model saved: {modified_path}")
            
        except Exception as e:
            logger.error(f"❌ [{self.ticker}] RETRAINING FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.is_retraining = False
            logger.info(f"[{self.ticker}] Retraining flag reset to False")
    
    def get_status(self) -> Dict:
        return {
            'ticker': self.ticker,
            'prediction_count': self.prediction_count,
            'training_count': self.training_count,
            'buffer_size': len(self.buffer),
            'buffer_needed': self.lookback,
            'rmse': float(self.current_rmse),
            'retrain_buffer_size': len(self.retrain_buffer),
            'using_modified_model': self.using_modified_model,
            'is_retraining': self.is_retraining
        }

# ═══════════════════════════════════════════════════════════════════════════
# PREDICTOR MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class PredictorManager:
    def __init__(self, enable_training: bool = True, prefill_buffers: bool = True):
        self.predictors: Dict[str, RealtimeStockPredictor] = {}
        self.enable_training = enable_training
        self.prefill_buffers = prefill_buffers
    
    def get_predictor(self, ticker: str) -> Optional[RealtimeStockPredictor]:
        if ticker not in self.predictors:
            try:
                self.predictors[ticker] = RealtimeStockPredictor(
                    ticker,
                    enable_training=self.enable_training,
                    prefill_buffer=self.prefill_buffers
                )
            except Exception as e:
                logger.error(f"Failed to load predictor for {ticker}: {e}")
                return None
        
        return self.predictors[ticker]
    
    def predict(self, ticker: str, data_point: Dict) -> Dict:
        predictor = self.get_predictor(ticker)
        if predictor is None:
            return {
                'ticker': ticker,
                'ready': False,
                'error': 'Predictor not available'
            }
        
        return predictor.predict(data_point)
    
    def get_all_statuses(self) -> Dict:
        return {
            ticker: predictor.get_status()
            for ticker, predictor in self.predictors.items()
        }

# Global manager instance
_manager = None

def initialize_manager(enable_training: bool = ENABLE_ONLINE_LEARNING, 
                      prefill_buffers: bool = PREFILL_BUFFER):
    global _manager
    _manager = PredictorManager(
        enable_training=enable_training,
        prefill_buffers=prefill_buffers
    )
    logger.info("✅ LSTM Predictor Manager initialized (ENHANCED with LOGGING)")

def predict_stock(ticker: str, data_point: Dict) -> Dict:
    """Main prediction function - call this from Pathway"""
    global _manager
    if _manager is None:
        initialize_manager()
    
    return _manager.predict(ticker, data_point)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("="*70)
    logger.info("🚀 LSTM REAL-TIME PREDICTION - ENHANCED WITH LOGGING")
    logger.info("="*70)
    logger.info("Enhancements:")
    logger.info("  ✅ Proper logging for Pathway compatibility")
    logger.info("  ✅ Fixed retraining triggers")
    logger.info("  ✅ Detailed condition checking")
    logger.info("  ✅ Separate retraining thread")

if __name__ == "__main__":
    main()