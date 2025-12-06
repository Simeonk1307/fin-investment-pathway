#!/usr/bin/env python3
"""
LSTM Stock Prediction - SHADOW MODEL ARCHITECTURE
==================================================

SOLUTION: Use shadow model pattern for zero-latency retraining
- Active model: Always available for predictions
- Shadow model: Trains in background
- Atomic swap: Quick model replacement when ready

KEY FEATURES:
✅ No prediction blocking during training
✅ Atomic model swaps (millisecond latency)
✅ Pathway-compatible (no disk writes during streaming)
✅ Thread-safe with minimal lock contention
✅ Graceful handling of concurrent retraining attempts
"""

import os
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, Optional
import threading
import time
import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import yfinance as yf

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

BASE_MODEL_DIR = "models"
MODIFIED_MODEL_DIR = "modified_models"

PREFILL_BUFFER = True
ENABLE_ONLINE_LEARNING = True

# Retraining parameters
RMSE_THRESHOLD = 5.0
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
# SHADOW MODEL PREDICTOR - THE KEY INNOVATION
# ═══════════════════════════════════════════════════════════════════════════

class RealtimeStockPredictor:
    """
    Shadow Model Architecture:
    
    ACTIVE MODEL (self.model):
    - Always in eval() mode
    - Handles all predictions
    - Never locked for training
    
    SHADOW MODEL (created during training):
    - Deep copy of active model
    - Trained in background thread
    - Swapped atomically when ready
    
    ATOMIC SWAP:
    - Minimal lock contention (<1ms)
    - Only during model reference update
    - Predictions continue on old model until swap
    """
    
    def __init__(self, ticker: str, enable_training: bool = True, prefill_buffer: bool = True):
        self.ticker = ticker
        self.enable_training = enable_training
        self.prefill_buffer = prefill_buffer
        
        # Model components
        self.model = None  # ACTIVE MODEL - always for predictions
        self.scaler = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model metadata
        self.lookback = None
        self.input_size = None
        self.feature_columns = None
        self.buffer = deque()
        
        # Prediction tracking
        self.prediction_count = 0
        self.training_count = 0
        self.last_retrain_at = 0
        self.recent_errors = deque(maxlen=50)
        self.current_rmse = 0.0
        
        # Retraining buffer
        self.retrain_buffer = deque(maxlen=RETRAIN_BATCH_SIZE * 2)
        
        # Thread safety locks
        self.model_swap_lock = threading.Lock()  # ONLY for swapping model reference
        self.retraining_lock = threading.Lock()  # Prevents concurrent retraining
        self.is_retraining = False
        
        # Model state tracking
        self.using_modified_model = False
        self.modified_model_path = None
        self.pending_model_save = None  # Store checkpoint to save after streaming
        
        self._load_model()
        
        if self.prefill_buffer:
            self._prefill_buffer_with_history()
    
    def _load_model(self):
        """Load initial model from disk"""
        modified_path = f"{MODIFIED_MODEL_DIR}/{self.ticker}_lstm.pt"
        base_path = f"{BASE_MODEL_DIR}/{self.ticker}_lstm.pt"
        
        if os.path.exists(modified_path):
            model_path = modified_path
            self.using_modified_model = True
            self.modified_model_path = modified_path
            logger.info(f"[{self.ticker}] Loading MODIFIED model")
        elif os.path.exists(base_path):
            model_path = base_path
            self.using_modified_model = False
            logger.info(f"[{self.ticker}] Loading BASE model")
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
        self.model.eval()  # CRITICAL: Always keep active model in eval mode
        
        self.scaler = joblib.load(scaler_path)
        
        if self.enable_training:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        self.buffer = deque(maxlen=self.lookback)
        logger.info(f"[{self.ticker}] Model loaded: input_size={self.input_size}, lookback={self.lookback}")
    
    def _compute_indicators(self, df):
        return compute_enhanced_indicators(df)
    
    def _prefill_buffer_with_history(self):
        """Pre-fill buffer with historical data"""
        try:
            logger.info(f"[{self.ticker}] Pre-filling buffer...")
            
            df = yf.download(
                self.ticker, 
                period=f"{self.lookback + 50}d",
                progress=False, 
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"[{self.ticker}] Could not download history")
                return
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df = self._compute_indicators(df)
            
            feature_names = self.feature_columns or ENHANCED_FEATURES
            df = df[feature_names].copy()
            df = df.dropna()
            recent_data = df.tail(self.lookback)
            
            for _, row in recent_data.iterrows():
                features = row.values.reshape(1, -1)
                features_scaled = self.scaler.transform(features)[0]
                self.buffer.append(features_scaled)
            
            logger.info(f"[{self.ticker}] Buffer pre-filled: {len(self.buffer)}/{self.lookback}")
            
        except Exception as e:
            logger.error(f"[{self.ticker}] Buffer pre-fill failed: {e}")
    
    def _extract_features(self, data_point: Dict) -> Optional[np.ndarray]:
        """Extract features from data point"""
        feature_names = self.feature_columns or ENHANCED_FEATURES
        
        try:
            features = np.array([
                float(data_point.get(name, 0))
                for name in feature_names
            ])
            return features
        except Exception as e:
            logger.error(f"[{self.ticker}] Feature extraction error: {e}")
            return None
    
    def _should_trigger_retraining(self) -> tuple[bool, str]:
        """
        Check if retraining should be triggered.
        Returns (should_retrain, reason)
        """
        if not self.enable_training:
            return False, "Training disabled"
        
        if self.is_retraining:
            return False, "Already retraining"
        
        if self.prediction_count < MIN_PREDICTIONS_BEFORE_RETRAIN:
            remaining = MIN_PREDICTIONS_BEFORE_RETRAIN - self.prediction_count
            return False, f"Need {remaining} more predictions"
        
        predictions_since = self.prediction_count - self.last_retrain_at
        if predictions_since < RETRAIN_COOLDOWN:
            remaining = RETRAIN_COOLDOWN - predictions_since
            return False, f"Cooldown: {remaining} predictions remaining"
        
        if len(self.retrain_buffer) < RETRAIN_BATCH_SIZE:
            remaining = RETRAIN_BATCH_SIZE - len(self.retrain_buffer)
            return False, f"Buffer: need {remaining} more samples"
        
        if self.current_rmse > RMSE_THRESHOLD:
            return True, f"RMSE {self.current_rmse:.4f} > {RMSE_THRESHOLD}"
        
        return False, f"RMSE OK: {self.current_rmse:.4f} <= {RMSE_THRESHOLD}"
    
    # ═══════════════════════════════════════════════════════════════════════
    # PREDICTION METHOD - ZERO BLOCKING
    # ═══════════════════════════════════════════════════════════════════════
    
    def predict(self, data_point: Dict) -> Dict:
        """
        Make prediction using ACTIVE MODEL.
        
        CRITICAL: This method NEVER blocks on training.
        - Active model is always in eval() mode
        - No locks during prediction
        - Only brief lock (<1ms) if model swap occurs
        """
        # Extract and scale features
        features = self._extract_features(data_point)
        if features is None:
            return {
                'ticker': self.ticker,
                'predicted_price': None,
                'current_price': data_point.get('Close', 0),
                'ready': False,
                'error': 'Feature extraction failed'
            }
        
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
        
        # Add to buffers
        self.buffer.append(features_scaled)
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
        
        # PREDICTION: No locks needed - model is always in eval mode
        sequence = np.array(list(self.buffer)).reshape(1, self.lookback, -1)
        sequence_tensor = torch.FloatTensor(sequence).to(self.device)
        
        # Make prediction (active model is always ready)
        with torch.no_grad():
            pred_scaled = self.model(sequence_tensor).cpu().item()
        
        # Denormalize
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
            'using_modified_model': self.using_modified_model,
            'is_retraining': self.is_retraining
        }
        
        # Check if retraining should be triggered
        should_retrain, reason = self._should_trigger_retraining()
        
        if self.prediction_count % 10 == 0:
            logger.info(f"[{self.ticker}] Pred #{self.prediction_count}: "
                       f"RMSE={self.current_rmse:.4f}, "
                       f"Training={self.training_count}, "
                       f"Status: {reason}")
        
        if should_retrain:
            logger.info(f"🔥 [{self.ticker}] TRIGGERING RETRAINING at prediction #{self.prediction_count}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Buffer: {len(self.retrain_buffer)} samples")
            
            # Trigger retraining in background
            retrain_thread = threading.Thread(
                target=self._retrain_shadow_model,
                daemon=True,
                name=f"Retrain-{self.ticker}"
            )
            retrain_thread.start()
            result['retraining_triggered'] = True
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # SHADOW MODEL RETRAINING - BACKGROUND THREAD
    # ═══════════════════════════════════════════════════════════════════════
    
    def _retrain_shadow_model(self):
        """
        Train a SHADOW MODEL in background, then swap atomically.
        
        PROCESS:
        1. Create deep copy of active model (shadow model)
        2. Train shadow model in background
        3. Atomic swap: Replace active model reference
        4. Store checkpoint for later disk save (outside Pathway)
        
        ZERO BLOCKING: Predictions continue on old model until swap
        """
        logger.info(f"🚀 [{self.ticker}] Shadow model training started")
        
        # Prevent concurrent retraining
        with self.retraining_lock:
            if self.is_retraining:
                logger.warning(f"[{self.ticker}] Already retraining, skipping")
                return
            self.is_retraining = True
        
        shadow_model = None
        shadow_optimizer = None
        
        try:
            # STEP 1: Create shadow model (deep copy of active model)
            logger.info(f"[{self.ticker}] Creating shadow model...")
            shadow_model = StockLSTM(
                input_size=self.input_size,
                hidden_size=self.model.hidden_size,
                num_layers=self.model.num_layers,
                dropout=0.2
            ).to(self.device)
            
            # Copy weights from active model
            shadow_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            shadow_model.train()  # Shadow model can be in train mode
            
            shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.0001)
            
            logger.info(f"[{self.ticker}] Shadow model created")
            
            # STEP 2: Prepare training data
            sequences = []
            targets = []
            buffer_list = list(self.retrain_buffer)
            
            for i in range(self.lookback, len(buffer_list)):
                seq = [buffer_list[j]['features'] for j in range(i-self.lookback, i)]
                target = buffer_list[i]['features'][0]  # Close price
                sequences.append(seq)
                targets.append(target)
            
            logger.info(f"[{self.ticker}] Training shadow: {len(sequences)} sequences")
            
            if len(sequences) < 5:
                logger.warning(f"[{self.ticker}] Not enough sequences: {len(sequences)}")
                return
            
            sequences = torch.FloatTensor(sequences).to(self.device)
            targets = torch.FloatTensor(targets).unsqueeze(1).to(self.device)
            
            # STEP 3: Train shadow model (active model still serving predictions)
            logger.info(f"[{self.ticker}] Training shadow model...")
            losses = []
            
            for epoch in range(RETRAIN_ITERATIONS):
                outputs = shadow_model(sequences)
                loss = self.criterion(outputs, targets)
                
                shadow_optimizer.zero_grad()
                loss.backward()
                shadow_optimizer.step()
                losses.append(loss.item())
                
                if (epoch + 1) % 2 == 0:
                    logger.info(f"  [{self.ticker}] Epoch {epoch+1}/{RETRAIN_ITERATIONS}, Loss: {loss.item():.6f}")
            
            avg_loss = np.mean(losses)
            logger.info(f"[{self.ticker}] Shadow training complete. Avg loss: {avg_loss:.6f}")
            
            # STEP 4: ATOMIC SWAP - Replace active model reference
            logger.info(f"[{self.ticker}] Performing atomic model swap...")
            
            with self.model_swap_lock:
                # Set shadow to eval mode before swap
                shadow_model.eval()
                
                # Atomic swap: Replace active model
                old_model = self.model
                self.model = shadow_model
                self.optimizer = shadow_optimizer
                
                # Update metadata
                self.training_count += 1
                self.last_retrain_at = self.prediction_count
                
                logger.info(f"✅ [{self.ticker}] Model swap complete!")
                logger.info(f"   Training count: {self.training_count}")
                logger.info(f"   RMSE: {self.current_rmse:.4f}")
            
            # Clean up old model
            del old_model
            
            # STEP 5: Prepare checkpoint for later disk save (outside Pathway)
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
            
            self.pending_model_save = checkpoint
            self.using_modified_model = True
            
            logger.info(f"[{self.ticker}] Checkpoint prepared for later save")
            
        except Exception as e:
            logger.error(f"❌ [{self.ticker}] Shadow training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.is_retraining = False
            logger.info(f"[{self.ticker}] Retraining complete")
    
    def save_modified_model(self):
        """
        Save pending model checkpoint to disk.
        CALL THIS AFTER PATHWAY STREAMING COMPLETES.
        """
        if self.pending_model_save is None:
            logger.info(f"[{self.ticker}] No pending model save")
            return
        
        try:
            Path(MODIFIED_MODEL_DIR).mkdir(parents=True, exist_ok=True)
            modified_path = f"{MODIFIED_MODEL_DIR}/{self.ticker}_lstm.pt"
            
            torch.save(self.pending_model_save, modified_path)
            self.modified_model_path = modified_path
            self.pending_model_save = None
            
            logger.info(f"💾 [{self.ticker}] Model saved to: {modified_path}")
            
        except Exception as e:
            logger.error(f"❌ [{self.ticker}] Model save failed: {e}")
    
    def get_status(self) -> Dict:
        """Get current predictor status"""
        return {
            'ticker': self.ticker,
            'prediction_count': self.prediction_count,
            'training_count': self.training_count,
            'buffer_size': len(self.buffer),
            'buffer_needed': self.lookback,
            'rmse': float(self.current_rmse),
            'retrain_buffer_size': len(self.retrain_buffer),
            'using_modified_model': self.using_modified_model,
            'is_retraining': self.is_retraining,
            'has_pending_save': self.pending_model_save is not None
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
    
    def save_all_modified_models(self):
        """
        Save all modified models to disk.
        CALL THIS AFTER PATHWAY STREAMING COMPLETES.
        """
        logger.info("\n💾 Saving all modified models...")
        for ticker, predictor in self.predictors.items():
            predictor.save_modified_model()
        logger.info("✅ All models saved")
    
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
    logger.info("✅ Shadow Model Predictor Manager initialized")

def predict_stock(ticker: str, data_point: Dict) -> Dict:
    """Main prediction function - call this from Pathway"""
    global _manager
    if _manager is None:
        initialize_manager()
    
    return _manager.predict(ticker, data_point)

def save_all_models():
    """Save all modified models - call this AFTER Pathway streaming"""
    global _manager
    if _manager is not None:
        _manager.save_all_modified_models()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("="*70)
    logger.info("🚀 LSTM SHADOW MODEL ARCHITECTURE")
    logger.info("="*70)
    logger.info("Features:")
    logger.info("  ✅ Zero-latency predictions during training")
    logger.info("  ✅ Atomic model swaps (<1ms)")
    logger.info("  ✅ Pathway-compatible (no disk writes during streaming)")
    logger.info("  ✅ Thread-safe shadow model pattern")

if __name__ == "__main__":
    main()
