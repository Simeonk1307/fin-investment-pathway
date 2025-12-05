import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - EDIT THIS SECTION
# ═══════════════════════════════════════════════════════════════════════════

# Stocks to train
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
try:
    TICKERS = eval(os.getenv("TICKERS", "[]"))
    print(f"Raw TICKERS from env: {TICKERS,type(TICKERS)}")
    input("Press enter to continue... ")
except Exception:
    raise ValueError("Please set TICKERS environment variable as comma-separated list, e.g., 'AAPL,MSFT,GOOGL'")
# TICKERS = os.getenv("TICKERS",TICKERS)

# Data parameters
LOOKBACK_PERIOD = 20        # Past timesteps for prediction
TRAIN_YEARS = "4y"          # Historical data to download
TEST_SPLIT = 0.2            # Test set fraction

# Model architecture
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Training parameters
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 5                # Early stopping patience

# Output paths
MODEL_DIR = "models"
RESULTS_DIR = "outputs/lstm/training"

# FEATURE CONFIGURATION
# CRITICAL: First feature MUST be 'Close' (target variable)
# Add/remove features here - they'll be automatically used
FEATURE_COLUMNS = [
    'Close',           # TARGET - Must be first!
    'Volume',
    'SMA_5',
    'SMA_20',
    'SMA_50',
    'RSI',
    'MACD',
    'MACD_Signal',
    'BB_Middle',
    'Momentum',
    'Volume_Ratio'
]

# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def compute_technical_indicators(df):
    """
    Compute technical indicators for stock data.
    
    TO ADD NEW INDICATORS:
    1. Add calculation here
    2. Add column name to FEATURE_COLUMNS above
    3. Retrain - that's it!
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
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
    
    # Fill NaN values
    df = df.ffill().bfill()
    
    return df

def download_stock_data(ticker, period=TRAIN_YEARS):
    """Download and prepare stock data with technical indicators"""
    print(f"  📥 Downloading {ticker} data...")
    
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    
    # Compute indicators
    df = compute_technical_indicators(df)
    
    # Select features
    df = df[FEATURE_COLUMNS].copy()
    df = df.dropna()
    
    print(f"  ✅ Downloaded {len(df)} data points")
    
    return df

# ═══════════════════════════════════════════════════════════════════════════
# PYTORCH COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences"""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(data, lookback):
    """Create LSTM sequences from time series data"""
    sequences = []
    targets = []
    
    for i in range(lookback, len(data)):
        seq = data[i-lookback:i]
        target = data[i, 0]  # Close price (first column)
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


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
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def train_model(ticker):
    """Train LSTM model for a single ticker"""
    
    print(f"\n{'='*70}")
    print(f"🎯 Training: {ticker}")
    print(f"{'='*70}")
    
    # Download data
    df = download_stock_data(ticker)
    data = df.values
    
    # Split train/test
    train_size = int(len(data) * (1 - TEST_SPLIT))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, LOOKBACK_PERIOD)
    X_test, y_test = create_sequences(test_scaled, LOOKBACK_PERIOD)
    
    print(f"  📊 Train samples: {len(X_train)}")
    print(f"  📊 Test samples: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_train.shape[2]
    
    model = StockLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"  🔥 Training on {device}...")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(MODEL_DIR, exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_size': input_size,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'lookback': LOOKBACK_PERIOD,
                'feature_columns': FEATURE_COLUMNS,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'epoch': epoch,
                'scaler_mean': scaler.data_min_.tolist(),
                'scaler_scale': scaler.scale_.tolist()
            }
            torch.save(checkpoint, f"{MODEL_DIR}/{ticker}_lstm.pt")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.6f}, Test: {test_loss:.6f}")
        
        if patience_counter >= PATIENCE:
            print(f"  ⏸️  Early stopping at epoch {epoch+1}")
            break
    
    # Save scaler separately
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")
    
    print(f"  ✅ Model saved: {MODEL_DIR}/{ticker}_lstm.pt")
    print(f"  ✅ Scaler saved: {MODEL_DIR}/{ticker}_scaler.pkl")
    print(f"  ✅ Best test loss: {best_loss:.6f}")
    
    return {
        'ticker': ticker,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'best_test_loss': best_loss,
        'input_size': input_size,
        'lookback': LOOKBACK_PERIOD
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("🚀 LSTM STOCK PREDICTION - TRAINING PIPELINE")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  Tickers: {TICKERS}")
    print(f"  Lookback: {LOOKBACK_PERIOD} days")
    print(f"  Training data: {TRAIN_YEARS}")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Create directories
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Train all tickers
    results = []
    failed = []
    
    for ticker in TICKERS:
        try:
            result = train_model(ticker)
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ Error training {ticker}: {e}")
            failed.append(ticker)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'lookback_period': LOOKBACK_PERIOD,
            'train_years': TRAIN_YEARS,
            'test_split': TEST_SPLIT,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'features': FEATURE_COLUMNS
        },
        'results': results,
        'failed': failed
    }
    
    with open(f"{RESULTS_DIR}/training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Models: {MODEL_DIR}/")
    print(f"📁 Summary: {RESULTS_DIR}/training_summary.json")
    print(f"✅ Trained: {len(results)} models")
    if failed:
        print(f"❌ Failed: {len(failed)} models - {', '.join(failed)}")
    
    print("\n🎯 Next Steps:")
    print("  → Run: python test_lstm_stocks.py")
    print("  → Models will auto-load for real-time prediction")


if __name__ == "__main__":
    main()