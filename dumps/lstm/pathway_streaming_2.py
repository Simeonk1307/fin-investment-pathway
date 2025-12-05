#!/usr/bin/env python3
"""
LSTM Stock Prediction - Pathway Streaming Pipeline
===================================================

Real-time stock prediction using Pathway streaming with LSTM models.
Automatically loads historical data and streams predictions.

IMPORTANT: Model files in models/ directory are READ-ONLY.
Online retraining updates models IN-MEMORY only, not on disk.
This is because Pathway streaming cannot write to model files during execution.

FEATURES:
- Downloads 1 year historical data for all tickers
- Computes technical indicators
- Time-aligned multi-ticker streaming
- Real-time LSTM predictions with auto buffer fill
- Online retraining (in-memory only, not saved to disk)
- Pathway-compatible streaming pipeline

USAGE:
    python pathway_stream.py

NO ARGUMENTS NEEDED - Everything configured below.

Author: Stock Prediction Team
"""

import os
from venv import logger
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

# Import our LSTM predictor
from dumps.lstm.LSTM_TEST_TEST import initialize_manager, predict_stock

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False
    print("⚠️  Pathway not installed. Install with: pip install pathway")

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Tickers to stream
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Data parameters
HISTORICAL_PERIOD = "1y"  # Download 1 year of historical data
DATA_DIR = "data"
OUTPUT_DIR = "outputs/pathway_predictions"

# Streaming simulation
STREAM_DELAY_MS = 100  # Delay between streaming batches (milliseconds)

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_technical_indicators(df):
    """
    Compute technical indicators for stock data.
    Same as training script to ensure consistency.
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


def download_ticker_data(ticker: str, period: str = HISTORICAL_PERIOD) -> pd.DataFrame:
    """Download historical data for a single ticker with indicators"""
    print(f"  📥 Downloading {ticker}...")
    
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.reset_index()
    
    # Compute indicators
    df = compute_technical_indicators(df)
    
    # Add ticker column
    df['ticker'] = ticker
    
    # Select required columns
    required_cols = [
        'Date', 'ticker', 'Close', 'Volume',
        'SMA_5', 'SMA_20', 'SMA_50', 'RSI',
        'MACD', 'MACD_Signal', 'BB_Middle',
        'Momentum', 'Volume_Ratio'
    ]
    
    df = df[required_cols].copy()
    df = df.dropna()
    
    print(f"  ✅ {ticker}: {len(df)} data points")
    
    return df


def load_all_tickers_data(tickers: List[str], period: str = HISTORICAL_PERIOD) -> pd.DataFrame:
    """
    Download and combine data for all tickers.
    Returns a single DataFrame with all tickers aligned by timestamp.
    """
    print("\n" + "="*70)
    print("📊 DOWNLOADING HISTORICAL DATA FOR ALL TICKERS")
    print("="*70)
    
    all_data = []
    
    for ticker in tickers:
        try:
            df = download_ticker_data(ticker, period)
            all_data.append(df)
        except Exception as e:
            print(f"  ❌ Error downloading {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No data downloaded for any ticker")
    
    # Combine all ticker data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and ticker
    combined_df = combined_df.sort_values(['Date', 'ticker']).reset_index(drop=True)
    
    # Save to CSV
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    csv_path = f"{DATA_DIR}/stocks_historical.csv"
    combined_df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Combined data: {len(combined_df)} rows")
    print(f"📁 Saved to: {csv_path}")
    print(f"📅 Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    print(f"🏢 Tickers: {', '.join(combined_df['ticker'].unique())}")
    
    return combined_df


def create_streaming_data_with_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create streaming-ready data with proper time alignment.
    All tickers for a given date get the same timestamp for live join.
    
    IMPORTANT: This creates a static CSV for Pathway to stream.
    The CSV is read-only during streaming.
    """
    print("\n🔄 Preparing streaming data...")
    
    # Get unique dates
    unique_dates = sorted(df['Date'].unique())
    
    # Assign monotonic time values for streaming
    streaming_data = []
    
    for time_idx, date in enumerate(unique_dates, start=1):
        date_data = df[df['Date'] == date].copy()
        date_data['time'] = time_idx  # All tickers at same date get same time
        streaming_data.append(date_data)
    
    streaming_df = pd.concat(streaming_data, ignore_index=True)
    
    # Save streaming data
    streaming_path = f"{DATA_DIR}/stocks_streaming.csv"
    streaming_df.to_csv(streaming_path, index=False)
    
    print(f"✅ Streaming data ready: {len(streaming_df)} rows")
    print(f"📁 Saved to: {streaming_path}")
    print(f"⏱️  Time steps: {len(unique_dates)}")
    
    return streaming_df


# ═══════════════════════════════════════════════════════════════════════════
# PATHWAY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def run_pathway_streaming_pipeline():
    """
    Main Pathway streaming pipeline.
    
    CRITICAL: Models are updated IN-MEMORY only during streaming.
    Model files in models/ directory remain unchanged.
    This is by design - Pathway streaming cannot write to disk during execution.
    """
    if not PATHWAY_AVAILABLE:
        print("❌ Pathway not available. Install with: pip install pathway")
        return
    
    print("\n" + "="*70)
    print("🚀 PATHWAY STREAMING PIPELINE - LSTM PREDICTIONS")
    print("="*70)
    print("\n⚠️  IMPORTANT: Model updates happen IN-MEMORY only")
    print("   Model files in models/ directory are READ-ONLY during streaming")
    
    # Initialize LSTM predictor manager with buffer pre-fill
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    print("\n📦 Initializing LSTM models...")
    initialize_manager(enable_training=True, prefill_buffers=True)
    
    # Check if streaming data exists
    streaming_path = f"{DATA_DIR}/stocks_streaming.csv"
    # if not os.path.exists(streaming_path):
    #     print(f"\n❌ Streaming data not found: {streaming_path}")
    #     print("Run load_all_tickers_data() first!")
    #     return
    
    print("\n📖 Setting up Pathway streaming source...")
    input("Press Enter to continue...")
    logger.info(f"Setting up Pathway streaming source from: {streaming_path}")
    import time
    time.sleep(1)
    
    # Define schema for Pathway
    class StockSchema(pw.Schema):
        Date: str
        ticker: str
        Close: float
        Volume: float
        SMA_5: float
        SMA_20: float
        SMA_50: float
        RSI: float
        MACD: float
        MACD_Signal: float
        BB_Middle: float
        Momentum: float
        Volume_Ratio: float
        time: int
    
    # Read CSV with streaming simulation
    stocks_table = pw.io.csv.read(
        streaming_path,
        schema=StockSchema,
        mode="static",
        autocommit_duration_ms=STREAM_DELAY_MS
    )
    
    print("✅ Pathway source configured")
    
    # Define prediction UDF
    @pw.udf
    def make_prediction(
        ticker: str,
        close: float,
        volume: float,
        sma_5: float,
        sma_20: float,
        sma_50: float,
        rsi: float,
        macd: float,
        macd_signal: float,
        bb_middle: float,
        momentum: float,
        volume_ratio: float
    ) -> pw.Json:
        """
        Pathway UDF that calls our LSTM predictor.
        
        NOTE: Any model retraining happens in-memory.
        The model state is updated for the current session only.
        Original model files in models/ directory remain unchanged.
        """
        data_point = {
            'Close': close,
            'Volume': volume,
            'SMA_5': sma_5,
            'SMA_20': sma_20,
            'SMA_50': sma_50,
            'RSI': rsi,
            'MACD': macd,
            'MACD_Signal': macd_signal,
            'BB_Middle': bb_middle,
            'Momentum': momentum,
            'Volume_Ratio': volume_ratio
        }
        logger.info(f"Making prediction for {ticker} with data: {data_point}")
        result = predict_stock(ticker, data_point)
        logger.info(f"Prediction for {ticker}: {result}")
        return result
    
    # Make predictions
    print("🔮 Making predictions...")
    
    predictions_table = stocks_table.select(
        date=pw.this.Date,
        ticker=pw.this.ticker,
        current_price=pw.this.Close,
        prediction_result=make_prediction(
            pw.this.ticker,
            pw.this.Close,
            pw.this.Volume,
            pw.this.SMA_5,
            pw.this.SMA_20,
            pw.this.SMA_50,
            pw.this.RSI,
            pw.this.MACD,
            pw.this.MACD_Signal,
            pw.this.BB_Middle,
            pw.this.Momentum,
            pw.this.Volume_Ratio
        )
    )
    
    # Extract results from JSON
    @pw.udf
    def get_json_value(data: pw.Json, key: str, default=None):
        """Extract value from JSON result"""
        try:
            val = data[key]
            return val if val is not None else default
        except:
            return default
    @pw.udf
    def get_json_value_boolean(data: pw.Json, key: str, default=False)->bool:
        """Extract boolean value from JSON result"""
        try:
            val = data[key]
            return bool(val)
        except:
            return default
    
    # Create final results table
    results_table = predictions_table.select(
        date=pw.this.date,
        ticker=pw.this.ticker,
        current_price=pw.this.current_price,
        predicted_price=get_json_value(pw.this.prediction_result, "predicted_price", 0.0),
        ready=get_json_value_boolean(pw.this.prediction_result, "ready", False),
        rmse=get_json_value(pw.this.prediction_result, "rmse", 0.0),
        prediction_count=get_json_value(pw.this.prediction_result, "prediction_count", 0),
        training_count=get_json_value(pw.this.prediction_result, "training_count", 0),
        retrained=get_json_value_boolean(pw.this.prediction_result, "retrained", False),
        buffer_size=get_json_value(pw.this.prediction_result, "buffer_size", 0),
        message=get_json_value(pw.this.prediction_result, "message", "")
    )
    pw.io.csv.write(results_table, f"{OUTPUT_DIR}/all_predictions.csv")
    # Filter only ready 
    @pw.udf 
    def print_checkpoint(ready:bool)->bool:
        if ready:
            logger.info("Checkpoint reached: A prediction is ready.")
            import time
            time.sleep(0.5)
        return ready
    ready_predictions = results_table.filter(pw.this.ready)
    # checkpoint = ready_predictions.select(
    #     ready=print_checkpoint(pw.this.ready)
    # )
    # Output results
    
    # pw.io.jsonlines.write(ready_predictions, f"{OUTPUT_DIR}/predictions.jsonl")
    pw.io.csv.write(ready_predictions, f"{OUTPUT_DIR}/predictions.csv")
    
    print("\n" + "="*70)
    print("✅ PATHWAY PIPELINE CONFIGURED")
    print("="*70)
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"📊 Predictions will be saved to:")
    print(f"   - {OUTPUT_DIR}/predictions.jsonl")
    print(f"   - {OUTPUT_DIR}/predictions.csv")
    print("\n⚠️  Remember: Model updates are IN-MEMORY only")
    print("   Original models in models/ remain unchanged")
    print("\n🚀 Starting Pathway execution...")
    print("="*70 + "\n")
    
    # Run Pathway
    pw.run()
    
    print("\n" + "="*70)
    print("✅ STREAMING COMPLETE!")
    print("="*70)


# ═══════════════════════════════════════════════════════════════════════════
# NON-PATHWAY DEMO (Fallback)
# ═══════════════════════════════════════════════════════════════════════════

def run_demo_without_pathway():
    """
    Run demo without Pathway using pandas iteration.
    Simulates the same streaming behavior.
    """
    print("\n" + "="*70)
    print("🚀 DEMO MODE - STREAMING WITHOUT PATHWAY")
    print("="*70)
    
    # Initialize models with buffer pre-fill
    print("\n📦 Initializing LSTM models...")
    initialize_manager(enable_training=True, prefill_buffers=True)
    
    # Load streaming data
    streaming_path = f"{DATA_DIR}/stocks_streaming.csv"
    if not os.path.exists(streaming_path):
        print(f"❌ Streaming data not found: {streaming_path}")
        return
    
    df = pd.read_csv(streaming_path)
    
    print(f"\n📊 Loaded {len(df)} data points")
    print(f"🏢 Tickers: {', '.join(df['ticker'].unique())}")
    print(f"⏱️  Time steps: {df['time'].nunique()}")
    
    # Process data by time step (simulating streaming)
    predictions = []
    
    print("\n🔄 Processing streaming data...\n")
    
    for time_step in sorted(df['time'].unique()):
        time_data = df[df['time'] == time_step]
        
        print(f"⏱️  Time step {time_step} ({time_data.iloc[0]['Date']}) - {len(time_data)} tickers")
        
        for _, row in time_data.iterrows():
            data_point = {
                'Close': row['Close'],
                'Volume': row['Volume'],
                'SMA_5': row['SMA_5'],
                'SMA_20': row['SMA_20'],
                'SMA_50': row['SMA_50'],
                'RSI': row['RSI'],
                'MACD': row['MACD'],
                'MACD_Signal': row['MACD_Signal'],
                'BB_Middle': row['BB_Middle'],
                'Momentum': row['Momentum'],
                'Volume_Ratio': row['Volume_Ratio']
            }
            
            result = predict_stock(row['ticker'], data_point)
            
            if result['ready']:
                result['date'] = row['Date']
                result['time_step'] = time_step
                predictions.append(result)
                
                print(f"  ✅ [{result['ticker']}] Pred #{result['prediction_count']}: "
                      f"${result['predicted_price']:.2f} (Actual: ${result['current_price']:.2f}) "
                      f"RMSE: {result['rmse']:.4f}"
                      f"{' 🔄 RETRAINED' if result.get('retrained') else ''}")
    
    # Save predictions
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        df_predictions.to_csv(f"{OUTPUT_DIR}/predictions_demo.csv", index=False)
        
        # import json
        # with open(f"{OUTPUT_DIR}/predictions_demo.json", 'w') as f:
        #     json.dump(predictions, f, indent=2)
        
        print("\n" + "="*70)
        print(f"✅ DEMO COMPLETE! Generated {len(predictions)} predictions")
        print("="*70)
        print(f"📁 Saved to:")
        print(f"   - {OUTPUT_DIR}/predictions_demo.csv")
        print(f"   - {OUTPUT_DIR}/predictions_demo.json")
        print("="*70)
    else:
        print("\n⚠️  No predictions generated")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("📈 LSTM STOCK PREDICTION - PATHWAY STREAMING")
    print("="*70)
    
    # Step 1: Load historical data
    print("\n📋 Step 1: Loading historical data...")
    try:
        combined_df = load_all_tickers_data(TICKERS, HISTORICAL_PERIOD)
        streaming_df = create_streaming_data_with_timestamps(combined_df)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Run streaming pipeline
    print("\n📋 Step 2: Running streaming pipeline...")
    
    if PATHWAY_AVAILABLE:
    # if False:
        try:
            run_pathway_streaming_pipeline()
        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        except Exception as e:
            print(f"❌ Pathway error: {e}")
            import traceback
            traceback.print_exc()
            print("\n⚠️  Falling back to demo mode...")
            run_demo_without_pathway()
    else:
        print("\n⚠️  Pathway not available - running demo mode")
        run_demo_without_pathway()


if __name__ == "__main__":
    main()