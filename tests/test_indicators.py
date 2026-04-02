"""Test script for indicators module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.strategy import (
    add_sma,
    add_ema,
    add_vwap,
    add_rsi,
    add_atr,
    add_all_indicators,
    get_trend,
    get_rsi_zone,
    is_rsi_exiting_oversold,
    is_price_crossing_vwap_up,
)


def create_sample_ohlcv(n: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price movement
    base_price = 50000
    returns = np.random.randn(n) * 0.02  # 2% daily volatility
    close = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = np.roll(close, 1)
    open_[0] = base_price
    
    # Volume
    volume = np.random.randint(1000, 10000, n).astype(float)
    
    # Timestamps
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min")
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def main():
    print("=" * 50)
    print("Indicators Module Test")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_ohlcv(100)
    print(f"\n✓ Created sample OHLCV data: {len(df)} candles")
    print(f"  Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    
    # Test SMA
    print("\n1. Testing add_sma(period=50)...")
    df_sma = add_sma(df, period=50)
    print(f"   ✓ Added column 'sma_50'")
    print(f"   Latest SMA: ${df_sma['sma_50'].iloc[-1]:,.2f}")
    
    # Test EMA
    print("\n2. Testing add_ema(period=20)...")
    df_ema = add_ema(df, period=20)
    print(f"   ✓ Added column 'ema_20'")
    print(f"   Latest EMA: ${df_ema['ema_20'].iloc[-1]:,.2f}")
    
    # Test RSI
    print("\n3. Testing add_rsi(period=14)...")
    df_rsi = add_rsi(df, period=14)
    print(f"   ✓ Added column 'rsi_14'")
    print(f"   Latest RSI: {df_rsi['rsi_14'].iloc[-1]:.2f}")
    
    # Test VWAP
    print("\n4. Testing add_vwap()...")
    df_vwap = add_vwap(df)
    print(f"   ✓ Added column 'vwap'")
    print(f"   Latest VWAP: ${df_vwap['vwap'].iloc[-1]:,.2f}")
    
    # Test ATR
    print("\n5. Testing add_atr(period=14)...")
    df_atr = add_atr(df, period=14)
    print(f"   ✓ Added column 'atr_14'")
    print(f"   Latest ATR: ${df_atr['atr_14'].iloc[-1]:,.2f}")
    
    # Test all indicators
    print("\n6. Testing add_all_indicators()...")
    df_all = add_all_indicators(df, sma_period=50, rsi_period=14, include_vwap=True, include_atr=True)
    print(f"   ✓ Added columns: {[c for c in df_all.columns if c not in df.columns]}")
    
    # Test trend detection
    print("\n7. Testing get_trend()...")
    trend = get_trend(df_all, ma_column="sma_50")
    print(f"   ✓ Current trend: {trend.upper()}")
    print(f"   (Close: ${df_all['close'].iloc[-1]:,.2f} vs SMA: ${df_all['sma_50'].iloc[-1]:,.2f})")
    
    # Test RSI zone
    print("\n8. Testing get_rsi_zone()...")
    rsi_zone = get_rsi_zone(df_all)
    print(f"   ✓ RSI zone: {rsi_zone.upper()}")
    print(f"   (RSI: {df_all['rsi_14'].iloc[-1]:.2f})")
    
    # Test signal functions
    print("\n9. Testing signal detection functions...")
    exiting_oversold = is_rsi_exiting_oversold(df_all)
    crossing_vwap_up = is_price_crossing_vwap_up(df_all)
    print(f"   RSI exiting oversold: {exiting_oversold}")
    print(f"   Price crossing VWAP up: {crossing_vwap_up}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
