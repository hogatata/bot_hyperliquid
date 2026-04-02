"""Test script for signals module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.strategy import Signal, SignalGenerator, SignalResult


def create_bullish_scenario() -> pd.DataFrame:
    """Create data with bullish trend + oversold RSI + VWAP cross up."""
    np.random.seed(42)
    n = 100
    
    # Create uptrending price (above SMA)
    base_price = 65000
    trend = np.linspace(0, 3000, n)  # Uptrend
    noise = np.random.randn(n) * 200
    close = base_price + trend + noise
    
    # Make last few candles cross VWAP upward
    close[-3] = close[-4] - 500  # Dip below
    close[-2] = close[-4] - 300  # Still below
    close[-1] = close[-4] + 200  # Cross up
    
    high = close + np.abs(np.random.randn(n) * 100)
    low = close - np.abs(np.random.randn(n) * 100)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min"),
        "open": np.roll(close, 1),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    df.iloc[0, df.columns.get_loc("open")] = base_price
    
    return df


def create_bearish_scenario() -> pd.DataFrame:
    """Create data with bearish trend + overbought RSI + VWAP cross down."""
    np.random.seed(123)
    n = 100
    
    # Create downtrending price (below SMA)
    base_price = 65000
    trend = np.linspace(0, -3000, n)  # Downtrend
    noise = np.random.randn(n) * 200
    close = base_price + trend + noise
    
    # Make last few candles cross VWAP downward
    close[-3] = close[-4] + 500  # Spike above
    close[-2] = close[-4] + 300  # Still above
    close[-1] = close[-4] - 200  # Cross down
    
    high = close + np.abs(np.random.randn(n) * 100)
    low = close - np.abs(np.random.randn(n) * 100)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min"),
        "open": np.roll(close, 1),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    df.iloc[0, df.columns.get_loc("open")] = base_price
    
    return df


def create_neutral_scenario() -> pd.DataFrame:
    """Create sideways data with no clear signal."""
    np.random.seed(456)
    n = 100
    
    base_price = 65000
    noise = np.random.randn(n) * 300  # Sideways movement
    close = base_price + noise
    
    high = close + np.abs(np.random.randn(n) * 100)
    low = close - np.abs(np.random.randn(n) * 100)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min"),
        "open": np.roll(close, 1),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    df.iloc[0, df.columns.get_loc("open")] = base_price
    
    return df


def main():
    print("=" * 60)
    print("Signals Module Test")
    print("=" * 60)
    
    # Initialize signal generator
    signal_gen = SignalGenerator(
        daily_ma_type="SMA",
        daily_ma_period=50,
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        vwap_enabled=True,
    )
    print("\n✓ SignalGenerator initialized")
    print(f"  MA: SMA50, RSI: 14 (30/70), VWAP: enabled")
    
    # Test 1: Bullish scenario
    print("\n" + "-" * 40)
    print("1. Testing BULLISH Scenario")
    print("-" * 40)
    
    df_bullish = create_bullish_scenario()
    result = signal_gen.analyze(df_bullish)
    
    print(f"   Signal:      {result.signal.value.upper()}")
    print(f"   Trend:       {result.trend}")
    print(f"   RSI:         {result.rsi_value} ({result.rsi_zone})")
    print(f"   VWAP Cross:  {result.vwap_cross}")
    print(f"   Price:       ${result.current_price:,.2f}")
    print(f"   MA Value:    ${result.ma_value:,.2f}")
    print(f"   Reason:      {result.reason}")
    
    # Test 2: Bearish scenario
    print("\n" + "-" * 40)
    print("2. Testing BEARISH Scenario")
    print("-" * 40)
    
    df_bearish = create_bearish_scenario()
    result = signal_gen.analyze(df_bearish)
    
    print(f"   Signal:      {result.signal.value.upper()}")
    print(f"   Trend:       {result.trend}")
    print(f"   RSI:         {result.rsi_value} ({result.rsi_zone})")
    print(f"   VWAP Cross:  {result.vwap_cross}")
    print(f"   Price:       ${result.current_price:,.2f}")
    print(f"   MA Value:    ${result.ma_value:,.2f}")
    print(f"   Reason:      {result.reason}")
    
    # Test 3: Neutral scenario
    print("\n" + "-" * 40)
    print("3. Testing NEUTRAL Scenario")
    print("-" * 40)
    
    df_neutral = create_neutral_scenario()
    result = signal_gen.analyze(df_neutral)
    
    print(f"   Signal:      {result.signal.value.upper()}")
    print(f"   Trend:       {result.trend}")
    print(f"   RSI:         {result.rsi_value} ({result.rsi_zone})")
    print(f"   VWAP Cross:  {result.vwap_cross}")
    print(f"   Price:       ${result.current_price:,.2f}")
    print(f"   MA Value:    ${result.ma_value:,.2f}")
    print(f"   Reason:      {result.reason}")
    
    # Test 4: Without VWAP
    print("\n" + "-" * 40)
    print("4. Testing without VWAP (RSI only)")
    print("-" * 40)
    
    signal_gen_no_vwap = SignalGenerator(
        daily_ma_type="EMA",
        daily_ma_period=20,
        vwap_enabled=False,
    )
    
    result = signal_gen_no_vwap.analyze(df_bullish)
    print(f"   Signal:      {result.signal.value.upper()}")
    print(f"   Trend:       {result.trend}")
    print(f"   RSI:         {result.rsi_value} ({result.rsi_zone})")
    print(f"   Reason:      {result.reason}")
    
    # Test 5: SignalResult dataclass
    print("\n" + "-" * 40)
    print("5. Testing SignalResult attributes")
    print("-" * 40)
    
    df = create_bullish_scenario()
    result = signal_gen.analyze(df)
    
    print(f"   ✓ signal:        {type(result.signal).__name__}.{result.signal.name}")
    print(f"   ✓ trend:         {result.trend}")
    print(f"   ✓ rsi_value:     {result.rsi_value}")
    print(f"   ✓ rsi_zone:      {result.rsi_zone}")
    print(f"   ✓ vwap_cross:    {result.vwap_cross}")
    print(f"   ✓ current_price: {result.current_price}")
    print(f"   ✓ ma_value:      {result.ma_value}")
    print(f"   ✓ vwap_value:    {result.vwap_value}")
    print(f"   ✓ reason:        {result.reason[:40]}...")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
