#!/usr/bin/env python3
"""Run parameter optimization on historical data.

Usage:
    uv run python backtest/run_optimization.py
    
This script:
1. Fetches last 30 days of candle data from Hyperliquid
2. Runs grid search optimization on strategy parameters
3. Prints a detailed report
4. Saves the best configuration to config_optimized.json
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exchange.client import HyperliquidClient
from backtest.backtester import Backtester
from backtest.optimizer import ParameterOptimizer, print_optimization_report


def fetch_historical_data(
    symbol: str = "BTC",
    interval: str = "15m",
    days: int = 30,
    use_testnet: bool = True,
):
    """Fetch historical candle data from Hyperliquid.
    
    Note: This requires .env to be configured, but we'll use public endpoints
    for read-only data fetching.
    """
    from hyperliquid.info import Info
    from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL
    import pandas as pd
    
    # We can use Info without authentication for public data
    base_url = TESTNET_API_URL if use_testnet else MAINNET_API_URL
    
    # Apply the spotMeta patch
    from hyperliquid.api import API
    _original_post = API.post
    
    def _patched_post(self, url_path: str, payload: dict = None):
        if payload and payload.get("type") == "spotMeta":
            return {"universe": [], "tokens": []}
        return _original_post(self, url_path, payload)
    
    API.post = _patched_post
    
    info = Info(base_url=base_url, skip_ws=True)
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    print(f"Fetching {symbol} {interval} candles from {start_time.date()} to {end_time.date()}...")
    
    # Fetch candles
    candles = info.candles_snapshot(
        name=symbol,
        interval=interval,
        startTime=start_ms,
        endTime=end_ms,
    )
    
    if not candles:
        raise ValueError(f"No candle data returned for {symbol}")
    
    # Convert to DataFrame
    df = pd.DataFrame(candles)
    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    })
    
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"✓ Fetched {len(df)} candles")
    print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
    
    return df


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     HYPERLIQUID BOT - PARAMETER OPTIMIZER                 ║
    ║     ─────────────────────────────────────                 ║
    ║     Grid search for optimal strategy parameters           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    SYMBOL = "BTC"
    INTERVAL = "15m"
    DAYS = 30
    USE_TESTNET = False  # Use mainnet for real historical data
    
    INITIAL_CAPITAL = 10000.0
    POSITION_SIZE_PERCENT = 5.0
    LEVERAGE = 5
    
    OPTIMIZATION_METRIC = "total_pnl"  # Options: total_pnl, sharpe_ratio, profit_factor, win_rate
    QUICK_MODE = False  # Set to True for faster testing with fewer combinations
    
    # Fetch historical data
    print("=" * 60)
    print("STEP 1: Fetching Historical Data")
    print("=" * 60)
    
    try:
        df = fetch_historical_data(
            symbol=SYMBOL,
            interval=INTERVAL,
            days=DAYS,
            use_testnet=USE_TESTNET,
        )
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("\nTrying with sample data for demonstration...")
        
        # Generate sample data for demonstration
        import numpy as np
        import pandas as pd
        
        np.random.seed(42)
        n = DAYS * 24 * 4  # 15-minute candles
        
        base_price = 65000
        returns = np.random.randn(n) * 0.005
        close = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=datetime.now(), periods=n, freq="15min"),
            "open": np.roll(close, 1),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.003)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.003)),
            "close": close,
            "volume": np.random.randint(100, 1000, n).astype(float),
        })
        df.iloc[0, 1] = base_price
        
        print(f"✓ Generated {len(df)} sample candles")
    
    # Run optimization
    print("\n" + "=" * 60)
    print("STEP 2: Running Parameter Optimization")
    print("=" * 60 + "\n")
    
    optimizer = ParameterOptimizer(
        initial_capital=INITIAL_CAPITAL,
        position_size_percent=POSITION_SIZE_PERCENT,
        leverage=LEVERAGE,
    )
    
    if QUICK_MODE:
        result = optimizer.quick_optimize(
            df,
            metric=OPTIMIZATION_METRIC,
            show_progress=True,
        )
    else:
        result = optimizer.optimize(
            df,
            metric=OPTIMIZATION_METRIC,
            min_trades=5,
            show_progress=True,
        )
    
    # Print report
    print_optimization_report(result)
    
    # Save config
    print("\n" + "=" * 60)
    print("STEP 3: Saving Optimized Configuration")
    print("=" * 60)
    
    config_path = result.save_config("config_optimized.json")
    print(f"\n✓ Saved optimized config to: {config_path}")
    print("\nTo use these parameters, copy config_optimized.json to config.json:")
    print("  cp config_optimized.json config.json")
    
    # Show the config
    print("\nOptimized config.json content:")
    print("-" * 40)
    import json
    config = result.to_config_json()
    print(json.dumps(config, indent=2))
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
