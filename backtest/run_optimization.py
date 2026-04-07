#!/usr/bin/env python3
"""Run parameter optimization on historical data.

Usage:
    uv run python backtest/run_optimization.py
    
This script:
1. Fetches last 30 days of candle data from Hyperliquid
2. Runs grid search optimization on strategy parameters
3. Prints a detailed report
4. Directly overwrites config.json with optimized settings
5. Sends Telegram notification with results

Can be run as a daily cron job for autonomous parameter adaptation.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from backtest.backtester import Backtester
from backtest.optimizer import ParameterOptimizer, print_optimization_report
from src.utils.notifier import TelegramNotifier


def load_telegram_credentials() -> tuple[str, str]:
    """Load Telegram credentials from .env file."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    return bot_token, chat_id


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


def send_optimization_notification(
    notifier: TelegramNotifier,
    result,
    config_path: str,
) -> bool:
    """Send Telegram notification with optimization results.
    
    Args:
        notifier: TelegramNotifier instance.
        result: OptimizationResult object.
        config_path: Path where config was saved.
        
    Returns:
        True if notification sent successfully.
    """
    if not notifier.enabled:
        print("ℹ Telegram notifications disabled - skipping alert")
        return False
    
    r = result.best_result
    p = result.best_params
    
    # Build notification message
    message = (
        "<b>🔄 Daily Optimization Complete</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        "<b>📊 Expected Performance:</b>\n"
        f"  • Win Rate: <b>{r.win_rate:.1f}%</b>\n"
        f"  • Total PnL: <b>${r.total_pnl:+,.2f}</b> ({r.total_pnl_percent:+.1f}%)\n"
        f"  • Profit Factor: <b>{r.profit_factor:.2f}</b>\n"
        f"  • Sharpe Ratio: <b>{r.sharpe_ratio:.2f}</b>\n\n"
        
        "<b>⚙️ New Parameters:</b>\n"
        f"  • MA: {p.get('ma_type', 'SMA')}{p.get('ma_period', 50)}\n"
        f"  • RSI: {p.get('rsi_period', 14)} ({p.get('rsi_oversold', 30)}/{p.get('rsi_overbought', 70)})\n"
        f"  • Stop Loss: {p.get('stop_loss_percent', 2.0)}%\n"
        f"  • Take Profit: {p.get('take_profit_percent', 4.0)}%\n"
        f"  • VWAP: {'✓' if p.get('use_vwap', True) else '✗'}\n\n"
        
        f"<b>✅ {config_path} saved successfully</b>\n"
        f"Bot will use new settings on next restart/loop."
    )
    
    return notifier._send_message(message)


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
    
    # Output config path (directly overwrite config.json)
    CONFIG_OUTPUT_PATH = str(PROJECT_ROOT / "config.json")
    EXISTING_CONFIG_PATH = str(PROJECT_ROOT / "config.json")
    
    # Initialize Telegram notifier
    bot_token, chat_id = load_telegram_credentials()
    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        enabled=bool(bot_token and chat_id),
    )
    
    if notifier.enabled:
        print("✓ Telegram notifications enabled")
    else:
        print("ℹ Telegram notifications disabled (no credentials)")
    
    # Fetch historical data
    print("\n" + "=" * 60)
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
    
    # Save config (merge with existing to preserve notifications, bot settings, etc.)
    print("\n" + "=" * 60)
    print("STEP 3: Saving Optimized Configuration")
    print("=" * 60)
    
    # Use smart merging to preserve existing settings
    config_path = result.save_config(
        filepath=CONFIG_OUTPUT_PATH,
        merge_existing=True,
        existing_config_path=EXISTING_CONFIG_PATH,
    )
    
    print(f"\n✓ Updated config.json with optimized parameters")
    print(f"  Path: {config_path}")
    print("\nPreserved settings: notifications, bot, symbols, margin_type, etc.")
    print("Updated settings: MA, RSI, SL, TP, VWAP, filters")
    
    # Show the final config
    print("\nFinal config.json content:")
    print("-" * 40)
    import json
    with open(config_path, "r") as f:
        config = json.load(f)
    print(json.dumps(config, indent=2))
    
    # Send Telegram notification
    print("\n" + "=" * 60)
    print("STEP 4: Sending Telegram Notification")
    print("=" * 60)
    
    if send_optimization_notification(notifier, result, "config.json"):
        print("✓ Telegram notification sent successfully")
    else:
        print("ℹ Telegram notification skipped or failed")
    
    # Save optimization timestamp
    timestamp_file = PROJECT_ROOT / "last_optimization.txt"
    with open(timestamp_file, "w") as f:
        f.write(f"Last optimization: {datetime.now().isoformat()}\n")
        f.write(f"Symbol: {SYMBOL}\n")
        f.write(f"Days analyzed: {DAYS}\n")
        f.write(f"Best Win Rate: {result.best_result.win_rate:.1f}%\n")
        f.write(f"Best PnL: ${result.best_result.total_pnl:,.2f}\n")
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print("\nThe trading bot will automatically use the new settings on its next loop.")


if __name__ == "__main__":
    main()
