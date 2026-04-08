#!/usr/bin/env python3
"""Bayesian Optimization for ATR-based volatility targeting using Optuna.

Usage:
    uv run python backtest/optuna_optimizer.py
    
This script uses Optuna for Bayesian hyperparameter optimization instead of
exhaustive grid search, providing faster convergence to optimal parameters.

Features:
- TPE (Tree-structured Parzen Estimator) sampler for intelligent search
- Automatic pruning of unpromising trials
- Risk-adjusted reward function: score = total_pnl / (max_drawdown_pct + 1)
- FULL parameter search space (ma_type, periods, RSI, ATR, VWAP)
- Automatic config.json update with backup
- Telegram notification with results
"""

import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.exceptions import TrialPruned

# Set Optuna logging to WARNING to avoid console spam
optuna.logging.set_verbosity(optuna.logging.WARNING)

from dotenv import load_dotenv
import pandas as pd
import numpy as np

from backtest.backtester import Backtester, BacktestResult
from src.utils.notifier import TelegramNotifier


# =============================================================================
# Configuration
# =============================================================================

SYMBOL = "BTC"
INTERVAL = "15m"
DAYS = 30
USE_TESTNET = False  # Use mainnet for real historical data

INITIAL_CAPITAL = 10000.0
RISK_PERCENT_PER_TRADE = 2.0
MAX_LEVERAGE = 10

N_TRIALS = 1000  # Increased for comprehensive search
MIN_TRADES = 5  # Prune trials with fewer trades

CONFIG_FILE = PROJECT_ROOT / "config.json"
CONFIG_BACKUP = PROJECT_ROOT / "config.json.bak"


# =============================================================================
# Data Fetching (reused from run_optimization.py)
# =============================================================================

def fetch_historical_data(
    symbol: str = "BTC",
    interval: str = "15m",
    days: int = 30,
    use_testnet: bool = True,
) -> pd.DataFrame:
    """Fetch historical candle data from Hyperliquid.
    
    Note: Uses public endpoints - no authentication required for read-only data.
    """
    from hyperliquid.info import Info
    from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL
    
    base_url = TESTNET_API_URL if use_testnet else MAINNET_API_URL
    
    # Apply the spotMeta patch for testnet compatibility
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


def generate_sample_data(days: int = 30) -> pd.DataFrame:
    """Generate sample data for demonstration/testing."""
    np.random.seed(42)
    n = days * 24 * 4  # 15-minute candles
    
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
    return df


# =============================================================================
# Telegram Notification
# =============================================================================

def load_telegram_credentials() -> tuple[str, str]:
    """Load Telegram credentials from .env file."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(env_path)
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    
    return bot_token, chat_id


def send_optuna_notification(
    notifier: TelegramNotifier,
    study: optuna.Study,
    best_result: Optional[BacktestResult],
    config_updated: bool = False,
) -> bool:
    """Send Telegram notification with Optuna optimization results."""
    if not notifier.enabled:
        return False
    
    if best_result is None:
        # All trials were pruned
        message = (
            "<b>⚠️ Optuna Optimization Failed</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            f"<b>📈 Trials Run:</b> {len(study.trials)}\n"
            "<b>Status:</b> All trials were pruned\n\n"
            
            "<b>Reason:</b> No parameter combination generated\n"
            f"sufficient trades (MIN_TRADES={MIN_TRADES})\n\n"
            
            "<i>Try increasing the data range or adjusting MIN_TRADES.</i>"
        )
        return notifier._send_message(message)
    
    best = study.best_params
    
    config_status = "✅ Auto-updated" if config_updated else "⚠️ Manual update needed"
    
    message = (
        "<b>🎯 Optuna Optimization Complete</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        f"<b>📊 Best Score:</b> {study.best_value:.2f}\n"
        f"<b>📈 Trials Run:</b> {len(study.trials)}\n"
        f"<b>⚙️ Config:</b> {config_status}\n\n"
        
        "<b>🏆 Expected Performance:</b>\n"
        f"  • Win Rate: <b>{best_result.win_rate:.1f}%</b>\n"
        f"  • Total PnL: <b>${best_result.total_pnl:+,.2f}</b>\n"
        f"  • Max Drawdown: <b>{best_result.max_drawdown_percent:.1f}%</b>\n"
        f"  • Sharpe Ratio: <b>{best_result.sharpe_ratio:.2f}</b>\n\n"
        
        "<b>📈 Strategy Parameters:</b>\n"
        f"  • MA: {best.get('ma_type', 'N/A')} ({best.get('ma_period', 'N/A')})\n"
        f"  • RSI: {best.get('rsi_period', 'N/A')} ({best.get('rsi_oversold', 'N/A')}/{best.get('rsi_overbought', 'N/A')})\n"
        f"  • VWAP: {best.get('use_vwap', 'N/A')}\n\n"
        
        "<b>🛡️ Risk Parameters:</b>\n"
        f"  • ATR Period: {best.get('atr_period', 'N/A')}\n"
        f"  • ATR SL Mult: <b>{best.get('atr_sl_multiplier', 'N/A')}</b>\n"
        f"  • ATR Trail Mult: <b>{best.get('atr_trailing_multiplier', 'N/A')}</b>"
    )
    
    return notifier._send_message(message)


# =============================================================================
# Optuna Objective Function
# =============================================================================

def create_objective(df: pd.DataFrame, backtester: Backtester):
    """Create an Optuna objective function with the given data and backtester.
    
    Returns a callable that Optuna will use to evaluate trials.
    Optimizes ALL strategic parameters for maximum exploration.
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.
        
        Defines FULL hyperparameter search space and returns risk-adjusted score.
        """
        # =================================================================
        # FULL HYPERPARAMETER SEARCH SPACE
        # =================================================================
        
        # Moving Average parameters
        ma_type = trial.suggest_categorical("ma_type", ["SMA", "EMA"])
        ma_period = trial.suggest_int("ma_period", 50, 200, step=10)
        
        # RSI parameters
        rsi_period = trial.suggest_int("rsi_period", 7, 21)
        rsi_oversold = trial.suggest_int("rsi_oversold", 20, 35)
        rsi_overbought = trial.suggest_int("rsi_overbought", 65, 80)
        
        # VWAP
        use_vwap = trial.suggest_categorical("use_vwap", [True, False])
        
        # ATR parameters
        atr_period = trial.suggest_int("atr_period", 7, 21)
        atr_sl_multiplier = trial.suggest_float("atr_sl_multiplier", 1.0, 3.5, step=0.1)
        atr_trailing_multiplier = trial.suggest_float("atr_trailing_multiplier", 1.5, 4.0, step=0.1)
        
        # Run backtest with ALL suggested parameters
        result = backtester.run(
            df=df,
            ma_type=ma_type,
            ma_period=ma_period,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            use_vwap=use_vwap,
            atr_period=atr_period,
            atr_sl_multiplier=atr_sl_multiplier,
            atr_trailing_multiplier=atr_trailing_multiplier,
            volatility_filter_enabled=False,
        )
        
        # Prune if insufficient trades
        if result.total_trades < MIN_TRADES:
            raise TrialPruned(f"Only {result.total_trades} trades (min: {MIN_TRADES})")
        
        # Calculate risk-adjusted score: total_pnl / (max_drawdown_pct + 1)
        # The +1 prevents division by zero and penalizes drawdown
        score = result.total_pnl / (result.max_drawdown_percent + 1)
        
        # Store additional metrics as user attributes for analysis
        trial.set_user_attr("total_trades", result.total_trades)
        trial.set_user_attr("win_rate", result.win_rate)
        trial.set_user_attr("total_pnl", result.total_pnl)
        trial.set_user_attr("max_drawdown_pct", result.max_drawdown_percent)
        trial.set_user_attr("sharpe_ratio", result.sharpe_ratio)
        trial.set_user_attr("profit_factor", result.profit_factor)
        
        return score
    
    return objective


# =============================================================================
# Main Execution
# =============================================================================

def run_optimization(df: pd.DataFrame, n_trials: int = N_TRIALS) -> tuple[optuna.Study, Optional[BacktestResult]]:
    """Run Optuna optimization and return the study and best result.
    
    Args:
        df: Historical OHLCV data.
        n_trials: Number of trials to run.
        
    Returns:
        Tuple of (optuna.Study, BacktestResult for best params or None if all pruned)
    """
    # Initialize backtester
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        risk_percent_per_trade=RISK_PERCENT_PER_TRADE,
        max_leverage=MAX_LEVERAGE,
    )
    
    # Create objective function
    objective = create_objective(df, backtester)
    
    # Create study with TPE sampler (Bayesian optimization)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="atr_volatility_targeting",
    )
    
    # Run optimization
    print(f"\n🔄 Running Optuna optimization ({n_trials} trials)...")
    print("-" * 50)
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),  # Catch exceptions but continue optimization
    )
    
    # Check if any trials completed successfully
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("\n⚠️ Warning: All trials were pruned or failed!")
        print("   Try increasing MIN_TRADES threshold or using more historical data.")
        return study, None
    
    # Get best result by re-running with best params
    best = study.best_params
    best_result = backtester.run(
        df=df,
        ma_type=best["ma_type"],
        ma_period=best["ma_period"],
        rsi_period=best["rsi_period"],
        rsi_oversold=best["rsi_oversold"],
        rsi_overbought=best["rsi_overbought"],
        use_vwap=best["use_vwap"],
        atr_period=best["atr_period"],
        atr_sl_multiplier=best["atr_sl_multiplier"],
        atr_trailing_multiplier=best["atr_trailing_multiplier"],
        volatility_filter_enabled=False,
    )
    
    return study, best_result


def print_summary(study: optuna.Study, best_result: Optional[BacktestResult]):
    """Print a clean summary of optimization results."""
    
    print("\n" + "=" * 60)
    print("🎯 OPTUNA OPTIMIZATION RESULTS")
    print("=" * 60)
    
    # Trial statistics
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    print("\n📊 Trial Statistics:")
    print(f"   • Completed: {completed}")
    print(f"   • Pruned: {pruned}")
    print(f"   • Failed: {failed}")
    
    if best_result is None:
        print("\n⚠️ No valid trials completed!")
        print("   All parameter combinations resulted in fewer than MIN_TRADES trades.")
        print("   Consider:")
        print("   - Using more historical data (increase DAYS)")
        print("   - Lowering MIN_TRADES threshold")
        print("   - Widening parameter search space")
        return
    
    print(f"\n📊 Best Score: {study.best_value:.4f}")
    print(f"   (score = total_pnl / (max_drawdown_pct + 1))")
    
    best = study.best_params
    
    print("\n📈 Best Parameters:")
    print("   Strategy:")
    print(f"   • MA Type: {best.get('ma_type', 'N/A')}")
    print(f"   • MA Period: {best.get('ma_period', 'N/A')}")
    print(f"   • RSI Period: {best.get('rsi_period', 'N/A')}")
    print(f"   • RSI Oversold: {best.get('rsi_oversold', 'N/A')}")
    print(f"   • RSI Overbought: {best.get('rsi_overbought', 'N/A')}")
    print(f"   • Use VWAP: {best.get('use_vwap', 'N/A')}")
    print("   Risk Management:")
    print(f"   • ATR Period: {best.get('atr_period', 'N/A')}")
    print(f"   • ATR SL Multiplier: {best.get('atr_sl_multiplier', 'N/A'):.1f}")
    print(f"   • ATR Trailing Multiplier: {best.get('atr_trailing_multiplier', 'N/A'):.1f}")
    
    print("\n🏆 Performance Metrics:")
    print(f"   • Total Trades: {best_result.total_trades}")
    print(f"   • Win Rate: {best_result.win_rate:.1f}%")
    print(f"   • Total PnL: ${best_result.total_pnl:+,.2f} ({best_result.total_pnl_percent:+.1f}%)")
    print(f"   • Max Drawdown: ${best_result.max_drawdown:,.2f} ({best_result.max_drawdown_percent:.1f}%)")
    print(f"   • Profit Factor: {best_result.profit_factor:.2f}")
    print(f"   • Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
    print(f"   • Avg Leverage: {best_result.average_leverage:.1f}x")
    print(f"   • Chandelier Exits: {best_result.chandelier_exits}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Automatic Config Update
# =============================================================================

def update_config_with_best_params(best_params: dict) -> bool:
    """Update config.json with best parameters from optimization.
    
    Creates a backup before modifying and only updates relevant fields.
    
    Args:
        best_params: Dictionary of best parameters from Optuna study
        
    Returns:
        True if config was successfully updated, False otherwise
    """
    try:
        # Read current config
        if not CONFIG_FILE.exists():
            print(f"❌ Config file not found: {CONFIG_FILE}")
            return False
        
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        
        # Create backup
        shutil.copy(CONFIG_FILE, CONFIG_BACKUP)
        print(f"✓ Backup created: {CONFIG_BACKUP}")
        
        # Update strategy section
        if "strategy" not in config:
            config["strategy"] = {}
        
        config["strategy"]["daily_ma_type"] = best_params["ma_type"]
        config["strategy"]["daily_ma_period"] = best_params["ma_period"]
        config["strategy"]["rsi_period"] = best_params["rsi_period"]
        config["strategy"]["rsi_oversold"] = best_params["rsi_oversold"]
        config["strategy"]["rsi_overbought"] = best_params["rsi_overbought"]
        config["strategy"]["vwap_enabled"] = best_params["use_vwap"]
        
        # Update risk_management section
        if "risk_management" not in config:
            config["risk_management"] = {}
        
        config["risk_management"]["atr_period"] = best_params["atr_period"]
        config["risk_management"]["atr_sl_multiplier"] = round(best_params["atr_sl_multiplier"], 1)
        config["risk_management"]["atr_trailing_multiplier"] = round(best_params["atr_trailing_multiplier"], 1)
        
        # Write updated config
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     HYPERLIQUID BOT - OPTUNA BAYESIAN OPTIMIZER           ║
    ║     ─────────────────────────────────────────────         ║
    ║     Full Parameter Search with Auto Config Update         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print(f"Search Space: {N_TRIALS} trials")
    print("Parameters: ma_type, ma_period, rsi_period, rsi_oversold, rsi_overbought,")
    print("            use_vwap, atr_period, atr_sl_multiplier, atr_trailing_multiplier")
    
    # Step 1: Fetch historical data
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
        print("\nUsing sample data for demonstration...")
        df = generate_sample_data(DAYS)
    
    # Step 2: Run Optuna optimization
    print("\n" + "=" * 60)
    print("STEP 2: Running Bayesian Optimization")
    print("=" * 60)
    
    study, best_result = run_optimization(df, n_trials=N_TRIALS)
    
    # Step 3: Print summary
    print_summary(study, best_result)
    
    # Track if config was updated
    config_updated = False
    
    # Step 4: Auto-update config.json (if optimization succeeded)
    if best_result is not None:
        print("\n" + "=" * 60)
        print("STEP 3: Updating config.json")
        print("=" * 60)
        
        config_updated = update_config_with_best_params(study.best_params)
        
        if config_updated:
            print("✓ config.json updated with best parameters")
            print("\n✅ config.json has been automatically updated with the best parameters.")
        else:
            print("⚠️ Failed to update config.json - manual update required")
    
    # Step 5: Send Telegram notification
    print("\n" + "=" * 60)
    print("STEP 4: Sending Telegram Notification")
    print("=" * 60)
    
    bot_token, chat_id = load_telegram_credentials()
    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        enabled=bool(bot_token and chat_id),
    )
    
    if send_optuna_notification(notifier, study, best_result, config_updated):
        print("✓ Telegram notification sent successfully")
    else:
        print("ℹ Telegram notification skipped or failed")
    
    # Step 6: Save results to log file
    print("\n" + "=" * 60)
    print("STEP 5: Saving Results Log")
    print("=" * 60)
    
    timestamp_file = PROJECT_ROOT / "last_optuna_optimization.txt"
    
    if best_result is None:
        # No valid trials completed
        with open(timestamp_file, "w") as f:
            f.write(f"Last Optuna optimization: {datetime.now().isoformat()}\n")
            f.write(f"Symbol: {SYMBOL}\n")
            f.write(f"Days analyzed: {DAYS}\n")
            f.write(f"Trials: {N_TRIALS}\n")
            f.write("Status: FAILED - All trials were pruned\n")
            f.write("Reason: No parameter combination generated >= MIN_TRADES trades\n")
        
        print(f"⚠️ Results saved to {timestamp_file} (optimization failed)")
        print("\n❌ Optimization could not find valid parameters.")
        print("   Try increasing data range or lowering MIN_TRADES.")
        return study, best_result
    
    best = study.best_params
    with open(timestamp_file, "w") as f:
        f.write(f"Last Optuna optimization: {datetime.now().isoformat()}\n")
        f.write(f"Symbol: {SYMBOL}\n")
        f.write(f"Days analyzed: {DAYS}\n")
        f.write(f"Trials: {N_TRIALS}\n")
        f.write(f"Best Score: {study.best_value:.4f}\n")
        f.write(f"Config Updated: {config_updated}\n")
        f.write(f"\nBest Strategy Params:\n")
        f.write(f"  ma_type: {best['ma_type']}\n")
        f.write(f"  ma_period: {best['ma_period']}\n")
        f.write(f"  rsi_period: {best['rsi_period']}\n")
        f.write(f"  rsi_oversold: {best['rsi_oversold']}\n")
        f.write(f"  rsi_overbought: {best['rsi_overbought']}\n")
        f.write(f"  use_vwap: {best['use_vwap']}\n")
        f.write(f"\nBest Risk Params:\n")
        f.write(f"  atr_period: {best['atr_period']}\n")
        f.write(f"  atr_sl_multiplier: {best['atr_sl_multiplier']:.1f}\n")
        f.write(f"  atr_trailing_multiplier: {best['atr_trailing_multiplier']:.1f}\n")
        f.write(f"\nPerformance:\n")
        f.write(f"  Win Rate: {best_result.win_rate:.1f}%\n")
        f.write(f"  Total PnL: ${best_result.total_pnl:,.2f}\n")
        f.write(f"  Max Drawdown: {best_result.max_drawdown_percent:.1f}%\n")
        f.write(f"  Sharpe Ratio: {best_result.sharpe_ratio:.2f}\n")
    
    print(f"✓ Results saved to {timestamp_file}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    if config_updated:
        print("\n✅ config.json has been automatically updated with the best parameters.")
        print(f"   Backup saved to: {CONFIG_BACKUP}")
    else:
        print("\n⚠️ config.json was NOT updated. Apply these settings manually:")
    
    print(f"\n   Strategy:")
    print(f"   • MA Type: {best['ma_type']}")
    print(f"   • MA Period: {best['ma_period']}")
    print(f"   • RSI Period: {best['rsi_period']}")
    print(f"   • RSI Oversold/Overbought: {best['rsi_oversold']}/{best['rsi_overbought']}")
    print(f"   • VWAP Enabled: {best['use_vwap']}")
    print(f"\n   Risk Management:")
    print(f"   • ATR Period: {best['atr_period']}")
    print(f"   • ATR SL Multiplier: {best['atr_sl_multiplier']:.1f}")
    print(f"   • ATR Trailing Multiplier: {best['atr_trailing_multiplier']:.1f}")
    
    return study, best_result


if __name__ == "__main__":
    main()
