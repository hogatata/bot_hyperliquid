#!/usr/bin/env python3
"""Robustness Check for Optimized Trading Strategy Parameters.

Usage:
    uv run python backtest/robustness_check.py
    
This script validates optimized parameters from config.json against overfitting:
1. Out-of-Sample (OOS) Test: Compare performance on training vs unseen data
2. Sensitivity Test: Verify strategy doesn't collapse with slight param changes

A robust strategy should:
- Perform reasonably well on unseen data (not just the training period)
- Not show dramatic performance drops with small parameter variations
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from backtest.backtester import Backtester, BacktestResult


# =============================================================================
# Configuration
# =============================================================================

SYMBOL = "BTC"
INTERVAL = "15m"
TOTAL_DAYS = 90  # Total data to fetch
IN_SAMPLE_DAYS = 30  # Last 30 days (what Optuna trained on)
OUT_OF_SAMPLE_DAYS = 60  # First 60 days (blind test period)

USE_TESTNET = False  # Use mainnet for real data

INITIAL_CAPITAL = 10000.0
RISK_PERCENT_PER_TRADE = 2.0
MAX_LEVERAGE = 10

CONFIG_FILE = PROJECT_ROOT / "config.json"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestResult:
    """Result from a single backtest run."""
    name: str
    total_pnl: float
    total_pnl_percent: float
    win_rate: float
    total_trades: int
    max_drawdown_percent: float
    sharpe_ratio: float
    profit_factor: float
    days: int
    
    @property
    def pnl_per_day(self) -> float:
        """Normalized PnL per day for fair comparison."""
        return self.total_pnl / self.days if self.days > 0 else 0.0


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_historical_data(
    symbol: str = "BTC",
    interval: str = "15m",
    days: int = 90,
    use_testnet: bool = False,
) -> pd.DataFrame:
    """Fetch historical candle data from Hyperliquid."""
    from hyperliquid.info import Info
    from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL
    
    base_url = TESTNET_API_URL if use_testnet else MAINNET_API_URL
    
    # Apply the spotMeta patch for compatibility
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


def generate_sample_data(days: int = 90) -> pd.DataFrame:
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


def split_data(df: pd.DataFrame, in_sample_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into out-of-sample (older) and in-sample (recent) periods.
    
    Args:
        df: Full historical DataFrame
        in_sample_days: Number of days for in-sample (training) period
        
    Returns:
        Tuple of (df_out_of_sample, df_in_sample)
    """
    # Calculate split point based on candles per day (96 for 15m candles)
    candles_per_day = 24 * 4  # 15-minute candles
    in_sample_candles = in_sample_days * candles_per_day
    
    # Split: first part is OOS, last part is in-sample
    split_idx = len(df) - in_sample_candles
    
    df_out_of_sample = df.iloc[:split_idx].reset_index(drop=True)
    df_in_sample = df.iloc[split_idx:].reset_index(drop=True)
    
    return df_out_of_sample, df_in_sample


# =============================================================================
# Config Loading
# =============================================================================

def load_config() -> dict:
    """Load configuration from config.json."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def extract_strategy_params(config: dict) -> dict:
    """Extract strategy and risk management parameters from config."""
    strategy = config.get("strategy", {})
    risk_mgmt = config.get("risk_management", {})
    
    return {
        "ma_type": strategy.get("daily_ma_type", "SMA"),
        "ma_period": strategy.get("daily_ma_period", 100),
        "rsi_period": strategy.get("rsi_period", 14),
        "rsi_oversold": strategy.get("rsi_oversold", 30),
        "rsi_overbought": strategy.get("rsi_overbought", 70),
        "use_vwap": strategy.get("vwap_enabled", True),
        "atr_period": risk_mgmt.get("atr_period", 14),
        "atr_sl_multiplier": risk_mgmt.get("atr_sl_multiplier", 1.5),
        "atr_trailing_multiplier": risk_mgmt.get("atr_trailing_multiplier", 2.0),
    }


# =============================================================================
# Backtesting
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    params: dict,
    name: str,
    days: int,
) -> TestResult:
    """Run a single backtest and return structured results."""
    backtester = Backtester(
        initial_capital=INITIAL_CAPITAL,
        risk_percent_per_trade=RISK_PERCENT_PER_TRADE,
        max_leverage=MAX_LEVERAGE,
    )
    
    result = backtester.run(
        df=df,
        ma_type=params["ma_type"],
        ma_period=params["ma_period"],
        rsi_period=params["rsi_period"],
        rsi_oversold=params["rsi_oversold"],
        rsi_overbought=params["rsi_overbought"],
        use_vwap=params["use_vwap"],
        atr_period=params["atr_period"],
        atr_sl_multiplier=params["atr_sl_multiplier"],
        atr_trailing_multiplier=params["atr_trailing_multiplier"],
        volatility_filter_enabled=False,
    )
    
    return TestResult(
        name=name,
        total_pnl=result.total_pnl,
        total_pnl_percent=result.total_pnl_percent,
        win_rate=result.win_rate,
        total_trades=result.total_trades,
        max_drawdown_percent=result.max_drawdown_percent,
        sharpe_ratio=result.sharpe_ratio,
        profit_factor=result.profit_factor,
        days=days,
    )


# =============================================================================
# Test Runners
# =============================================================================

def run_oos_test(
    df_in_sample: pd.DataFrame,
    df_out_of_sample: pd.DataFrame,
    params: dict,
) -> tuple[TestResult, TestResult]:
    """Run Out-of-Sample validation test.
    
    Returns:
        Tuple of (in_sample_result, out_of_sample_result)
    """
    print("\n📊 Running In-Sample backtest...")
    in_sample_result = run_backtest(
        df_in_sample,
        params,
        "In-Sample (Training)",
        IN_SAMPLE_DAYS,
    )
    
    print("📊 Running Out-of-Sample backtest...")
    oos_result = run_backtest(
        df_out_of_sample,
        params,
        "Out-of-Sample (Blind)",
        OUT_OF_SAMPLE_DAYS,
    )
    
    return in_sample_result, oos_result


def run_sensitivity_test(
    df: pd.DataFrame,
    base_params: dict,
) -> list[TestResult]:
    """Run sensitivity tests with parameter variations.
    
    Tests 4 variations to check for parameter cliff effects.
    """
    variations = []
    
    # Variation 1: RSI Period + 1
    params_1 = base_params.copy()
    params_1["rsi_period"] = base_params["rsi_period"] + 1
    print(f"📊 Testing RSI Period + 1 ({params_1['rsi_period']})...")
    variations.append(run_backtest(
        df, params_1, f"RSI Period = {params_1['rsi_period']}", OUT_OF_SAMPLE_DAYS
    ))
    
    # Variation 2: RSI Period - 1
    params_2 = base_params.copy()
    params_2["rsi_period"] = max(2, base_params["rsi_period"] - 1)
    print(f"📊 Testing RSI Period - 1 ({params_2['rsi_period']})...")
    variations.append(run_backtest(
        df, params_2, f"RSI Period = {params_2['rsi_period']}", OUT_OF_SAMPLE_DAYS
    ))
    
    # Variation 3: ATR Trailing + 0.3
    params_3 = base_params.copy()
    params_3["atr_trailing_multiplier"] = base_params["atr_trailing_multiplier"] + 0.3
    print(f"📊 Testing ATR Trailing + 0.3 ({params_3['atr_trailing_multiplier']:.1f})...")
    variations.append(run_backtest(
        df, params_3, f"ATR Trail = {params_3['atr_trailing_multiplier']:.1f}", OUT_OF_SAMPLE_DAYS
    ))
    
    # Variation 4: ATR Trailing - 0.3
    params_4 = base_params.copy()
    params_4["atr_trailing_multiplier"] = max(0.5, base_params["atr_trailing_multiplier"] - 0.3)
    print(f"📊 Testing ATR Trailing - 0.3 ({params_4['atr_trailing_multiplier']:.1f})...")
    variations.append(run_backtest(
        df, params_4, f"ATR Trail = {params_4['atr_trailing_multiplier']:.1f}", OUT_OF_SAMPLE_DAYS
    ))
    
    return variations


# =============================================================================
# Report Generation
# =============================================================================

def print_report(
    params: dict,
    in_sample: TestResult,
    out_of_sample: TestResult,
    sensitivity_results: list[TestResult],
):
    """Print comprehensive robustness check report."""
    
    print("\n")
    print("=" * 70)
    print("                    🔬 ROBUSTNESS CHECK REPORT")
    print("=" * 70)
    
    # Parameters Section
    print("\n📋 LOADED PARAMETERS FROM config.json:")
    print("-" * 50)
    print(f"   MA Type: {params['ma_type']} | Period: {params['ma_period']}")
    print(f"   RSI Period: {params['rsi_period']} | Oversold: {params['rsi_oversold']} | Overbought: {params['rsi_overbought']}")
    print(f"   VWAP Enabled: {params['use_vwap']}")
    print(f"   ATR Period: {params['atr_period']}")
    print(f"   ATR SL Multiplier: {params['atr_sl_multiplier']:.1f}")
    print(f"   ATR Trailing Multiplier: {params['atr_trailing_multiplier']:.1f}")
    
    # Test 1: Out-of-Sample Validation
    print("\n")
    print("=" * 70)
    print("   TEST 1: OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)
    print("\n   Comparing performance on training data vs unseen data...")
    print()
    
    # Table header
    print(f"   {'Dataset':<25} {'Days':>6} {'PnL':>12} {'PnL/Day':>10} {'Win Rate':>10} {'Trades':>8}")
    print("   " + "-" * 73)
    
    # In-sample row
    pnl_color_is = "+" if in_sample.total_pnl >= 0 else ""
    print(f"   {'In-Sample (Training)':<25} {in_sample.days:>6} "
          f"${pnl_color_is}{in_sample.total_pnl:>10,.2f} "
          f"${in_sample.pnl_per_day:>8,.2f} "
          f"{in_sample.win_rate:>9.1f}% "
          f"{in_sample.total_trades:>8}")
    
    # Out-of-sample row
    pnl_color_oos = "+" if out_of_sample.total_pnl >= 0 else ""
    print(f"   {'Out-of-Sample (Blind)':<25} {out_of_sample.days:>6} "
          f"${pnl_color_oos}{out_of_sample.total_pnl:>10,.2f} "
          f"${out_of_sample.pnl_per_day:>8,.2f} "
          f"{out_of_sample.win_rate:>9.1f}% "
          f"{out_of_sample.total_trades:>8}")
    
    # Calculate degradation
    if in_sample.pnl_per_day != 0:
        pnl_degradation = ((out_of_sample.pnl_per_day - in_sample.pnl_per_day) / abs(in_sample.pnl_per_day)) * 100
    else:
        pnl_degradation = 0
    
    print()
    print(f"   📉 Performance Change (PnL/Day): {pnl_degradation:+.1f}%")
    
    # Additional metrics comparison
    print()
    print(f"   {'Metric':<25} {'In-Sample':>15} {'Out-of-Sample':>15}")
    print("   " + "-" * 55)
    print(f"   {'Max Drawdown':<25} {in_sample.max_drawdown_percent:>14.1f}% {out_of_sample.max_drawdown_percent:>14.1f}%")
    print(f"   {'Sharpe Ratio':<25} {in_sample.sharpe_ratio:>15.2f} {out_of_sample.sharpe_ratio:>15.2f}")
    print(f"   {'Profit Factor':<25} {in_sample.profit_factor:>15.2f} {out_of_sample.profit_factor:>15.2f}")
    
    # Test 2: Sensitivity Analysis
    print("\n")
    print("=" * 70)
    print("   TEST 2: SENSITIVITY ANALYSIS (CLIFF TEST)")
    print("=" * 70)
    print("\n   Testing slight parameter variations on Out-of-Sample data...")
    print(f"   Baseline OOS PnL: ${out_of_sample.total_pnl:+,.2f}")
    print()
    
    # Table header
    print(f"   {'Variation':<25} {'PnL':>12} {'Δ vs Baseline':>15} {'Win Rate':>10} {'Trades':>8}")
    print("   " + "-" * 72)
    
    for result in sensitivity_results:
        delta = result.total_pnl - out_of_sample.total_pnl
        delta_pct = (delta / abs(out_of_sample.total_pnl) * 100) if out_of_sample.total_pnl != 0 else 0
        
        pnl_sign = "+" if result.total_pnl >= 0 else ""
        delta_sign = "+" if delta >= 0 else ""
        
        print(f"   {result.name:<25} "
              f"${pnl_sign}{result.total_pnl:>10,.2f} "
              f"{delta_sign}${delta:>7,.0f} ({delta_pct:+.0f}%) "
              f"{result.win_rate:>9.1f}% "
              f"{result.total_trades:>8}")
    
    # Calculate sensitivity score
    baseline_pnl = out_of_sample.total_pnl
    max_deviation = max(abs(r.total_pnl - baseline_pnl) for r in sensitivity_results)
    avg_pnl = sum(r.total_pnl for r in sensitivity_results) / len(sensitivity_results)
    
    print()
    print(f"   📊 Average PnL across variations: ${avg_pnl:+,.2f}")
    print(f"   📊 Max deviation from baseline: ${max_deviation:,.2f}")
    
    # Check for cliff effects (dramatic drops)
    cliff_detected = any(r.total_pnl < baseline_pnl * 0.5 for r in sensitivity_results)
    
    # Final Verdict
    print("\n")
    print("=" * 70)
    print("                         📋 FINAL VERDICT")
    print("=" * 70)
    print()
    
    # Verdict criteria
    oos_profitable = out_of_sample.total_pnl >= 0
    oos_positive_sharpe = out_of_sample.sharpe_ratio > 0
    reasonable_degradation = pnl_degradation > -70  # Not more than 70% worse
    no_cliff = not cliff_detected
    
    # Print individual checks
    print(f"   {'✅' if oos_profitable else '❌'} Out-of-Sample PnL: ${out_of_sample.total_pnl:+,.2f}")
    print(f"   {'✅' if oos_positive_sharpe else '⚠️'} Out-of-Sample Sharpe: {out_of_sample.sharpe_ratio:.2f}")
    print(f"   {'✅' if reasonable_degradation else '⚠️'} Performance Degradation: {pnl_degradation:+.1f}%")
    print(f"   {'✅' if no_cliff else '⚠️'} No Cliff Effects Detected: {no_cliff}")
    print()
    
    # Overall verdict
    if not oos_profitable:
        print("   " + "=" * 60)
        print("   ❌ WARNING: SEVERE OVERFITTING DETECTED!")
        print("   ❌ Strategy fails on unseen data.")
        print("   " + "=" * 60)
        print()
        print("   Recommendations:")
        print("   • The optimized parameters are likely overfit to training data")
        print("   • Consider using a longer training period (DAYS > 30)")
        print("   • Try adding regularization in the objective function")
        print("   • Use walk-forward optimization instead of single split")
    elif cliff_detected:
        print("   " + "=" * 60)
        print("   ⚠️ WARNING: CLIFF EFFECTS DETECTED!")
        print("   ⚠️ Strategy is sensitive to small parameter changes.")
        print("   " + "=" * 60)
        print()
        print("   Recommendations:")
        print("   • Parameters may be at a local optimum edge")
        print("   • Consider widening parameter search bounds")
        print("   • Use ensemble of nearby parameter sets")
    elif pnl_degradation < -50:
        print("   " + "=" * 60)
        print("   ⚠️ CAUTION: SIGNIFICANT PERFORMANCE DEGRADATION")
        print("   ⚠️ Strategy performs notably worse on unseen data.")
        print("   " + "=" * 60)
        print()
        print("   Recommendations:")
        print("   • Some overfitting may be present")
        print("   • Monitor live performance closely")
        print("   • Consider more conservative position sizing")
    else:
        print("   " + "=" * 60)
        print("   ✅ STRATEGY APPEARS ROBUST ON UNSEEN DATA")
        print("   ✅ Parameters passed overfitting checks.")
        print("   " + "=" * 60)
        print()
        print("   Notes:")
        print("   • Strategy shows consistent behavior across periods")
        print("   • No severe cliff effects with parameter variations")
        print("   • Ready for paper trading validation")
    
    print()
    print("=" * 70)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     HYPERLIQUID BOT - ROBUSTNESS CHECK                    ║
    ║     ─────────────────────────────────────────────         ║
    ║     Out-of-Sample & Sensitivity Testing                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Load config
    print("=" * 60)
    print("STEP 1: Loading Configuration")
    print("=" * 60)
    
    try:
        config = load_config()
        params = extract_strategy_params(config)
        print(f"✓ Loaded parameters from {CONFIG_FILE}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Step 2: Fetch historical data
    print("\n" + "=" * 60)
    print("STEP 2: Fetching Historical Data")
    print("=" * 60)
    
    try:
        df = fetch_historical_data(
            symbol=SYMBOL,
            interval=INTERVAL,
            days=TOTAL_DAYS,
            use_testnet=USE_TESTNET,
        )
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("\nUsing sample data for demonstration...")
        df = generate_sample_data(TOTAL_DAYS)
    
    # Step 3: Split data
    print("\n" + "=" * 60)
    print("STEP 3: Splitting Data")
    print("=" * 60)
    
    df_oos, df_is = split_data(df, in_sample_days=IN_SAMPLE_DAYS)
    
    print(f"✓ Out-of-Sample (Blind): {len(df_oos)} candles (~{OUT_OF_SAMPLE_DAYS} days)")
    print(f"  Date range: {df_oos['timestamp'].iloc[0].date()} to {df_oos['timestamp'].iloc[-1].date()}")
    print(f"✓ In-Sample (Training): {len(df_is)} candles (~{IN_SAMPLE_DAYS} days)")
    print(f"  Date range: {df_is['timestamp'].iloc[0].date()} to {df_is['timestamp'].iloc[-1].date()}")
    
    # Step 4: Run Out-of-Sample Test
    print("\n" + "=" * 60)
    print("STEP 4: Running Out-of-Sample Test")
    print("=" * 60)
    
    in_sample_result, oos_result = run_oos_test(df_is, df_oos, params)
    
    # Step 5: Run Sensitivity Test
    print("\n" + "=" * 60)
    print("STEP 5: Running Sensitivity Test")
    print("=" * 60)
    
    sensitivity_results = run_sensitivity_test(df_oos, params)
    
    # Step 6: Generate Report
    print_report(params, in_sample_result, oos_result, sensitivity_results)
    
    return in_sample_result, oos_result, sensitivity_results


if __name__ == "__main__":
    main()
