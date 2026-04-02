"""Parameter optimizer using grid search to find best strategy parameters."""

from dataclasses import dataclass, field
from itertools import product
from typing import Optional
import json

import pandas as pd
import numpy as np

from .backtester import Backtester, BacktestResult


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    
    best_params: dict
    best_result: BacktestResult
    all_results: list[BacktestResult] = field(default_factory=list)
    optimization_metric: str = "total_pnl"
    
    def to_config_json(self) -> dict:
        """Convert best parameters to config.json format."""
        p = self.best_params
        return {
            "trading": {
                "symbols": ["BTC", "ETH"],
                "leverage": p.get("leverage", 5),
                "margin_type": "isolated",
                "position_size_percent": p.get("position_size_percent", 5.0)
            },
            "strategy": {
                "daily_ma_type": p.get("ma_type", "SMA"),
                "daily_ma_period": p.get("ma_period", 50),
                "intraday_timeframe": "15m",
                "rsi_period": p.get("rsi_period", 14),
                "rsi_oversold": p.get("rsi_oversold", 30),
                "rsi_overbought": p.get("rsi_overbought", 70),
                "vwap_enabled": p.get("use_vwap", True)
            },
            "risk_management": {
                "stop_loss_percent": p.get("stop_loss_percent", 2.0),
                "take_profit_percent": p.get("take_profit_percent", 4.0),
                "use_atr_for_sl": False,
                "atr_period": 14,
                "atr_multiplier": 1.5
            },
            "bot": {
                "loop_interval_seconds": 60,
                "log_level": "INFO"
            }
        }
    
    def save_config(self, filepath: str = "config_optimized.json"):
        """Save best parameters to a config file."""
        config = self.to_config_json()
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)
        return filepath


class ParameterOptimizer:
    """Grid search optimizer for strategy parameters."""
    
    # Default parameter grids
    DEFAULT_GRIDS = {
        "ma_type": ["SMA", "EMA"],
        "ma_period": [20, 50, 100],
        "rsi_period": [7, 14, 21],
        "rsi_oversold": [25, 30, 35],
        "rsi_overbought": [65, 70, 75],
        "stop_loss_percent": [1.0, 1.5, 2.0, 2.5, 3.0],
        "take_profit_percent": [2.0, 3.0, 4.0, 5.0, 6.0],
        "use_vwap": [True, False],
    }
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size_percent: float = 5.0,
        leverage: int = 5,
    ):
        """Initialize the optimizer.
        
        Args:
            initial_capital: Starting capital for backtests.
            position_size_percent: Position size as % of capital.
            leverage: Leverage multiplier.
        """
        self.backtester = Backtester(
            initial_capital=initial_capital,
            position_size_percent=position_size_percent,
            leverage=leverage,
        )
    
    def optimize(
        self,
        df: pd.DataFrame,
        param_grids: Optional[dict] = None,
        metric: str = "total_pnl",
        min_trades: int = 5,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """Run grid search optimization.
        
        Args:
            df: DataFrame with OHLCV data.
            param_grids: Dict of parameter names to list of values to try.
                        Uses DEFAULT_GRIDS if not provided.
            metric: Metric to optimize ("total_pnl", "sharpe_ratio", "profit_factor", "win_rate").
            min_trades: Minimum number of trades required for valid result.
            show_progress: Print progress updates.
            
        Returns:
            OptimizationResult with best parameters and all results.
        """
        grids = param_grids or self.DEFAULT_GRIDS
        
        # Generate all parameter combinations
        param_names = list(grids.keys())
        param_values = list(grids.values())
        combinations = list(product(*param_values))
        
        total = len(combinations)
        if show_progress:
            print(f"Testing {total} parameter combinations...")
            print(f"Optimizing for: {metric}")
            print("-" * 50)
        
        all_results = []
        best_result = None
        best_metric_value = float("-inf")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            # Run backtest
            result = self.backtester.run(df, **params)
            
            # Skip if not enough trades
            if result.total_trades < min_trades:
                continue
            
            all_results.append(result)
            
            # Get metric value
            metric_value = getattr(result, metric, 0)
            
            # Check if best
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_result = result
            
            # Progress update
            if show_progress and (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{total} combinations tested...")
        
        if show_progress:
            print("-" * 50)
            print(f"Optimization complete! Tested {len(all_results)} valid combinations.")
        
        if best_result is None:
            raise ValueError("No valid results found. Try reducing min_trades or adjusting parameters.")
        
        return OptimizationResult(
            best_params=best_result.params,
            best_result=best_result,
            all_results=all_results,
            optimization_metric=metric,
        )
    
    def quick_optimize(
        self,
        df: pd.DataFrame,
        metric: str = "total_pnl",
        show_progress: bool = True,
    ) -> OptimizationResult:
        """Run a quick optimization with reduced parameter grid.
        
        Good for initial testing before full optimization.
        """
        quick_grids = {
            "ma_type": ["SMA", "EMA"],
            "ma_period": [20, 50],
            "rsi_period": [14],
            "rsi_oversold": [30],
            "rsi_overbought": [70],
            "stop_loss_percent": [1.5, 2.0, 2.5],
            "take_profit_percent": [3.0, 4.0, 5.0],
            "use_vwap": [True, False],
        }
        
        return self.optimize(df, quick_grids, metric, show_progress=show_progress)


def print_optimization_report(result: OptimizationResult):
    """Print a formatted report of optimization results."""
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION REPORT")
    print("=" * 60)
    
    print(f"\nOptimization Metric: {result.optimization_metric}")
    print(f"Total Combinations Tested: {len(result.all_results)}")
    
    print("\n" + "-" * 40)
    print("BEST PARAMETERS")
    print("-" * 40)
    
    for key, value in result.best_params.items():
        print(f"  {key:25} : {value}")
    
    print("\n" + "-" * 40)
    print("PERFORMANCE METRICS")
    print("-" * 40)
    
    r = result.best_result
    print(f"  Total Trades            : {r.total_trades}")
    print(f"  Win Rate                : {r.win_rate:.1f}%")
    print(f"  Total PnL               : ${r.total_pnl:,.2f} ({r.total_pnl_percent:+.1f}%)")
    print(f"  Average Trade           : ${r.average_pnl:,.2f}")
    print(f"  Average Win             : ${r.average_win:,.2f}")
    print(f"  Average Loss            : ${r.average_loss:,.2f}")
    print(f"  Profit Factor           : {r.profit_factor:.2f}")
    print(f"  Max Drawdown            : ${r.max_drawdown:,.2f} ({r.max_drawdown_percent:.1f}%)")
    print(f"  Sharpe Ratio            : {r.sharpe_ratio:.2f}")
    
    # Trade breakdown
    print("\n" + "-" * 40)
    print("TRADE BREAKDOWN")
    print("-" * 40)
    
    sl_exits = sum(1 for t in r.trades if t.exit_reason == "stop_loss")
    tp_exits = sum(1 for t in r.trades if t.exit_reason == "take_profit")
    other_exits = r.total_trades - sl_exits - tp_exits
    
    print(f"  Stop Loss Exits         : {sl_exits} ({sl_exits/r.total_trades*100:.1f}%)")
    print(f"  Take Profit Exits       : {tp_exits} ({tp_exits/r.total_trades*100:.1f}%)")
    print(f"  Other Exits             : {other_exits}")
    
    # Top 5 results
    print("\n" + "-" * 40)
    print("TOP 5 PARAMETER SETS")
    print("-" * 40)
    
    sorted_results = sorted(
        result.all_results,
        key=lambda x: getattr(x, result.optimization_metric, 0),
        reverse=True
    )[:5]
    
    for i, res in enumerate(sorted_results, 1):
        p = res.params
        metric_val = getattr(res, result.optimization_metric, 0)
        print(f"\n  #{i}: {result.optimization_metric}={metric_val:.2f}")
        print(f"      MA: {p['ma_type']}{p['ma_period']}, RSI: {p['rsi_period']} ({p['rsi_oversold']}/{p['rsi_overbought']})")
        print(f"      SL: {p['stop_loss_percent']}%, TP: {p['take_profit_percent']}%, VWAP: {p['use_vwap']}")
        print(f"      Trades: {res.total_trades}, Win Rate: {res.win_rate:.1f}%, PF: {res.profit_factor:.2f}")
    
    print("\n" + "=" * 60)
