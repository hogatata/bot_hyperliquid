"""Backtesting engine for simulating trading strategies on historical data.

Supports advanced features:
- Trailing Stop Loss: SL trails price by ATR multiplier when in profit
- Volatility Filter: Skips entries when ATR is below threshold
- Macro Filters simulation for regime detection
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.indicators import add_sma, add_ema, add_rsi, add_vwap, add_atr


class Side(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Record of a single trade."""
    
    entry_time: datetime
    exit_time: Optional[datetime]
    side: Side
    entry_price: float
    exit_price: Optional[float]
    size: float
    stop_loss: float
    take_profit: float
    atr_at_entry: float = 0.0  # ATR value at entry (for trailing)
    highest_price: float = 0.0  # Track highest price (for LONG trailing)
    lowest_price: float = 0.0   # Track lowest price (for SHORT trailing)
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""  # "stop_loss", "take_profit", "trailing_stop", "signal", "end_of_data"
    was_filtered: bool = False  # Track if trade was skipped by filter


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    
    # Parameters used
    params: dict
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    average_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    profit_factor: float = 0.0  # gross profit / gross loss
    sharpe_ratio: float = 0.0
    
    # Advanced metrics
    trailing_stop_exits: int = 0
    filtered_signals: int = 0  # Signals blocked by volatility filter
    
    # Trade list
    trades: list = field(default_factory=list)
    
    # Equity curve
    equity_curve: list = field(default_factory=list)


class Backtester:
    """Backtesting engine that simulates the trading strategy on historical data."""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size_percent: float = 5.0,
        leverage: int = 5,
        commission_percent: float = 0.05,  # 0.05% per trade (taker fee)
    ):
        """Initialize the backtester.
        
        Args:
            initial_capital: Starting capital in USD.
            position_size_percent: Percentage of capital per trade.
            leverage: Leverage multiplier.
            commission_percent: Commission per trade as percentage.
        """
        self.initial_capital = initial_capital
        self.position_size_percent = position_size_percent
        self.leverage = leverage
        self.commission_percent = commission_percent
    
    def run(
        self,
        df: pd.DataFrame,
        ma_type: str = "SMA",
        ma_period: int = 50,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        stop_loss_percent: float = 2.0,
        take_profit_percent: float = 4.0,
        use_vwap: bool = True,
        # New advanced parameters
        trailing_stop_enabled: bool = False,
        trailing_atr_multiplier: float = 1.5,
        volatility_filter_enabled: bool = False,
        volatility_atr_period: int = 14,
        volatility_lookback: int = 20,
        volatility_threshold: float = 0.5,
        use_atr_for_sl: bool = False,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0,
    ) -> BacktestResult:
        """Run backtest with given parameters.
        
        Args:
            df: DataFrame with OHLCV data.
            ma_type: Moving average type ("SMA" or "EMA").
            ma_period: Period for moving average.
            rsi_period: Period for RSI.
            rsi_oversold: RSI oversold threshold.
            rsi_overbought: RSI overbought threshold.
            stop_loss_percent: Stop loss as percentage.
            take_profit_percent: Take profit as percentage.
            use_vwap: Whether to use VWAP in signals.
            trailing_stop_enabled: Enable trailing stop loss.
            trailing_atr_multiplier: ATR multiplier for trailing distance.
            volatility_filter_enabled: Enable volatility filter.
            volatility_atr_period: ATR period for volatility calculation.
            volatility_lookback: Periods to calculate average ATR.
            volatility_threshold: Ratio threshold (current ATR / avg ATR).
            use_atr_for_sl: Use ATR-based SL instead of percentage.
            atr_sl_multiplier: ATR multiplier for stop loss.
            atr_tp_multiplier: ATR multiplier for take profit.
            
        Returns:
            BacktestResult with all metrics and trades.
        """
        # Store parameters
        params = {
            "ma_type": ma_type,
            "ma_period": ma_period,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "stop_loss_percent": stop_loss_percent,
            "take_profit_percent": take_profit_percent,
            "use_vwap": use_vwap,
            "leverage": self.leverage,
            "position_size_percent": self.position_size_percent,
            # Advanced params
            "trailing_stop_enabled": trailing_stop_enabled,
            "trailing_atr_multiplier": trailing_atr_multiplier,
            "volatility_filter_enabled": volatility_filter_enabled,
            "volatility_threshold": volatility_threshold,
            "use_atr_for_sl": use_atr_for_sl,
            "atr_sl_multiplier": atr_sl_multiplier,
            "atr_tp_multiplier": atr_tp_multiplier,
        }
        
        # Add indicators
        df = df.copy()
        ma_column = f"{ma_type.lower()}_{ma_period}"
        rsi_column = f"rsi_{rsi_period}"
        
        if ma_type.upper() == "SMA":
            df = add_sma(df, period=ma_period, column_name=ma_column)
        else:
            df = add_ema(df, period=ma_period, column_name=ma_column)
        
        df = add_rsi(df, period=rsi_period, column_name=rsi_column)
        
        if use_vwap:
            df = add_vwap(df)
        
        # Add ATR for trailing stop and volatility filter
        atr_column = f"atr_{volatility_atr_period}"
        df = add_atr(df, period=volatility_atr_period, column_name=atr_column)
        
        # Initialize state
        capital = self.initial_capital
        equity_curve = [capital]
        trades = []
        current_position: Optional[Trade] = None
        filtered_signals = 0  # Track filtered entries
        
        # Skip initial rows where indicators are NaN
        start_idx = max(ma_period, rsi_period) + 1
        
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            prev_prev_row = df.iloc[i - 2] if i >= 2 else prev_row
            
            current_price = row["close"]
            current_time = row["timestamp"] if "timestamp" in df.columns else i
            current_atr = row[atr_column] if not pd.isna(row[atr_column]) else 0
            
            # Check if we have a position
            if current_position is not None:
                # Update trailing stop if enabled
                if trailing_stop_enabled and current_position.atr_at_entry > 0:
                    trail_distance = current_position.atr_at_entry * trailing_atr_multiplier
                    
                    if current_position.side == Side.LONG:
                        # Track highest price
                        if row["high"] > current_position.highest_price:
                            current_position.highest_price = row["high"]
                            # Calculate new trailing SL
                            new_sl = current_position.highest_price - trail_distance
                            # Only move SL up (never down)
                            if new_sl > current_position.stop_loss:
                                current_position.stop_loss = new_sl
                    else:  # SHORT
                        # Track lowest price
                        if row["low"] < current_position.lowest_price:
                            current_position.lowest_price = row["low"]
                            # Calculate new trailing SL
                            new_sl = current_position.lowest_price + trail_distance
                            # Only move SL down (never up)
                            if new_sl < current_position.stop_loss:
                                current_position.stop_loss = new_sl
                
                # Check for stop loss or take profit
                if current_position.side == Side.LONG:
                    # Check SL (price went below)
                    if row["low"] <= current_position.stop_loss:
                        exit_price = current_position.stop_loss
                        exit_reason = "trailing_stop" if trailing_stop_enabled else "stop_loss"
                        current_position = self._close_position(
                            current_position, exit_price, current_time, exit_reason
                        )
                        capital += current_position.pnl
                        trades.append(current_position)
                        current_position = None
                    # Check TP (price went above)
                    elif row["high"] >= current_position.take_profit:
                        exit_price = current_position.take_profit
                        current_position = self._close_position(
                            current_position, exit_price, current_time, "take_profit"
                        )
                        capital += current_position.pnl
                        trades.append(current_position)
                        current_position = None
                else:  # SHORT
                    # Check SL (price went above)
                    if row["high"] >= current_position.stop_loss:
                        exit_price = current_position.stop_loss
                        exit_reason = "trailing_stop" if trailing_stop_enabled else "stop_loss"
                        current_position = self._close_position(
                            current_position, exit_price, current_time, exit_reason
                        )
                        capital += current_position.pnl
                        trades.append(current_position)
                        current_position = None
                    # Check TP (price went below)
                    elif row["low"] <= current_position.take_profit:
                        exit_price = current_position.take_profit
                        current_position = self._close_position(
                            current_position, exit_price, current_time, "take_profit"
                        )
                        capital += current_position.pnl
                        trades.append(current_position)
                        current_position = None
            
            # Check for entry signals (only if no position)
            if current_position is None:
                signal = self._check_signal(
                    row, prev_row, prev_prev_row,
                    ma_column, rsi_column,
                    rsi_oversold, rsi_overbought,
                    use_vwap
                )
                
                if signal is not None:
                    # Apply volatility filter if enabled
                    if volatility_filter_enabled:
                        avg_atr = df[atr_column].iloc[max(0, i-volatility_lookback):i].mean()
                        if avg_atr > 0 and current_atr < (avg_atr * volatility_threshold):
                            # Skip entry - market too choppy
                            filtered_signals += 1
                            signal = None
                
                if signal is not None:
                    # Calculate position size
                    position_value = (capital * self.position_size_percent / 100) * self.leverage
                    size = position_value / current_price
                    
                    # Calculate SL/TP (ATR-based or percentage-based)
                    if use_atr_for_sl and current_atr > 0:
                        if signal == Side.LONG:
                            sl = current_price - (current_atr * atr_sl_multiplier)
                            tp = current_price + (current_atr * atr_tp_multiplier)
                        else:
                            sl = current_price + (current_atr * atr_sl_multiplier)
                            tp = current_price - (current_atr * atr_tp_multiplier)
                    else:
                        if signal == Side.LONG:
                            sl = current_price * (1 - stop_loss_percent / 100)
                            tp = current_price * (1 + take_profit_percent / 100)
                        else:
                            sl = current_price * (1 + stop_loss_percent / 100)
                            tp = current_price * (1 - take_profit_percent / 100)
                    
                    # Open position with ATR tracking for trailing
                    current_position = Trade(
                        entry_time=current_time,
                        exit_time=None,
                        side=signal,
                        entry_price=current_price,
                        exit_price=None,
                        size=size,
                        stop_loss=sl,
                        take_profit=tp,
                        atr_at_entry=current_atr,
                        highest_price=current_price,
                        lowest_price=current_price,
                    )
            
            # Update equity curve
            if current_position is not None:
                # Mark to market
                if current_position.side == Side.LONG:
                    unrealized = (current_price - current_position.entry_price) * current_position.size
                else:
                    unrealized = (current_position.entry_price - current_price) * current_position.size
                equity_curve.append(capital + unrealized)
            else:
                equity_curve.append(capital)
        
        # Close any remaining position at end
        if current_position is not None:
            last_price = df.iloc[-1]["close"]
            last_time = df.iloc[-1]["timestamp"] if "timestamp" in df.columns else len(df) - 1
            current_position = self._close_position(
                current_position, last_price, last_time, "end_of_data"
            )
            capital += current_position.pnl
            trades.append(current_position)
            equity_curve[-1] = capital
        
        # Calculate metrics
        return self._calculate_metrics(params, trades, equity_curve, filtered_signals)
    
    def _check_signal(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        prev_prev_row: pd.Series,
        ma_column: str,
        rsi_column: str,
        rsi_oversold: int,
        rsi_overbought: int,
        use_vwap: bool,
    ) -> Optional[Side]:
        """Check for entry signal."""
        
        current_price = row["close"]
        ma_value = row[ma_column]
        rsi_value = row[rsi_column]
        prev_rsi = prev_row[rsi_column]
        prev_prev_rsi = prev_prev_row[rsi_column]
        
        if pd.isna(ma_value) or pd.isna(rsi_value):
            return None
        
        # Determine trend
        is_bullish = current_price > ma_value
        is_bearish = current_price < ma_value
        
        # Check RSI conditions
        rsi_exiting_oversold = (
            (prev_rsi < rsi_oversold or prev_prev_rsi < rsi_oversold) and
            rsi_value >= rsi_oversold
        )
        rsi_exiting_overbought = (
            (prev_rsi > rsi_overbought or prev_prev_rsi > rsi_overbought) and
            rsi_value <= rsi_overbought
        )
        
        # Check VWAP conditions
        vwap_cross_up = True
        vwap_cross_down = True
        
        if use_vwap and "vwap" in row.index:
            vwap = row["vwap"]
            prev_vwap = prev_row["vwap"]
            prev_close = prev_row["close"]
            
            if not pd.isna(vwap) and not pd.isna(prev_vwap):
                vwap_cross_up = prev_close < prev_vwap and current_price > vwap
                vwap_cross_down = prev_close > prev_vwap and current_price < vwap
        
        # Generate signals
        if is_bullish and rsi_exiting_oversold:
            if not use_vwap or vwap_cross_up:
                return Side.LONG
        
        if is_bearish and rsi_exiting_overbought:
            if not use_vwap or vwap_cross_down:
                return Side.SHORT
        
        return None
    
    def _close_position(
        self,
        position: Trade,
        exit_price: float,
        exit_time: datetime,
        reason: str,
    ) -> Trade:
        """Close a position and calculate PnL."""
        
        position.exit_price = exit_price
        position.exit_time = exit_time
        position.exit_reason = reason
        
        # Calculate PnL
        if position.side == Side.LONG:
            gross_pnl = (exit_price - position.entry_price) * position.size
        else:
            gross_pnl = (position.entry_price - exit_price) * position.size
        
        # Subtract commission (entry + exit)
        commission = (position.entry_price * position.size * self.commission_percent / 100) * 2
        position.pnl = gross_pnl - commission
        
        # Calculate percentage
        position_value = position.entry_price * position.size
        position.pnl_percent = (position.pnl / position_value) * 100 * self.leverage
        
        return position
    
    def _calculate_metrics(
        self,
        params: dict,
        trades: list[Trade],
        equity_curve: list[float],
        filtered_signals: int = 0,
    ) -> BacktestResult:
        """Calculate all backtest metrics."""
        
        result = BacktestResult(params=params, trades=trades, equity_curve=equity_curve)
        result.filtered_signals = filtered_signals
        
        if not trades:
            return result
        
        # Basic counts
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in trades if t.pnl < 0)
        result.win_rate = result.winning_trades / result.total_trades * 100
        
        # Count trailing stop exits
        result.trailing_stop_exits = sum(1 for t in trades if t.exit_reason == "trailing_stop")
        
        # PnL metrics
        pnls = [t.pnl for t in trades]
        result.total_pnl = sum(pnls)
        result.total_pnl_percent = (result.total_pnl / self.initial_capital) * 100
        result.average_pnl = np.mean(pnls)
        
        winning_pnls = [t.pnl for t in trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in trades if t.pnl < 0]
        
        result.average_win = np.mean(winning_pnls) if winning_pnls else 0
        result.average_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Max drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity)
        drawdown_percent = drawdown / peak * 100
        
        result.max_drawdown = np.max(drawdown)
        result.max_drawdown_percent = np.max(drawdown_percent)
        
        # Sharpe ratio (simplified, assuming daily returns)
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return result
