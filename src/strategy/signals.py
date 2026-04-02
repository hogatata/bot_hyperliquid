"""Trading signals combining indicators for entry/exit logic.

This module implements the strategy logic:
- Daily trend detection (price vs SMA)
- Intraday entry signals (VWAP crossover + RSI)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from .indicators import (
    add_all_indicators,
    add_ema,
    add_rsi,
    add_sma,
    add_vwap,
    get_trend,
    is_price_crossing_vwap_down,
    is_price_crossing_vwap_up,
    is_rsi_exiting_overbought,
    is_rsi_exiting_oversold,
)


class Signal(Enum):
    """Trading signal type."""

    LONG = "long"
    SHORT = "short"
    NO_SIGNAL = "no_signal"


@dataclass
class SignalResult:
    """Result of signal analysis."""

    signal: Signal
    trend: str  # "bullish", "bearish", "neutral"
    rsi_value: float
    rsi_zone: str  # "oversold", "overbought", "neutral"
    vwap_cross: str  # "up", "down", "none"
    current_price: float
    ma_value: float
    vwap_value: float
    reason: str  # Human-readable explanation


class SignalGenerator:
    """Generates trading signals based on multi-timeframe analysis.

    Strategy Logic:
    - Daily Trend: Price > SMA = Bullish (only look for LONG)
                   Price < SMA = Bearish (only look for SHORT)

    - Entry Signal (LONG):
      1. Bullish daily trend (price > daily SMA)
      2. Price crosses VWAP upward on intraday
      3. RSI was oversold (<30) and is now rising

    - Entry Signal (SHORT):
      1. Bearish daily trend (price < daily SMA)
      2. Price crosses VWAP downward on intraday
      3. RSI was overbought (>70) and is now falling
    """

    def __init__(
        self,
        daily_ma_type: str = "SMA",
        daily_ma_period: int = 50,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        vwap_enabled: bool = True,
        rsi_lookback: int = 3,
    ):
        """Initialize the signal generator.

        Args:
            daily_ma_type: Type of MA for trend ("SMA" or "EMA").
            daily_ma_period: Period for daily MA.
            rsi_period: Period for RSI calculation.
            rsi_oversold: RSI level considered oversold.
            rsi_overbought: RSI level considered overbought.
            vwap_enabled: Whether to use VWAP in signals.
            rsi_lookback: Candles to look back for RSI zone exit.
        """
        self.daily_ma_type = daily_ma_type.upper()
        self.daily_ma_period = daily_ma_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.vwap_enabled = vwap_enabled
        self.rsi_lookback = rsi_lookback

        self.ma_column = f"{self.daily_ma_type.lower()}_{daily_ma_period}"
        self.rsi_column = f"rsi_{rsi_period}"

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators to the DataFrame.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with indicator columns added.
        """
        df = df.copy()

        # Add Moving Average for trend
        if self.daily_ma_type == "SMA":
            df = add_sma(df, period=self.daily_ma_period, column_name=self.ma_column)
        else:
            df = add_ema(df, period=self.daily_ma_period, column_name=self.ma_column)

        # Add RSI
        df = add_rsi(df, period=self.rsi_period, column_name=self.rsi_column)

        # Add VWAP if enabled
        if self.vwap_enabled:
            df = add_vwap(df)

        return df

    def analyze(self, df: pd.DataFrame) -> SignalResult:
        """Analyze the DataFrame and generate a trading signal.

        Args:
            df: DataFrame with OHLCV data (indicators will be added if missing).

        Returns:
            SignalResult with signal type and analysis details.
        """
        # Add indicators if not present
        if self.ma_column not in df.columns:
            df = self.add_indicators(df)

        if df.empty or len(df) < max(self.daily_ma_period, self.rsi_period):
            return SignalResult(
                signal=Signal.NO_SIGNAL,
                trend="neutral",
                rsi_value=50.0,
                rsi_zone="neutral",
                vwap_cross="none",
                current_price=0.0,
                ma_value=0.0,
                vwap_value=0.0,
                reason="Insufficient data for analysis",
            )

        latest = df.iloc[-1]
        current_price = latest["close"]
        ma_value = latest[self.ma_column]
        rsi_value = latest[self.rsi_column]
        vwap_value = latest.get("vwap", current_price)

        # Handle NaN values
        if pd.isna(ma_value) or pd.isna(rsi_value):
            return SignalResult(
                signal=Signal.NO_SIGNAL,
                trend="neutral",
                rsi_value=rsi_value if not pd.isna(rsi_value) else 50.0,
                rsi_zone="neutral",
                vwap_cross="none",
                current_price=current_price,
                ma_value=ma_value if not pd.isna(ma_value) else 0.0,
                vwap_value=vwap_value if not pd.isna(vwap_value) else 0.0,
                reason="Indicators not yet calculated (NaN)",
            )

        # Determine trend
        trend = get_trend(df, self.ma_column)

        # Determine RSI zone
        if rsi_value < self.rsi_oversold:
            rsi_zone = "oversold"
        elif rsi_value > self.rsi_overbought:
            rsi_zone = "overbought"
        else:
            rsi_zone = "neutral"

        # Check VWAP crossover
        vwap_cross = "none"
        if self.vwap_enabled and "vwap" in df.columns:
            if is_price_crossing_vwap_up(df):
                vwap_cross = "up"
            elif is_price_crossing_vwap_down(df):
                vwap_cross = "down"

        # Check RSI zone exits
        rsi_exiting_oversold = is_rsi_exiting_oversold(
            df, self.rsi_column, self.rsi_oversold, self.rsi_lookback
        )
        rsi_exiting_overbought = is_rsi_exiting_overbought(
            df, self.rsi_column, self.rsi_overbought, self.rsi_lookback
        )

        # Generate signal
        signal = Signal.NO_SIGNAL
        reason = "No entry conditions met"

        # LONG signal conditions
        if trend == "bullish":
            if self.vwap_enabled:
                if vwap_cross == "up" and rsi_exiting_oversold:
                    signal = Signal.LONG
                    reason = "Bullish trend + VWAP cross up + RSI exiting oversold"
                elif vwap_cross == "up":
                    reason = "Bullish trend + VWAP cross up (waiting for RSI)"
                elif rsi_exiting_oversold:
                    reason = "Bullish trend + RSI exiting oversold (waiting for VWAP)"
                else:
                    reason = "Bullish trend (waiting for entry signal)"
            else:
                # Without VWAP, just use RSI
                if rsi_exiting_oversold:
                    signal = Signal.LONG
                    reason = "Bullish trend + RSI exiting oversold"
                else:
                    reason = "Bullish trend (waiting for RSI signal)"

        # SHORT signal conditions
        elif trend == "bearish":
            if self.vwap_enabled:
                if vwap_cross == "down" and rsi_exiting_overbought:
                    signal = Signal.SHORT
                    reason = "Bearish trend + VWAP cross down + RSI exiting overbought"
                elif vwap_cross == "down":
                    reason = "Bearish trend + VWAP cross down (waiting for RSI)"
                elif rsi_exiting_overbought:
                    reason = "Bearish trend + RSI exiting overbought (waiting for VWAP)"
                else:
                    reason = "Bearish trend (waiting for entry signal)"
            else:
                # Without VWAP, just use RSI
                if rsi_exiting_overbought:
                    signal = Signal.SHORT
                    reason = "Bearish trend + RSI exiting overbought"
                else:
                    reason = "Bearish trend (waiting for RSI signal)"

        else:  # neutral trend
            reason = "Neutral trend (no clear direction)"

        return SignalResult(
            signal=signal,
            trend=trend,
            rsi_value=round(rsi_value, 2),
            rsi_zone=rsi_zone,
            vwap_cross=vwap_cross,
            current_price=round(current_price, 2),
            ma_value=round(ma_value, 2),
            vwap_value=round(vwap_value, 2) if not pd.isna(vwap_value) else 0.0,
            reason=reason,
        )
