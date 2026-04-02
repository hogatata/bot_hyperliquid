"""Technical indicators module - modular design for easy extension.

This module provides indicator calculation functions using pandas-ta.
Each function takes an OHLCV DataFrame and returns it with indicator columns added.

Supported indicators:
- SMA (Simple Moving Average) - for daily trend detection
- EMA (Exponential Moving Average) - alternative trend indicator
- VWAP (Volume Weighted Average Price) - intraday value zones
- RSI (Relative Strength Index) - overbought/oversold detection
- ATR (Average True Range) - volatility for dynamic SL/TP (future use)
"""

import pandas as pd
import pandas_ta as ta


def add_sma(
    df: pd.DataFrame,
    period: int = 50,
    column: str = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    """Add Simple Moving Average to DataFrame.

    Used for daily trend detection. If price > SMA, trend is bullish.

    Args:
        df: DataFrame with OHLCV data.
        period: SMA period (default 50 for daily trend).
        column: Column to calculate SMA on (default "close").
        column_name: Custom name for the new column (default "sma_{period}").

    Returns:
        DataFrame with SMA column added.
    """
    df = df.copy()
    col_name = column_name or f"sma_{period}"
    df[col_name] = ta.sma(df[column], length=period)
    return df


def add_ema(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    """Add Exponential Moving Average to DataFrame.

    Alternative to SMA, gives more weight to recent prices.

    Args:
        df: DataFrame with OHLCV data.
        period: EMA period (default 20).
        column: Column to calculate EMA on (default "close").
        column_name: Custom name for the new column (default "ema_{period}").

    Returns:
        DataFrame with EMA column added.
    """
    df = df.copy()
    col_name = column_name or f"ema_{period}"
    df[col_name] = ta.ema(df[column], length=period)
    return df


def add_vwap(
    df: pd.DataFrame,
    anchor: str | None = None,
    column_name: str = "vwap",
) -> pd.DataFrame:
    """Add Volume Weighted Average Price to DataFrame.

    VWAP represents the average price weighted by volume.
    Used as intraday value zone - price above VWAP is bullish, below is bearish.

    Args:
        df: DataFrame with high, low, close, volume columns.
            Must have a 'timestamp' column with datetime values.
        anchor: VWAP reset anchor (None for no reset, "D" for daily, etc.).
        column_name: Name for the VWAP column (default "vwap").

    Returns:
        DataFrame with VWAP column added.

    Note:
        For intraday charts without date boundaries, set anchor=None
        to calculate VWAP across the entire dataset.
    """
    df = df.copy()

    # VWAP requires DatetimeIndex - set timestamp as index temporarily
    if "timestamp" in df.columns:
        df_indexed = df.set_index("timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        df_indexed = df
    else:
        # Fallback: calculate VWAP manually without anchoring
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        df[column_name] = cumulative_tp_vol / cumulative_vol
        return df

    # VWAP requires high, low, close, volume
    vwap_series = ta.vwap(
        high=df_indexed["high"],
        low=df_indexed["low"],
        close=df_indexed["close"],
        volume=df_indexed["volume"],
        anchor=anchor,
    )

    # Reset index to get timestamp back as column
    if "timestamp" in df.columns:
        df[column_name] = vwap_series.values
    else:
        df[column_name] = vwap_series

    return df


def add_rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
    column_name: str | None = None,
) -> pd.DataFrame:
    """Add Relative Strength Index to DataFrame.

    RSI oscillates between 0-100.
    - RSI < 30: Oversold (potential buy signal)
    - RSI > 70: Overbought (potential sell signal)

    Args:
        df: DataFrame with OHLCV data.
        period: RSI period (default 14).
        column: Column to calculate RSI on (default "close").
        column_name: Custom name for the new column (default "rsi_{period}").

    Returns:
        DataFrame with RSI column added.
    """
    df = df.copy()
    col_name = column_name or f"rsi_{period}"
    df[col_name] = ta.rsi(df[column], length=period)
    return df


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
    column_name: str | None = None,
) -> pd.DataFrame:
    """Add Average True Range to DataFrame.

    ATR measures volatility. Used for dynamic SL/TP calculation.
    SL = entry_price - (ATR * multiplier)

    Args:
        df: DataFrame with high, low, close columns.
        period: ATR period (default 14).
        column_name: Custom name for the new column (default "atr_{period}").

    Returns:
        DataFrame with ATR column added.
    """
    df = df.copy()
    col_name = column_name or f"atr_{period}"
    df[col_name] = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=period,
    )
    return df


def add_all_indicators(
    df: pd.DataFrame,
    sma_period: int = 50,
    rsi_period: int = 14,
    atr_period: int = 14,
    include_vwap: bool = True,
    include_atr: bool = False,
) -> pd.DataFrame:
    """Add all configured indicators to DataFrame.

    Convenience function to add multiple indicators at once.

    Args:
        df: DataFrame with OHLCV data.
        sma_period: Period for SMA (default 50).
        rsi_period: Period for RSI (default 14).
        atr_period: Period for ATR (default 14).
        include_vwap: Whether to include VWAP (default True).
        include_atr: Whether to include ATR (default False).

    Returns:
        DataFrame with all indicator columns added.
    """
    df = add_sma(df, period=sma_period)
    df = add_rsi(df, period=rsi_period)

    if include_vwap:
        df = add_vwap(df)

    if include_atr:
        df = add_atr(df, period=atr_period)

    return df


def get_trend(df: pd.DataFrame, ma_column: str = "sma_50") -> str:
    """Determine the current trend based on price vs moving average.

    Args:
        df: DataFrame with close price and MA column.
        ma_column: Name of the moving average column.

    Returns:
        "bullish" if close > MA, "bearish" if close < MA, "neutral" if equal.
    """
    if df.empty or ma_column not in df.columns:
        return "neutral"

    latest = df.iloc[-1]
    close_price = latest["close"]
    ma_value = latest[ma_column]

    if pd.isna(ma_value):
        return "neutral"

    if close_price > ma_value:
        return "bullish"
    elif close_price < ma_value:
        return "bearish"
    else:
        return "neutral"


def get_rsi_zone(df: pd.DataFrame, rsi_column: str = "rsi_14", oversold: int = 30, overbought: int = 70) -> str:
    """Determine the RSI zone.

    Args:
        df: DataFrame with RSI column.
        rsi_column: Name of the RSI column.
        oversold: RSI level considered oversold (default 30).
        overbought: RSI level considered overbought (default 70).

    Returns:
        "oversold", "overbought", or "neutral".
    """
    if df.empty or rsi_column not in df.columns:
        return "neutral"

    rsi_value = df.iloc[-1][rsi_column]

    if pd.isna(rsi_value):
        return "neutral"

    if rsi_value < oversold:
        return "oversold"
    elif rsi_value > overbought:
        return "overbought"
    else:
        return "neutral"


def is_rsi_exiting_oversold(
    df: pd.DataFrame,
    rsi_column: str = "rsi_14",
    oversold: int = 30,
    lookback: int = 3,
) -> bool:
    """Check if RSI is exiting the oversold zone (rising from below threshold).

    This is a bullish signal: RSI was below oversold, now rising above it.

    Args:
        df: DataFrame with RSI column.
        rsi_column: Name of the RSI column.
        oversold: RSI level considered oversold (default 30).
        lookback: Number of candles to look back (default 3).

    Returns:
        True if RSI was oversold and is now rising above threshold.
    """
    if df.empty or rsi_column not in df.columns or len(df) < lookback:
        return False

    recent = df.tail(lookback)[rsi_column].values

    # Check: was below oversold in recent past, now above
    was_oversold = any(v < oversold for v in recent[:-1] if not pd.isna(v))
    current_rsi = recent[-1]

    if pd.isna(current_rsi):
        return False

    return was_oversold and current_rsi >= oversold


def is_rsi_exiting_overbought(
    df: pd.DataFrame,
    rsi_column: str = "rsi_14",
    overbought: int = 70,
    lookback: int = 3,
) -> bool:
    """Check if RSI is exiting the overbought zone (falling from above threshold).

    This is a bearish signal: RSI was above overbought, now falling below it.

    Args:
        df: DataFrame with RSI column.
        rsi_column: Name of the RSI column.
        overbought: RSI level considered overbought (default 70).
        lookback: Number of candles to look back (default 3).

    Returns:
        True if RSI was overbought and is now falling below threshold.
    """
    if df.empty or rsi_column not in df.columns or len(df) < lookback:
        return False

    recent = df.tail(lookback)[rsi_column].values

    # Check: was above overbought in recent past, now below
    was_overbought = any(v > overbought for v in recent[:-1] if not pd.isna(v))
    current_rsi = recent[-1]

    if pd.isna(current_rsi):
        return False

    return was_overbought and current_rsi <= overbought


def is_price_crossing_vwap_up(
    df: pd.DataFrame,
    vwap_column: str = "vwap",
    lookback: int = 2,
) -> bool:
    """Check if price is crossing VWAP upwards.

    Bullish signal: previous close < VWAP, current close > VWAP.

    Args:
        df: DataFrame with close and VWAP columns.
        vwap_column: Name of the VWAP column.
        lookback: Number of candles to check (default 2).

    Returns:
        True if price crossed VWAP from below to above.
    """
    if df.empty or vwap_column not in df.columns or len(df) < lookback:
        return False

    recent = df.tail(lookback)
    prev_close = recent.iloc[-2]["close"]
    prev_vwap = recent.iloc[-2][vwap_column]
    curr_close = recent.iloc[-1]["close"]
    curr_vwap = recent.iloc[-1][vwap_column]

    if any(pd.isna(v) for v in [prev_close, prev_vwap, curr_close, curr_vwap]):
        return False

    return prev_close < prev_vwap and curr_close > curr_vwap


def is_price_crossing_vwap_down(
    df: pd.DataFrame,
    vwap_column: str = "vwap",
    lookback: int = 2,
) -> bool:
    """Check if price is crossing VWAP downwards.

    Bearish signal: previous close > VWAP, current close < VWAP.

    Args:
        df: DataFrame with close and VWAP columns.
        vwap_column: Name of the VWAP column.
        lookback: Number of candles to check (default 2).

    Returns:
        True if price crossed VWAP from above to below.
    """
    if df.empty or vwap_column not in df.columns or len(df) < lookback:
        return False

    recent = df.tail(lookback)
    prev_close = recent.iloc[-2]["close"]
    prev_vwap = recent.iloc[-2][vwap_column]
    curr_close = recent.iloc[-1]["close"]
    curr_vwap = recent.iloc[-1][vwap_column]

    if any(pd.isna(v) for v in [prev_close, prev_vwap, curr_close, curr_vwap]):
        return False

    return prev_close > prev_vwap and curr_close < curr_vwap
