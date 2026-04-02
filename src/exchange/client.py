"""Hyperliquid exchange client wrapper.

This module provides a clean interface to the Hyperliquid SDK for:
- Authentication via private key
- Account balance retrieval
- Historical candle data (OHLCV)
- Order management (to be extended)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from eth_account import Account
from eth_account.signers.local import LocalAccount
from hyperliquid.api import API
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils.constants import MAINNET_API_URL, TESTNET_API_URL

# Monkey-patch to fix broken spotMeta on testnet (perpetuals-only bot)
_original_post = API.post


def _patched_post(self, url_path: str, payload: dict = None):
    """Intercept spotMeta requests and return empty response."""
    if payload and payload.get("type") == "spotMeta":
        return {"universe": [], "tokens": []}
    return _original_post(self, url_path, payload)


API.post = _patched_post


@dataclass
class AccountBalance:
    """Account balance information."""

    account_value: float
    total_margin_used: float
    total_position_value: float
    withdrawable: float
    raw_usd: float


@dataclass
class Position:
    """Open position information."""

    symbol: str
    side: str  # "long" or "short"
    size: float
    entry_price: float
    leverage: int
    leverage_type: str  # "cross" or "isolated"
    unrealized_pnl: float
    liquidation_price: Optional[float]
    margin_used: float


class HyperliquidClient:
    """Wrapper for Hyperliquid SDK with simplified interface."""

    # Valid intervals for candle data
    VALID_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"]

    def __init__(
        self,
        private_key: str,
        wallet_address: str,
        is_testnet: bool = True,
    ):
        """Initialize the Hyperliquid client.

        Args:
            private_key: Wallet private key (without 0x prefix).
            wallet_address: Wallet address (with 0x prefix).
            is_testnet: Use testnet if True, mainnet if False.
        """
        self.wallet_address = wallet_address
        self.is_testnet = is_testnet
        self.base_url = TESTNET_API_URL if is_testnet else MAINNET_API_URL

        # Create wallet from private key
        if not private_key.startswith("0x"):
            private_key = f"0x{private_key}"
        self.wallet: LocalAccount = Account.from_key(private_key)

        # Initialize SDK components
        self.info = Info(base_url=self.base_url, skip_ws=True)
        self.exchange = Exchange(
            wallet=self.wallet,
            base_url=self.base_url,
        )

    def get_account_balance(self) -> AccountBalance:
        """Fetch the account balance and margin summary.

        Returns:
            AccountBalance with account value, margin used, and withdrawable amount.

        Raises:
            Exception: If API request fails.
        """
        state = self.info.user_state(self.wallet_address)

        margin_summary = state.get("marginSummary", {})
        cross_margin = state.get("crossMarginSummary", {})

        return AccountBalance(
            account_value=float(margin_summary.get("accountValue", 0)),
            total_margin_used=float(margin_summary.get("totalMarginUsed", 0)),
            total_position_value=float(margin_summary.get("totalNtlPos", 0)),
            withdrawable=float(state.get("withdrawable", 0)),
            raw_usd=float(cross_margin.get("totalRawUsd", 0)),
        )

    def get_open_positions(self) -> list[Position]:
        """Fetch all open positions.

        Returns:
            List of Position objects for all open positions.
        """
        state = self.info.user_state(self.wallet_address)
        positions = []

        for asset_pos in state.get("assetPositions", []):
            pos = asset_pos.get("position", {})
            size = float(pos.get("szi", 0))

            if size == 0:
                continue  # Skip empty positions

            leverage_info = pos.get("leverage", {})
            liq_price = pos.get("liquidationPx")

            positions.append(
                Position(
                    symbol=pos.get("coin", ""),
                    side="long" if size > 0 else "short",
                    size=abs(size),
                    entry_price=float(pos.get("entryPx", 0) or 0),
                    leverage=int(leverage_info.get("value", 1)),
                    leverage_type=leverage_info.get("type", "cross"),
                    unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                    liquidation_price=float(liq_price) if liq_price else None,
                    margin_used=float(pos.get("marginUsed", 0)),
                )
            )

        return positions

    def get_candles(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV candle data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC", "ETH").
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            limit: Maximum number of candles to return (default 100).
            start_time: Start time for candles (default: calculated from limit and interval).
            end_time: End time for candles (default: now).

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Raises:
            ValueError: If interval is not valid.
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Must be one of: {self.VALID_INTERVALS}")

        # Calculate time range
        if end_time is None:
            end_time = datetime.now()

        if start_time is None:
            # Calculate start time based on interval and limit
            interval_seconds = self._interval_to_seconds(interval)
            start_time = end_time - timedelta(seconds=interval_seconds * limit)

        # Convert to milliseconds for API
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        # Fetch candles from API
        candles = self.info.candles_snapshot(
            name=symbol,
            interval=interval,
            startTime=start_ms,
            endTime=end_ms,
        )

        if not candles:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Rename columns: t=timestamp, o=open, h=high, l=low, c=close, v=volume
        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )

        # Select and convert columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Sort by timestamp and reset index
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df.tail(limit)

    def get_current_price(self, symbol: str) -> float:
        """Get the current mid price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC", "ETH").

        Returns:
            Current mid price as float.
        """
        all_mids = self.info.all_mids()
        return float(all_mids.get(symbol, 0))

    def set_leverage(self, symbol: str, leverage: int, is_cross: bool = False) -> dict:
        """Set leverage for a symbol.

        Args:
            symbol: Trading pair symbol.
            leverage: Leverage value (e.g., 5, 10, 20).
            is_cross: True for cross margin, False for isolated margin.

        Returns:
            API response dict.
        """
        return self.exchange.update_leverage(
            leverage=leverage,
            name=symbol,
            is_cross=is_cross,
        )

    def _interval_to_seconds(self, interval: str) -> int:
        """Convert interval string to seconds."""
        multipliers = {
            "m": 60,
            "h": 3600,
            "d": 86400,
            "w": 604800,
            "M": 2592000,  # ~30 days
        }

        unit = interval[-1]
        value = int(interval[:-1])

        return value * multipliers.get(unit, 60)

    def is_connected(self) -> bool:
        """Check if the client can connect to the API.

        Returns:
            True if connection is successful.
        """
        try:
            self.info.meta()
            return True
        except Exception:
            return False
