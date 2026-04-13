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
    """Account balance information.
    
    Aggregates both Spot wallet and Perpetuals (Clearinghouse) wallet
    to match the Hyperliquid Web UI total.
    """

    account_value: float  # Total: Perps account value + Spot USDC
    total_margin_used: float  # Margin locked in positions
    total_position_value: float  # Total notional position value
    withdrawable: float  # Free margin available (Perps + Spot)
    raw_usd: float  # Raw USDC in Perpetuals clearinghouse
    spot_usdc: float = 0.0  # USDC in Spot wallet


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
            # Ensure trading actions (market_close/order/cancel) target the same account
            # used for state queries (self.wallet_address), even if signer differs.
            account_address=self.wallet_address,
        )

    def get_account_balance(self) -> AccountBalance:
        """Fetch the TOTAL account balance aggregating Spot + Perpetuals wallets.

        Hyperliquid Architecture:
        - Spot Wallet: Holds your USDC when not deposited into perpetuals
        - Perpetuals (Clearinghouse) Wallet: Holds margin for perp trading
        
        The Web UI shows the SUM of both. This method replicates that behavior
        by querying both user_state() and spot_user_state().

        Returns:
            AccountBalance with combined account value matching the Web UI.

        Raises:
            Exception: If API request fails.
        """
        # =======================================================================
        # 1. Query Perpetuals State (Clearinghouse)
        # =======================================================================
        perp_state = self.info.user_state(self.wallet_address)

        margin_summary = perp_state.get("marginSummary", {})
        cross_margin = perp_state.get("crossMarginSummary", {})

        # Perpetuals account value (includes unrealized PnL)
        perp_margin_val = float(margin_summary.get("accountValue", 0) or 0)
        perp_cross_val = float(cross_margin.get("accountValue", 0) or 0)
        perp_account_value = max(perp_margin_val, perp_cross_val)
        
        # Raw USDC deposited in perpetuals clearinghouse
        perp_raw_usd = float(cross_margin.get("totalRawUsd", 0) or 0)
        
        # Withdrawable from perpetuals
        perp_withdrawable = float(perp_state.get("withdrawable", 0) or 0)
        
        # Margin used in positions
        margin_used = float(margin_summary.get("totalMarginUsed", 0) or 0)
        cross_margin_used = float(cross_margin.get("totalMarginUsed", 0) or 0)
        total_margin_used = max(margin_used, cross_margin_used)
        
        # Total notional position value
        total_position_value = float(margin_summary.get("totalNtlPos", 0) or 0)

        # =======================================================================
        # 2. Query Spot State
        # =======================================================================
        spot_usdc = 0.0
        try:
            spot_state = self.info.spot_user_state(self.wallet_address)
            
            # spot_user_state returns {"balances": [{"coin": "USDC", "total": "123.45", ...}, ...]}
            balances = spot_state.get("balances", [])
            
            for balance in balances:
                coin = balance.get("coin", "")
                if coin == "USDC":
                    # "total" includes all USDC (available + in orders)
                    # "hold" is locked in open orders
                    spot_usdc = float(balance.get("total", 0) or 0)
                    break
                    
        except Exception as e:
            # spot_user_state may fail on testnet or if no spot activity
            # This is non-fatal - just means no spot balance
            pass

        # =======================================================================
        # 3. Aggregation : Total Account Value
        # =======================================================================
        
        total_account_value = max(perp_account_value, spot_usdc)
        total_withdrawable = total_account_value - total_margin_used

        return AccountBalance(
            account_value=total_account_value,
            total_margin_used=total_margin_used,
            total_position_value=total_position_value,
            withdrawable=total_withdrawable,
            raw_usd=perp_raw_usd,
            spot_usdc=spot_usdc,
        )

    def normalize_size(self, symbol: str, size: float) -> float:
        """Normalize order size to asset size precision expected by Hyperliquid."""
        numeric_size = float(size)
        asset = self.info.coin_to_asset.get(symbol)
        if asset is None:
            return float(f"{numeric_size:.8f}")
        sz_decimals = self.info.asset_to_sz_decimals.get(asset, 6)
        return round(numeric_size, sz_decimals)

    def normalize_price(self, symbol: str, price: float) -> float:
        """Normalize order/trigger price to Hyperliquid wire precision."""
        numeric_price = float(price)
        asset = self.info.coin_to_asset.get(symbol)
        if asset is None:
            return float(f"{numeric_price:.6f}")
        sz_decimals = self.info.asset_to_sz_decimals.get(asset, 0)
        price_decimals = max(0, 6 - sz_decimals)
        # Match SDK approach: 5 significant digits, then clamp to allowed decimals.
        return round(float(f"{numeric_price:.5g}"), price_decimals)

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

    def get_order_book(self, symbol: str) -> dict:
        """Get the current order book (L2) for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC", "ETH").

        Returns:
            Dict with 'bids' and 'asks' lists: [{"px": price, "sz": size}, ...]
        """
        try:
            l2_data = self.info.l2_snapshot(symbol)
            return {
                "bids": [{"px": float(level["px"]), "sz": float(level["sz"])} for level in l2_data.get("levels", [[]])[0]],
                "asks": [{"px": float(level["px"]), "sz": float(level["sz"])} for level in l2_data.get("levels", [[], []])[1]],
            }
        except Exception:
            return {"bids": [], "asks": []}

    def get_best_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Get the best bid and ask prices.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Tuple of (best_bid, best_ask). Returns (0, 0) if unavailable.
        """
        book = self.get_order_book(symbol)
        best_bid = book["bids"][0]["px"] if book["bids"] else 0.0
        best_ask = book["asks"][0]["px"] if book["asks"] else 0.0
        return best_bid, best_ask

    def get_funding_rate(self, symbol: str) -> float:
        """Get the current funding rate for a perpetual.

        Funding rate is expressed as a percentage (e.g., 0.01 = 0.01%).
        Positive funding = longs pay shorts (crowded long).
        Negative funding = shorts pay longs (crowded short).

        Args:
            symbol: Trading pair symbol (e.g., "BTC").

        Returns:
            Current funding rate as percentage. Returns 0.0 if unavailable.
        """
        try:
            meta = self.info.meta()
            universe = meta.get("universe", [])
            for asset in universe:
                if asset.get("name") == symbol:
                    funding = asset.get("funding", "0")
                    return float(funding) * 100  # Convert to percentage
            return 0.0
        except Exception:
            return 0.0

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Get all open orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol to filter by.

        Returns:
            List of open order dicts.
        """
        try:
            # frontend_open_orders includes trigger metadata (isTrigger, triggerPx, orderType).
            orders = self.info.frontend_open_orders(self.wallet_address)
            if symbol:
                return [o for o in orders if o.get("coin") == symbol]
            return orders
        except Exception:
            # Fallback for older SDK behavior.
            try:
                orders = self.info.open_orders(self.wallet_address)
                if symbol:
                    return [o for o in orders if o.get("coin") == symbol]
                return orders
            except Exception:
                return []

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel a specific order.

        Args:
            symbol: Trading pair symbol.
            order_id: Order ID to cancel.

        Returns:
            API response dict.
        """
        return self.exchange.cancel(name=symbol, oid=order_id)

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

    def get_candles_since(
        self,
        symbol: str,
        interval: str,
        since: datetime,
    ) -> pd.DataFrame:
        """Fetch candles from a specific time until now.

        Useful for state recovery to find highest/lowest prices since entry.

        Args:
            symbol: Trading pair symbol (e.g., "BTC").
            interval: Candle interval (e.g., "15m", "1h").
            since: Start datetime.

        Returns:
            DataFrame with OHLCV data from 'since' until now.
        """
        now = datetime.now()
        
        # Calculate how many candles we need
        interval_seconds = self._interval_to_seconds(interval)
        time_diff = (now - since).total_seconds()
        estimated_candles = int(time_diff / interval_seconds) + 10  # Buffer
        
        return self.get_candles(
            symbol=symbol,
            interval=interval,
            limit=min(estimated_candles, 1000),  # API limit
            start_time=since,
            end_time=now,
        )

    def get_trigger_orders(self, symbol: str | None = None) -> list[dict]:
        """Get all open trigger orders (SL/TP orders).

        Args:
            symbol: Optional symbol to filter by.

        Returns:
            List of trigger order dicts with order details.
        """
        try:
            # Use frontend_open_orders to reliably detect trigger orders.
            orders = self.info.frontend_open_orders(self.wallet_address)
            
            # Filter to only trigger orders
            trigger_orders = []
            for order in orders:
                is_trigger = bool(order.get("isTrigger"))
                order_type = str(order.get("orderType", ""))
                if is_trigger or "trigger" in order_type.lower() or order.get("triggerCondition"):
                    if symbol is None or order.get("coin") == symbol:
                        trigger_orders.append(order)
            
            return trigger_orders
        except Exception:
            return []
