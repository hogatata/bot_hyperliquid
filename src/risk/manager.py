"""Risk manager for position sizing, SL/TP, and leverage."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from src.exchange import HyperliquidClient


class Side(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


@dataclass
class TradeSetup:
    """Calculated trade parameters."""

    symbol: str
    side: Side
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    risk_amount: float  # USD amount at risk
    potential_profit: float  # USD potential profit
    atr_value: float = 0.0  # Current ATR for trailing stop


@dataclass
class OrderResult:
    """Result from order execution."""

    success: bool
    order_id: Optional[int] = None
    message: str = ""
    details: Optional[dict] = None


@dataclass
class ActivePosition:
    """Tracks an active position for trailing stop management."""

    symbol: str
    side: Side
    entry_price: float
    size: float
    initial_sl: float
    current_sl: float
    atr_value: float
    atr_trail_multiplier: float
    highest_price: float  # Highest favorable price (for trailing)
    lowest_price: float   # Lowest favorable price (for trailing)
    sl_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None


class RiskManager:
    """Manages position sizing, SL/TP calculation, and order execution."""

    def __init__(
        self,
        client: HyperliquidClient,
        position_size_percent: float = 5.0,
        default_leverage: int = 5,
        stop_loss_percent: float = 2.0,
        take_profit_percent: float = 4.0,
        use_atr_for_sl: bool = False,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 3.0,
        use_limit_orders: bool = False,
        limit_order_timeout: int = 60,
        trailing_stop_enabled: bool = False,
        trailing_atr_multiplier: float = 1.5,
    ):
        """Initialize the risk manager.

        Args:
            client: HyperliquidClient instance.
            position_size_percent: Percentage of account to use per trade (default 5%).
            default_leverage: Default leverage (default 5x).
            stop_loss_percent: Fixed SL as percentage of entry (default 2%).
            take_profit_percent: Fixed TP as percentage of entry (default 4%).
            use_atr_for_sl: Use ATR for dynamic SL/TP calculation.
            atr_sl_multiplier: ATR multiplier for stop loss (default 1.5).
            atr_tp_multiplier: ATR multiplier for take profit (default 3.0).
            use_limit_orders: Use limit orders for entry (maker rebates).
            limit_order_timeout: Seconds to wait for limit fill (default 60).
            trailing_stop_enabled: Enable trailing stop loss.
            trailing_atr_multiplier: ATR multiplier for trailing distance (default 1.5).
        """
        self.client = client
        self.position_size_percent = position_size_percent
        self.default_leverage = default_leverage
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.use_atr_for_sl = use_atr_for_sl
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.use_limit_orders = use_limit_orders
        self.limit_order_timeout = limit_order_timeout
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_atr_multiplier = trailing_atr_multiplier

        # Track active positions for trailing stop
        self.active_positions: dict[str, ActivePosition] = {}

    # =========================================================================
    # State Recovery Methods
    # =========================================================================

    def recover_state_on_startup(
        self,
        symbols: list[str],
        atr_period: int = 14,
        candle_interval: str = "15m",
    ) -> dict[str, dict]:
        """Recover position state after bot restart.

        Queries Hyperliquid for existing open positions and reconstructs
        ActivePosition tracking objects with accurate highest/lowest prices.

        Args:
            symbols: List of symbols to check for positions.
            atr_period: ATR period for trailing stop calculation.
            candle_interval: Candle interval for historical data.

        Returns:
            Dict mapping symbol to recovery info:
            {
                "BTC": {
                    "position": Position object,
                    "recovered": True/False,
                    "highest_price": float or None,
                    "lowest_price": float or None,
                    "existing_orders": list of order dicts,
                    "message": str
                }
            }
        """
        from datetime import datetime, timedelta
        from src.strategy.indicators import add_atr

        recovery_results = {}

        # Get all open positions
        open_positions = self.client.get_open_positions()

        for symbol in symbols:
            # Find position for this symbol
            position = None
            for pos in open_positions:
                if pos.symbol == symbol:
                    position = pos
                    break

            if position is None:
                recovery_results[symbol] = {
                    "position": None,
                    "recovered": False,
                    "message": "No open position found",
                }
                continue

            # Found an open position - need to reconstruct state
            recovery_info = self._reconstruct_position_state(
                position=position,
                atr_period=atr_period,
                candle_interval=candle_interval,
            )
            recovery_results[symbol] = recovery_info

        return recovery_results

    def _reconstruct_position_state(
        self,
        position,  # Position dataclass from client
        atr_period: int = 14,
        candle_interval: str = "15m",
        lookback_hours: int = 48,
    ) -> dict:
        """Reconstruct ActivePosition for an existing position.

        Fetches historical candles to find the highest/lowest price since entry.

        Args:
            position: Position object from client.
            atr_period: ATR period for calculation.
            candle_interval: Candle interval for historical data.
            lookback_hours: How far back to look for entry (default 48 hours).

        Returns:
            Dict with recovery information.
        """
        from datetime import datetime, timedelta
        from src.strategy.indicators import add_atr

        symbol = position.symbol
        entry_price = position.entry_price
        side = Side.LONG if position.side == "long" else Side.SHORT

        try:
            # Fetch historical candles (last 48 hours)
            since_time = datetime.now() - timedelta(hours=lookback_hours)
            df = self.client.get_candles_since(
                symbol=symbol,
                interval=candle_interval,
                since=since_time,
            )

            if df.empty:
                return {
                    "position": position,
                    "recovered": False,
                    "message": "Could not fetch historical candles",
                }

            # Add ATR for trailing stop distance
            df = add_atr(df, period=atr_period)
            current_atr = df[f"atr_{atr_period}"].iloc[-1] if f"atr_{atr_period}" in df.columns else 0

            # Find the candle where entry likely occurred
            # Look for first candle where price crossed entry_price
            entry_idx = self._find_entry_candle_index(df, entry_price, side)

            if entry_idx is None:
                # Entry was before our lookback window - use all available data
                entry_idx = 0

            # Calculate highest/lowest since entry
            df_since_entry = df.iloc[entry_idx:]

            if side == Side.LONG:
                highest_price = df_since_entry["high"].max()
                lowest_price = entry_price  # Not relevant for long trailing
            else:
                highest_price = entry_price  # Not relevant for short trailing
                lowest_price = df_since_entry["low"].min()

            # Get existing SL/TP orders for this symbol
            existing_orders = self.client.get_open_orders(symbol)
            sl_order_id = None
            tp_order_id = None
            current_sl = None

            for order in existing_orders:
                order_type = order.get("orderType", {})
                if isinstance(order_type, dict):
                    trigger_info = order_type.get("trigger", {})
                    if trigger_info.get("tpsl") == "sl":
                        sl_order_id = order.get("oid")
                        current_sl = float(trigger_info.get("triggerPx", 0))
                    elif trigger_info.get("tpsl") == "tp":
                        tp_order_id = order.get("oid")

            # Calculate initial SL for reference
            if current_sl is None:
                # No existing SL - calculate default
                if side == Side.LONG:
                    current_sl = entry_price * (1 - self.stop_loss_percent / 100)
                else:
                    current_sl = entry_price * (1 + self.stop_loss_percent / 100)

            # Create ActivePosition
            active_pos = ActivePosition(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=position.size,
                initial_sl=current_sl,
                current_sl=current_sl,
                atr_value=current_atr if current_atr > 0 else entry_price * 0.02,  # Fallback: 2% of price
                atr_trail_multiplier=self.trailing_atr_multiplier,
                highest_price=highest_price,
                lowest_price=lowest_price,
                sl_order_id=sl_order_id,
                tp_order_id=tp_order_id,
            )

            # Register in active_positions
            self.active_positions[symbol] = active_pos

            return {
                "position": position,
                "recovered": True,
                "active_position": active_pos,
                "highest_price": highest_price,
                "lowest_price": lowest_price,
                "existing_sl_order": sl_order_id,
                "existing_tp_order": tp_order_id,
                "atr_value": current_atr,
                "message": f"Recovered {side.value.upper()} position @ ${entry_price:,.2f}",
            }

        except Exception as e:
            return {
                "position": position,
                "recovered": False,
                "message": f"Recovery failed: {str(e)}",
            }

    def _find_entry_candle_index(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: Side,
    ) -> int | None:
        """Find the index of the candle where entry likely occurred.

        Looks for the candle where price first crossed the entry_price
        in the direction of the trade.

        Args:
            df: DataFrame with OHLCV data.
            entry_price: The entry price of the position.
            side: LONG or SHORT.

        Returns:
            Index of likely entry candle, or None if not found.
        """
        for i in range(len(df)):
            row = df.iloc[i]

            if side == Side.LONG:
                # For long, look for candle where low <= entry <= high
                if row["low"] <= entry_price <= row["high"]:
                    return i
            else:
                # For short, look for candle where low <= entry <= high
                if row["low"] <= entry_price <= row["high"]:
                    return i

        return None

    def sync_existing_orders(self, symbol: str) -> dict:
        """Sync existing SL/TP orders without duplicating them.

        Checks for existing orders and updates ActivePosition tracking
        to match what's already on the exchange.

        Args:
            symbol: Trading symbol.

        Returns:
            Dict with sync results.
        """
        if symbol not in self.active_positions:
            return {"synced": False, "message": "No active position to sync"}

        pos = self.active_positions[symbol]
        existing_orders = self.client.get_open_orders(symbol)

        synced_sl = False
        synced_tp = False

        for order in existing_orders:
            order_type = order.get("orderType", {})
            if isinstance(order_type, dict):
                trigger_info = order_type.get("trigger", {})
                if trigger_info.get("tpsl") == "sl":
                    pos.sl_order_id = order.get("oid")
                    pos.current_sl = float(trigger_info.get("triggerPx", pos.current_sl))
                    synced_sl = True
                elif trigger_info.get("tpsl") == "tp":
                    pos.tp_order_id = order.get("oid")
                    synced_tp = True

        return {
            "synced": True,
            "sl_synced": synced_sl,
            "tp_synced": synced_tp,
            "sl_order_id": pos.sl_order_id,
            "tp_order_id": pos.tp_order_id,
        }

    def has_existing_sl_tp_orders(self, symbol: str) -> dict:
        """Check if there are existing SL/TP orders for a symbol.

        Used before placing new orders to avoid duplicates.

        Args:
            symbol: Trading symbol.

        Returns:
            Dict with {has_sl: bool, has_tp: bool, sl_price: float, tp_price: float}
        """
        result = {
            "has_sl": False,
            "has_tp": False,
            "sl_price": None,
            "tp_price": None,
            "sl_order_id": None,
            "tp_order_id": None,
        }

        try:
            orders = self.client.get_open_orders(symbol)
            for order in orders:
                order_type = order.get("orderType", {})
                if isinstance(order_type, dict):
                    trigger_info = order_type.get("trigger", {})
                    if trigger_info.get("tpsl") == "sl":
                        result["has_sl"] = True
                        result["sl_price"] = float(trigger_info.get("triggerPx", 0))
                        result["sl_order_id"] = order.get("oid")
                    elif trigger_info.get("tpsl") == "tp":
                        result["has_tp"] = True
                        result["tp_price"] = float(trigger_info.get("triggerPx", 0))
                        result["tp_order_id"] = order.get("oid")
        except Exception:
            pass

        return result

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        leverage: int | None = None,
    ) -> float:
        """Calculate position size based on account balance percentage.

        Args:
            account_balance: Total account value in USD.
            entry_price: Expected entry price.
            leverage: Leverage to use (default: self.default_leverage).

        Returns:
            Position size in base currency units.
        """
        leverage = leverage or self.default_leverage

        # Calculate USD amount to trade (% of account * leverage)
        usd_to_trade = (account_balance * self.position_size_percent / 100) * leverage

        # Convert to base currency size
        size = usd_to_trade / entry_price

        return size

    def calculate_sl_tp_fixed(
        self,
        entry_price: float,
        side: Side,
        sl_percent: float | None = None,
        tp_percent: float | None = None,
    ) -> tuple[float, float]:
        """Calculate SL/TP using fixed percentage.

        Args:
            entry_price: Entry price.
            side: Trade direction (LONG or SHORT).
            sl_percent: Stop loss percentage (default: self.stop_loss_percent).
            tp_percent: Take profit percentage (default: self.take_profit_percent).

        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        sl_pct = sl_percent or self.stop_loss_percent
        tp_pct = tp_percent or self.take_profit_percent

        if side == Side.LONG:
            stop_loss = entry_price * (1 - sl_pct / 100)
            take_profit = entry_price * (1 + tp_pct / 100)
        else:  # SHORT
            stop_loss = entry_price * (1 + sl_pct / 100)
            take_profit = entry_price * (1 - tp_pct / 100)

        return stop_loss, take_profit

    def calculate_sl_tp_atr(
        self,
        entry_price: float,
        side: Side,
        atr_value: float,
        sl_multiplier: float | None = None,
        tp_multiplier: float | None = None,
    ) -> tuple[float, float]:
        """Calculate SL/TP using ATR (dynamic volatility-based).

        Args:
            entry_price: Entry price.
            side: Trade direction (LONG or SHORT).
            atr_value: Current ATR value.
            sl_multiplier: ATR multiplier for SL (default: self.atr_sl_multiplier).
            tp_multiplier: ATR multiplier for TP (default: self.atr_tp_multiplier).

        Returns:
            Tuple of (stop_loss_price, take_profit_price).
        """
        sl_mult = sl_multiplier or self.atr_sl_multiplier
        tp_mult = tp_multiplier or self.atr_tp_multiplier

        sl_distance = atr_value * sl_mult
        tp_distance = atr_value * tp_mult

        if side == Side.LONG:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return stop_loss, take_profit

    def prepare_trade(
        self,
        symbol: str,
        side: Side,
        leverage: int | None = None,
        df_with_indicators: pd.DataFrame | None = None,
        atr_column: str = "atr_14",
    ) -> TradeSetup:
        """Prepare a complete trade setup with all parameters calculated.

        Args:
            symbol: Trading symbol (e.g., "BTC").
            side: Trade direction.
            leverage: Leverage to use (default: self.default_leverage).
            df_with_indicators: DataFrame with ATR column (required if use_atr_for_sl=True).
            atr_column: Name of the ATR column in DataFrame.

        Returns:
            TradeSetup with all calculated parameters.
        """
        leverage = leverage or self.default_leverage

        # Get current price and account balance
        entry_price = self.client.get_current_price(symbol)
        balance = self.client.get_account_balance()
        account_value = balance.account_value

        # Calculate position size
        size = self.calculate_position_size(account_value, entry_price, leverage)

        # Calculate SL/TP
        atr_value = 0.0
        if self.use_atr_for_sl and df_with_indicators is not None:
            if atr_column not in df_with_indicators.columns:
                raise ValueError(f"ATR column '{atr_column}' not found in DataFrame")

            atr_value = df_with_indicators[atr_column].iloc[-1]
            if pd.isna(atr_value):
                raise ValueError("ATR value is NaN - not enough data")

            stop_loss, take_profit = self.calculate_sl_tp_atr(entry_price, side, atr_value)
        else:
            stop_loss, take_profit = self.calculate_sl_tp_fixed(entry_price, side)

        # Calculate risk/reward
        if side == Side.LONG:
            risk_amount = (entry_price - stop_loss) * size
            potential_profit = (take_profit - entry_price) * size
        else:
            risk_amount = (stop_loss - entry_price) * size
            potential_profit = (entry_price - take_profit) * size

        return TradeSetup(
            symbol=symbol,
            side=side,
            size=round(size, 6),  # Round to reasonable precision
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            leverage=leverage,
            risk_amount=round(risk_amount, 2),
            potential_profit=round(potential_profit, 2),
            atr_value=atr_value if self.use_atr_for_sl and df_with_indicators is not None else 0.0,
        )

    def execute_trade(
        self,
        setup: TradeSetup,
        dry_run: bool = False,
    ) -> dict[str, OrderResult]:
        """Execute a complete trade with SL/TP orders.

        Flow:
        1. Set leverage (isolated margin)
        2. Place entry order (market or limit based on use_limit_orders)
        3. Place stop loss order
        4. Place take profit order
        5. Register position for trailing stop (if enabled)

        Args:
            setup: TradeSetup with all parameters.
            dry_run: If True, don't actually place orders (for testing).

        Returns:
            Dict with results for each step: {"leverage", "entry", "stop_loss", "take_profit"}
        """
        results = {}
        is_buy = setup.side == Side.LONG

        # Step 1: Set leverage (isolated margin)
        if dry_run:
            results["leverage"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would set leverage to {setup.leverage}x isolated",
            )
        else:
            try:
                response = self.client.set_leverage(
                    symbol=setup.symbol,
                    leverage=setup.leverage,
                    is_cross=False,  # ISOLATED margin as per spec
                )
                results["leverage"] = OrderResult(
                    success=True,
                    message=f"Leverage set to {setup.leverage}x isolated",
                    details=response,
                )
            except Exception as e:
                results["leverage"] = OrderResult(
                    success=False,
                    message=f"Failed to set leverage: {e}",
                )
                return results  # Abort if leverage fails

        # Step 2: Place entry order (Limit or Market)
        actual_entry_price = setup.entry_price
        if dry_run:
            order_type_str = "LIMIT" if self.use_limit_orders else "MARKET"
            results["entry"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would open {order_type_str} {setup.side.value.upper()} {setup.size} {setup.symbol} @ ~${setup.entry_price:,.2f}",
            )
        else:
            if self.use_limit_orders:
                results["entry"] = self._execute_limit_entry(setup, is_buy)
                if not results["entry"].success:
                    return results  # Abort if entry fails
                # Get actual fill price if available
                if results["entry"].details:
                    fill_price = self._extract_fill_price(results["entry"].details)
                    if fill_price:
                        actual_entry_price = fill_price
            else:
                results["entry"] = self._execute_market_entry(setup, is_buy)
                if not results["entry"].success:
                    return results  # Abort if entry fails

        # Step 3: Place Stop Loss order (trigger order, reduce_only)
        sl_order_id = None
        if dry_run:
            results["stop_loss"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would place SL @ ${setup.stop_loss:,.2f}",
            )
        else:
            results["stop_loss"] = self._place_stop_loss(setup, is_buy)
            sl_order_id = results["stop_loss"].order_id

        # Step 4: Place Take Profit order (trigger order, reduce_only)
        tp_order_id = None
        if dry_run:
            results["take_profit"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would place TP @ ${setup.take_profit:,.2f}",
            )
        else:
            results["take_profit"] = self._place_take_profit(setup, is_buy)
            tp_order_id = results["take_profit"].order_id

        # Step 5: Register for trailing stop management (if enabled)
        if self.trailing_stop_enabled and not dry_run and results["entry"].success:
            self.active_positions[setup.symbol] = ActivePosition(
                symbol=setup.symbol,
                side=setup.side,
                entry_price=actual_entry_price,
                size=setup.size,
                initial_sl=setup.stop_loss,
                current_sl=setup.stop_loss,
                atr_value=setup.atr_value,
                atr_trail_multiplier=self.trailing_atr_multiplier,
                highest_price=actual_entry_price,
                lowest_price=actual_entry_price,
                sl_order_id=sl_order_id,
                tp_order_id=tp_order_id,
            )

        return results

    def _execute_market_entry(self, setup: TradeSetup, is_buy: bool) -> OrderResult:
        """Execute market entry order."""
        try:
            response = self.client.exchange.market_open(
                name=setup.symbol,
                is_buy=is_buy,
                sz=setup.size,
            )
            order_id = self._extract_order_id(response)
            return OrderResult(
                success=response.get("status") == "ok",
                order_id=order_id,
                message=f"MARKET {setup.side.value.upper()} {setup.size} {setup.symbol}",
                details=response,
            )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Failed to place market entry: {e}",
            )

    def _execute_limit_entry(self, setup: TradeSetup, is_buy: bool) -> OrderResult:
        """Execute limit entry order with timeout for maker rebates.

        Places limit order at best bid (for longs) or best ask (for shorts).
        Waits up to limit_order_timeout seconds for fill.
        Cancels and aborts if not filled.
        """
        try:
            # Get best bid/ask for limit price
            best_bid, best_ask = self.client.get_best_bid_ask(setup.symbol)
            if best_bid == 0 or best_ask == 0:
                # Fallback to market if order book unavailable
                return self._execute_market_entry(setup, is_buy)

            # Place limit at best bid (long) or best ask (short) for maker rebate
            limit_price = best_bid if is_buy else best_ask

            response = self.client.exchange.order(
                name=setup.symbol,
                is_buy=is_buy,
                sz=setup.size,
                limit_px=limit_price,
                order_type={"limit": {"tif": "Gtc"}},  # Good-til-cancelled
                reduce_only=False,
            )

            if response.get("status") != "ok":
                return OrderResult(
                    success=False,
                    message=f"Failed to place limit order: {response}",
                    details=response,
                )

            order_id = self._extract_order_id(response)

            # Check if immediately filled
            statuses = response.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and "filled" in statuses[0]:
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    message=f"LIMIT {setup.side.value.upper()} {setup.size} {setup.symbol} @ ${limit_price:,.2f} (filled)",
                    details=response,
                )

            # Wait for fill with timeout
            start_time = time.time()
            while time.time() - start_time < self.limit_order_timeout:
                time.sleep(2)  # Check every 2 seconds

                # Check if order is still open
                open_orders = self.client.get_open_orders(setup.symbol)
                order_still_open = any(o.get("oid") == order_id for o in open_orders)

                if not order_still_open:
                    # Order filled or cancelled
                    return OrderResult(
                        success=True,
                        order_id=order_id,
                        message=f"LIMIT {setup.side.value.upper()} {setup.size} {setup.symbol} @ ${limit_price:,.2f} (filled)",
                        details=response,
                    )

            # Timeout - cancel unfilled order
            if order_id:
                self.client.cancel_order(setup.symbol, order_id)
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    message=f"LIMIT order timed out after {self.limit_order_timeout}s - cancelled",
                )

            return OrderResult(
                success=False,
                message="LIMIT order timed out - no order ID to cancel",
            )

        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Failed to place limit entry: {e}",
            )

    def _place_stop_loss(self, setup: TradeSetup, is_buy: bool) -> OrderResult:
        """Place stop loss trigger order."""
        try:
            sl_order_type = {
                "trigger": {
                    "triggerPx": setup.stop_loss,
                    "isMarket": True,
                    "tpsl": "sl",
                }
            }
            response = self.client.exchange.order(
                name=setup.symbol,
                is_buy=not is_buy,  # Opposite direction to close
                sz=setup.size,
                limit_px=setup.stop_loss,
                order_type=sl_order_type,
                reduce_only=True,
            )
            order_id = self._extract_order_id(response)
            return OrderResult(
                success=response.get("status") == "ok",
                order_id=order_id,
                message=f"SL placed @ ${setup.stop_loss:,.2f}",
                details=response,
            )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Failed to place SL: {e}",
            )

    def _place_take_profit(self, setup: TradeSetup, is_buy: bool) -> OrderResult:
        """Place take profit trigger order."""
        try:
            tp_order_type = {
                "trigger": {
                    "triggerPx": setup.take_profit,
                    "isMarket": True,
                    "tpsl": "tp",
                }
            }
            response = self.client.exchange.order(
                name=setup.symbol,
                is_buy=not is_buy,  # Opposite direction to close
                sz=setup.size,
                limit_px=setup.take_profit,
                order_type=tp_order_type,
                reduce_only=True,
            )
            order_id = self._extract_order_id(response)
            return OrderResult(
                success=response.get("status") == "ok",
                order_id=order_id,
                message=f"TP placed @ ${setup.take_profit:,.2f}",
                details=response,
            )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Failed to place TP: {e}",
            )

    def _extract_order_id(self, response: dict) -> int | None:
        """Extract order ID from API response."""
        if response.get("status") == "ok":
            statuses = response.get("response", {}).get("data", {}).get("statuses", [])
            if statuses:
                if "resting" in statuses[0]:
                    return statuses[0]["resting"]["oid"]
                elif "filled" in statuses[0]:
                    return statuses[0]["filled"]["oid"]
        return None

    def _extract_fill_price(self, response: dict) -> float | None:
        """Extract fill price from API response."""
        if response.get("status") == "ok":
            statuses = response.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and "filled" in statuses[0]:
                return float(statuses[0]["filled"].get("avgPx", 0))
        return None

    def update_trailing_stop(self, symbol: str, current_price: float) -> OrderResult | None:
        """Update trailing stop if price moved in our favor.

        Called from main loop to manage trailing stops locally.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.

        Returns:
            OrderResult if SL was updated, None otherwise.
        """
        if symbol not in self.active_positions:
            return None

        pos = self.active_positions[symbol]

        # Calculate new stop based on direction
        if pos.side == Side.LONG:
            # Track highest price for long positions
            if current_price > pos.highest_price:
                pos.highest_price = current_price

                # Calculate new SL based on highest price
                trail_distance = pos.atr_value * pos.atr_trail_multiplier
                new_sl = pos.highest_price - trail_distance

                # Only move SL up, never down
                if new_sl > pos.current_sl:
                    return self._update_sl_order(pos, new_sl)

        else:  # SHORT
            # Track lowest price for short positions
            if current_price < pos.lowest_price:
                pos.lowest_price = current_price

                # Calculate new SL based on lowest price
                trail_distance = pos.atr_value * pos.atr_trail_multiplier
                new_sl = pos.lowest_price + trail_distance

                # Only move SL down, never up
                if new_sl < pos.current_sl:
                    return self._update_sl_order(pos, new_sl)

        return None

    def _update_sl_order(self, pos: ActivePosition, new_sl: float) -> OrderResult:
        """Cancel old SL and place new one at updated price."""
        try:
            # Cancel existing SL order
            if pos.sl_order_id:
                self.client.cancel_order(pos.symbol, pos.sl_order_id)

            # Place new SL
            is_buy = pos.side == Side.SHORT  # Opposite direction to close
            sl_order_type = {
                "trigger": {
                    "triggerPx": new_sl,
                    "isMarket": True,
                    "tpsl": "sl",
                }
            }
            response = self.client.exchange.order(
                name=pos.symbol,
                is_buy=is_buy,
                sz=pos.size,
                limit_px=new_sl,
                order_type=sl_order_type,
                reduce_only=True,
            )

            if response.get("status") == "ok":
                pos.current_sl = new_sl
                pos.sl_order_id = self._extract_order_id(response)
                return OrderResult(
                    success=True,
                    order_id=pos.sl_order_id,
                    message=f"Trailing SL updated: ${pos.initial_sl:,.2f} → ${new_sl:,.2f}",
                    details=response,
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Failed to update trailing SL: {response}",
                )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Error updating trailing SL: {e}",
            )

    def clear_position_tracking(self, symbol: str):
        """Remove position from tracking (called when position closes)."""
        if symbol in self.active_positions:
            del self.active_positions[symbol]

    def cancel_all_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """Cancel all open orders (for kill switch).

        Args:
            symbol: If provided, only cancel orders for this symbol.

        Returns:
            List of OrderResults for each cancellation.
        """
        results = []

        try:
            open_orders = self.client.info.open_orders(self.client.wallet_address)

            for order in open_orders:
                if symbol and order["coin"] != symbol:
                    continue

                try:
                    response = self.client.exchange.cancel(
                        name=order["coin"],
                        oid=order["oid"],
                    )
                    results.append(
                        OrderResult(
                            success=response.get("status") == "ok",
                            order_id=order["oid"],
                            message=f"Cancelled order {order['oid']} for {order['coin']}",
                            details=response,
                        )
                    )
                except Exception as e:
                    results.append(
                        OrderResult(
                            success=False,
                            order_id=order["oid"],
                            message=f"Failed to cancel order {order['oid']}: {e}",
                        )
                    )
        except Exception as e:
            results.append(
                OrderResult(
                    success=False,
                    message=f"Failed to fetch open orders: {e}",
                )
            )

        return results

    def close_all_positions(self, symbol: str | None = None) -> list[OrderResult]:
        """Close all open positions at market (for kill switch).

        Args:
            symbol: If provided, only close position for this symbol.

        Returns:
            List of OrderResults for each closure.
        """
        results = []

        try:
            positions = self.client.get_open_positions()

            for pos in positions:
                if symbol and pos.symbol != symbol:
                    continue

                try:
                    response = self.client.exchange.market_close(
                        coin=pos.symbol,
                    )
                    results.append(
                        OrderResult(
                            success=response.get("status") == "ok" if response else False,
                            message=f"Closed {pos.side.upper()} {pos.size} {pos.symbol}",
                            details=response,
                        )
                    )
                except Exception as e:
                    results.append(
                        OrderResult(
                            success=False,
                            message=f"Failed to close {pos.symbol}: {e}",
                        )
                    )
        except Exception as e:
            results.append(
                OrderResult(
                    success=False,
                    message=f"Failed to fetch positions: {e}",
                )
            )

        return results

    def emergency_shutdown(self) -> dict:
        """Execute emergency shutdown (kill switch).

        1. Cancel all pending orders
        2. Close all open positions at market
        3. Clear all position tracking

        Returns:
            Dict with results for cancellations and closures.
        """
        # Clear all position tracking
        self.active_positions.clear()

        return {
            "cancelled_orders": self.cancel_all_orders(),
            "closed_positions": self.close_all_positions(),
        }
