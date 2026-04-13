"""Risk manager for ATR-based volatility targeting and Chandelier Exit."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from src.exchange import HyperliquidClient

logger = logging.getLogger(__name__)


class Side(Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


@dataclass
class TradeSetup:
    """Calculated trade parameters for ATR-based volatility targeting."""

    symbol: str
    side: Side
    size: float
    entry_price: float
    stop_loss: float
    calculated_leverage: int  # Dynamic based on position size
    risk_amount: float  # USD amount at risk (fixed % of account)
    account_value: float  # Account value at setup time
    risk_percent: float  # Configured risk percent per trade
    atr_value: float  # Current ATR for trailing stop
    sl_distance: float  # ATR * sl_multiplier


@dataclass
class OrderResult:
    """Result from order execution."""

    success: bool
    order_id: Optional[int] = None
    message: str = ""
    details: Optional[dict] = None


@dataclass
class ActivePosition:
    """Tracks an active position for Chandelier Exit trailing stop."""

    symbol: str
    side: Side
    entry_price: float
    size: float
    current_sl: float
    atr_value: float
    atr_trailing_multiplier: float
    highest_price: float  # For LONG: track highest since entry
    lowest_price: float   # For SHORT: track lowest since entry
    sl_order_id: Optional[int] = None


class RiskManager:
    """Manages ATR-based volatility targeting and Chandelier Exit trailing stop.
    
    Position Sizing (Volatility Targeting):
    - SL Distance = ATR * atr_sl_multiplier
    - Position Size = (Account Balance * risk_percent) / SL Distance
    - No fixed leverage - math dictates the size
    
    Trailing Stop (Chandelier Exit):
    - No fixed Take-Profit
    - LONG: New SL = max(Prev SL, Highest_High - ATR * atr_trailing_multiplier)
    - SHORT: New SL = min(Prev SL, Lowest_Low + ATR * atr_trailing_multiplier)
    """

    def __init__(
        self,
        client: HyperliquidClient,
        risk_percent_per_trade: float = 2.0,
        atr_sl_multiplier: float = 1.5,
        atr_trailing_multiplier: float = 2.0,
        max_leverage: int = 10,
        use_limit_orders: bool = False,
        limit_order_timeout: int = 60,
    ):
        """Initialize the risk manager.

        Args:
            client: HyperliquidClient instance.
            risk_percent_per_trade: Percentage of account to risk per trade (default 2%).
            atr_sl_multiplier: ATR multiplier for initial stop loss (default 1.5).
            atr_trailing_multiplier: ATR multiplier for trailing distance (default 2.0).
            max_leverage: Maximum allowed leverage (default 10x).
            use_limit_orders: Use limit orders for entry (maker rebates).
            limit_order_timeout: Seconds to wait for limit fill (default 60).
        """
        self.client = client
        self.risk_percent_per_trade = risk_percent_per_trade
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_trailing_multiplier = atr_trailing_multiplier
        self.max_leverage = max_leverage
        self.use_limit_orders = use_limit_orders
        self.limit_order_timeout = limit_order_timeout

        # Track active positions for Chandelier Exit
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

            # Get existing trigger orders (SL/TP) for this symbol
            existing_orders = self.client.get_trigger_orders(symbol)
            sl_order_id = None
            tp_order_id = None
            current_sl = None

            for order in existing_orders:
                order_type = str(order.get("orderType", "")).lower()
                trigger_condition = str(order.get("triggerCondition", "")).lower()
                if "sl" in order_type or "stop" in order_type or "stop" in trigger_condition:
                    sl_order_id = order.get("oid")
                    trigger_px = order.get("triggerPx")
                    if trigger_px is not None:
                        current_sl = float(trigger_px)
                elif "tp" in order_type or "take" in trigger_condition:
                    tp_order_id = order.get("oid")

            # Calculate initial SL using ATR
            sl_distance = current_atr * self.atr_sl_multiplier if current_atr > 0 else entry_price * 0.02
            if current_sl is None:
                # No existing SL - calculate based on ATR
                if side == Side.LONG:
                    current_sl = entry_price - sl_distance
                else:
                    current_sl = entry_price + sl_distance
                # Critical safety: recreate missing SL on startup recovery.
                try:
                    is_buy_close = side == Side.SHORT
                    safe_current_sl = self.client.normalize_price(symbol, current_sl)
                    safe_size = self.client.normalize_size(symbol, position.size)
                    recreated = self.client.exchange.order(
                        name=symbol,
                        is_buy=is_buy_close,
                        sz=safe_size,
                        limit_px=safe_current_sl,
                        order_type={
                            "trigger": {
                                "triggerPx": safe_current_sl,
                                "isMarket": True,
                                "tpsl": "sl",
                            }
                        },
                        reduce_only=True,
                    )
                    recreated_oid = self._extract_order_id(recreated)
                    if recreated.get("status") == "ok" and recreated_oid is not None:
                        current_sl = safe_current_sl
                        sl_order_id = recreated_oid
                        logger.warning(
                            "Recovered position had no SL. Recreated SL | symbol=%s sl=%.4f oid=%s",
                            symbol,
                            current_sl,
                            sl_order_id,
                        )
                    else:
                        logger.critical(
                            "Recovered position still unprotected: failed to recreate SL | symbol=%s response=%s",
                            symbol,
                            recreated,
                        )
                except Exception as e:
                    logger.critical(
                        "Recovered position still unprotected: exception recreating SL | symbol=%s error=%s",
                        symbol,
                        e,
                    )

            # Create ActivePosition
            active_pos = ActivePosition(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=position.size,
                current_sl=current_sl,
                atr_value=current_atr if current_atr > 0 else entry_price * 0.02,
                atr_trailing_multiplier=self.atr_trailing_multiplier,
                highest_price=highest_price,
                lowest_price=lowest_price,
                sl_order_id=sl_order_id,
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
        """Sync existing SL orders without duplicating them.

        For Chandelier Exit, we only track SL orders (no fixed TP).

        Args:
            symbol: Trading symbol.

        Returns:
            Dict with sync results.
        """
        if symbol not in self.active_positions:
            return {"synced": False, "message": "No active position to sync"}

        pos = self.active_positions[symbol]
        existing_orders = self.client.get_trigger_orders(symbol)

        synced_sl = False

        for order in existing_orders:
            order_type = str(order.get("orderType", "")).lower()
            trigger_condition = str(order.get("triggerCondition", "")).lower()
            if "sl" in order_type or "stop" in order_type or "stop" in trigger_condition:
                pos.sl_order_id = order.get("oid")
                trigger_px = order.get("triggerPx")
                if trigger_px is not None:
                    pos.current_sl = float(trigger_px)
                synced_sl = True

        return {
            "synced": True,
            "sl_synced": synced_sl,
            "sl_order_id": pos.sl_order_id,
        }

    def has_existing_sl_order(self, symbol: str) -> dict:
        """Check if there is an existing SL order for a symbol.

        For Chandelier Exit, we only track SL orders (no fixed TP).

        Args:
            symbol: Trading symbol.

        Returns:
            Dict with {has_sl: bool, sl_price: float, sl_order_id: int}
        """
        result = {
            "has_sl": False,
            "sl_price": None,
            "sl_order_id": None,
        }

        try:
            orders = self.client.get_trigger_orders(symbol)
            for order in orders:
                order_type = str(order.get("orderType", "")).lower()
                trigger_condition = str(order.get("triggerCondition", "")).lower()
                if "sl" in order_type or "stop" in order_type or "stop" in trigger_condition:
                    result["has_sl"] = True
                    trigger_px = order.get("triggerPx")
                    if trigger_px is not None:
                        result["sl_price"] = float(trigger_px)
                    result["sl_order_id"] = order.get("oid")
        except Exception:
            pass

        return result

    # =========================================================================
    # ATR-Based Volatility Targeting Position Sizing
    # =========================================================================

    def calculate_position_size_volatility_target(
        self,
        account_balance: float,
        entry_price: float,
        atr_value: float,
    ) -> tuple[float, float, float, int]:
        """Calculate position size using ATR-based volatility targeting.
        
        Formula: Position Size = (Account Balance * risk_percent) / SL Distance
        Where: SL Distance = ATR * atr_sl_multiplier
        
        This ensures we risk a fixed percentage of our account per trade,
        regardless of volatility.

        Args:
            account_balance: Total account value in USD.
            entry_price: Expected entry price.
            atr_value: Current ATR value.

        Returns:
            Tuple of (position_size, sl_distance, risk_amount, calculated_leverage).
        """
        # Calculate the dollar amount we're willing to risk
        risk_amount = account_balance * (self.risk_percent_per_trade / 100)
        
        # Calculate stop loss distance using ATR
        sl_distance = atr_value * self.atr_sl_multiplier
        
        # Position size = risk_amount / sl_distance (in base currency)
        position_size = risk_amount / sl_distance
        
        # Calculate notional value and required leverage
        notional_value = position_size * entry_price
        required_leverage = notional_value / account_balance
        
        # Cap leverage at maximum allowed
        calculated_leverage = min(int(required_leverage) + 1, self.max_leverage)
        
        # If required leverage exceeds max, reduce position size
        if required_leverage > self.max_leverage:
            max_notional = account_balance * self.max_leverage
            position_size = max_notional / entry_price
            # Recalculate risk with capped size
            risk_amount = position_size * sl_distance
        
        return position_size, sl_distance, risk_amount, calculated_leverage

    def calculate_atr_stop_loss(
        self,
        entry_price: float,
        side: Side,
        atr_value: float,
    ) -> float:
        """Calculate initial stop loss using ATR.

        Args:
            entry_price: Entry price.
            side: Trade direction (LONG or SHORT).
            atr_value: Current ATR value.

        Returns:
            Stop loss price.
        """
        sl_distance = atr_value * self.atr_sl_multiplier

        if side == Side.LONG:
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance

    def prepare_trade(
        self,
        symbol: str,
        side: Side,
        df_with_indicators: pd.DataFrame,
        atr_column: str = "atr_14",
    ) -> TradeSetup:
        """Prepare a trade setup using ATR-based volatility targeting.

        Args:
            symbol: Trading symbol (e.g., "BTC").
            side: Trade direction.
            df_with_indicators: DataFrame with ATR column (REQUIRED).
            atr_column: Name of the ATR column in DataFrame.

        Returns:
            TradeSetup with all calculated parameters.
        """
        # Get current price and account balance
        entry_price = self.client.get_current_price(symbol)
        balance = self.client.get_account_balance()
        account_value = balance.account_value

        # Get ATR value (REQUIRED for volatility targeting)
        if atr_column not in df_with_indicators.columns:
            raise ValueError(f"ATR column '{atr_column}' not found in DataFrame")

        atr_value = df_with_indicators[atr_column].iloc[-1]
        if pd.isna(atr_value) or atr_value <= 0:
            raise ValueError("ATR value is NaN or zero - not enough data")

        # Calculate position size using volatility targeting
        size, sl_distance, risk_amount, leverage = self.calculate_position_size_volatility_target(
            account_balance=account_value,
            entry_price=entry_price,
            atr_value=atr_value,
        )

        # Calculate stop loss
        stop_loss = self.calculate_atr_stop_loss(entry_price, side, atr_value)

        return TradeSetup(
            symbol=symbol,
            side=side,
            size=round(size, 6),
            entry_price=entry_price,
            stop_loss=round(stop_loss, 2),
            calculated_leverage=leverage,
            risk_amount=round(risk_amount, 2),
            account_value=round(account_value, 2),
            risk_percent=self.risk_percent_per_trade,
            atr_value=atr_value,
            sl_distance=sl_distance,
        )

    def execute_trade(
        self,
        setup: TradeSetup,
        dry_run: bool = False,
    ) -> dict[str, OrderResult]:
        """Execute a trade with Chandelier Exit (ATR trailing stop).

        Flow:
        1. Set leverage (isolated margin)
        2. Place entry order (market or limit based on use_limit_orders)
        3. Place initial stop loss order
        4. Register position for Chandelier Exit trailing stop

        Args:
            setup: TradeSetup with all parameters.
            dry_run: If True, don't actually place orders (for testing).

        Returns:
            Dict with results for each step: {"leverage", "entry", "stop_loss"}
        """
        results = {}
        is_buy = setup.side == Side.LONG
        estimated_risk_usd = abs(setup.entry_price - setup.stop_loss) * setup.size
        logger.info(
            (
                "Pre-trade risk snapshot | symbol=%s side=%s atr=%.8f entry=%.4f "
                "stop=%.4f size=%.6f estimated_risk_usd=%.2f target_risk_usd=%.2f "
                "account_value=%.2f risk_percent=%.2f"
            ),
            setup.symbol,
            setup.side.value.upper(),
            setup.atr_value,
            setup.entry_price,
            setup.stop_loss,
            setup.size,
            estimated_risk_usd,
            setup.risk_amount,
            setup.account_value,
            setup.risk_percent,
        )

        # Step 1: Set leverage (isolated margin)
        if dry_run:
            results["leverage"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would set leverage to {setup.calculated_leverage}x isolated",
            )
        else:
            try:
                response = self.client.set_leverage(
                    symbol=setup.symbol,
                    leverage=setup.calculated_leverage,
                    is_cross=False,  # ISOLATED margin as per spec
                )
                results["leverage"] = OrderResult(
                    success=True,
                    message=f"Leverage set to {setup.calculated_leverage}x isolated",
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
        sl_size = self._get_live_position_size(setup.symbol) if not dry_run else setup.size
        if sl_size is None or sl_size <= 0:
            sl_size = setup.size
        if abs(sl_size - setup.size) > 1e-9:
            logger.warning(
                "SL size adjusted from setup size using live position | symbol=%s setup_size=%.6f live_size=%.6f",
                setup.symbol,
                setup.size,
                sl_size,
            )
        # Recalculate SL from ACTUAL entry price to preserve configured max risk after slippage.
        sl_distance = abs(setup.entry_price - setup.stop_loss)
        if setup.side == Side.LONG:
            effective_stop_loss = actual_entry_price - sl_distance
        else:
            effective_stop_loss = actual_entry_price + sl_distance
        effective_stop_loss = round(effective_stop_loss, 2)
        sl_estimated_risk_usd = abs(actual_entry_price - effective_stop_loss) * sl_size
        logger.info(
            (
                "Preparing stop order | symbol=%s side=%s atr=%.8f entry=%.4f stop=%.4f "
                "size=%.6f estimated_risk_usd=%.2f"
            ),
            setup.symbol,
            setup.side.value.upper(),
            setup.atr_value,
            actual_entry_price,
            effective_stop_loss,
            sl_size,
            sl_estimated_risk_usd,
        )
        if dry_run:
            results["stop_loss"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would place SL @ ${effective_stop_loss:,.2f}",
            )
        else:
            results["stop_loss"] = self._place_stop_loss(
                setup,
                is_buy,
                order_size=sl_size,
                stop_loss_price=effective_stop_loss,
            )
            sl_order_id = results["stop_loss"].order_id
            if not results["stop_loss"].success:
                logger.critical(
                    "Stop-loss placement failed after entry. Triggering emergency close | symbol=%s reason=%s",
                    setup.symbol,
                    results["stop_loss"].message,
                )
                results["emergency_close"] = self._emergency_close_position(
                    symbol=setup.symbol,
                    reason=results["stop_loss"].message,
                )
                return results

        # Step 4: Register for Chandelier Exit trailing stop management
        if not dry_run and results["entry"].success:
            self.active_positions[setup.symbol] = ActivePosition(
                symbol=setup.symbol,
                side=setup.side,
                entry_price=actual_entry_price,
                size=sl_size,
                current_sl=effective_stop_loss,
                atr_value=setup.atr_value,
                atr_trailing_multiplier=self.atr_trailing_multiplier,
                highest_price=actual_entry_price,
                lowest_price=actual_entry_price,
                sl_order_id=sl_order_id,
            )

        return results

    def _execute_market_entry(self, setup: TradeSetup, is_buy: bool) -> OrderResult:
        """Execute market entry order."""
        try:
            safe_size = self.client.normalize_size(setup.symbol, setup.size)
            response = self.client.exchange.market_open(
                name=setup.symbol,
                is_buy=is_buy,
                sz=safe_size,
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
            safe_limit_price = self.client.normalize_price(setup.symbol, limit_price)
            safe_size = self.client.normalize_size(setup.symbol, setup.size)

            response = self.client.exchange.order(
                name=setup.symbol,
                is_buy=is_buy,
                sz=safe_size,
                limit_px=safe_limit_price,
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
                    message=f"LIMIT {setup.side.value.upper()} {setup.size} {setup.symbol} @ ${safe_limit_price:,.2f} (filled)",
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
                        message=f"LIMIT {setup.side.value.upper()} {setup.size} {setup.symbol} @ ${safe_limit_price:,.2f} (filled)",
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

    def _place_stop_loss(
        self,
        setup: TradeSetup,
        is_buy: bool,
        order_size: float,
        stop_loss_price: float,
    ) -> OrderResult:
        """Place stop loss trigger order."""
        try:
            safe_trigger_px = self.client.normalize_price(setup.symbol, stop_loss_price)
            safe_order_size = self.client.normalize_size(setup.symbol, order_size)
            sl_order_type = {
                "trigger": {
                    "triggerPx": safe_trigger_px,
                    "isMarket": True,
                    "tpsl": "sl",
                }
            }
            logger.info(
                (
                    "Submitting Hyperliquid trigger SL order | symbol=%s is_buy=%s "
                    "reduce_only=true size=%.6f limit_px=%.4f order_type=%s"
                ),
                setup.symbol,
                not is_buy,
                safe_order_size,
                safe_trigger_px,
                sl_order_type,
            )
            response = self.client.exchange.order(
                name=setup.symbol,
                is_buy=not is_buy,  # Opposite direction to close
                sz=safe_order_size,
                limit_px=safe_trigger_px,
                order_type=sl_order_type,
                reduce_only=True,
            )
            order_id = self._extract_order_id(response)
            success = response.get("status") == "ok" and order_id is not None
            if success:
                # Explicit exchange-side verification: confirm trigger order is actually visible.
                time.sleep(0.25)
                verified = self._verify_trigger_order_visible(
                    symbol=setup.symbol,
                    order_id=order_id,
                    expected_trigger_px=safe_trigger_px,
                )
                if not verified:
                    return OrderResult(
                        success=False,
                        order_id=order_id,
                        message=(
                            "SL order accepted but not visible in open trigger orders. "
                            "Treating as failure for safety."
                        ),
                        details=response,
                    )
            return OrderResult(
                success=success,
                order_id=order_id,
                message=(
                    f"SL placed @ ${stop_loss_price:,.2f}"
                    if success
                    else f"SL rejected/invalid response: {response}"
                ),
                details=response,
            )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Failed to place SL: {e}",
            )

    def _verify_trigger_order_visible(
        self,
        symbol: str,
        order_id: int,
        expected_trigger_px: float,
        tolerance: float = 0.5,
    ) -> bool:
        """Verify the newly created SL is present in frontend trigger orders."""
        try:
            trigger_orders = self.client.get_trigger_orders(symbol)
            for order in trigger_orders:
                if order.get("oid") != order_id:
                    continue
                trigger_px_raw = order.get("triggerPx")
                if trigger_px_raw is None:
                    return True
                trigger_px = float(trigger_px_raw)
                return abs(trigger_px - expected_trigger_px) <= tolerance
            logger.error(
                "SL not found in trigger orders after placement | symbol=%s oid=%s expected_trigger_px=%.4f",
                symbol,
                order_id,
                expected_trigger_px,
            )
            return False
        except Exception as e:
            logger.error("Failed to verify trigger order visibility for %s: %s", symbol, e)
            return False

    def _get_live_position_size(self, symbol: str) -> float | None:
        """Get the current absolute open size for a symbol."""
        try:
            positions = self.client.get_open_positions()
            for position in positions:
                if position.symbol == symbol:
                    return abs(position.size)
        except Exception as e:
            logger.warning("Could not fetch live position size for %s: %s", symbol, e)
        return None

    def _emergency_close_position(self, symbol: str, reason: str) -> OrderResult:
        """Immediately close a position if SL creation failed."""
        try:
            logger.critical(
                "EMERGENCY CLOSE | symbol=%s reason=%s",
                symbol,
                reason,
            )
            response = self.client.exchange.market_close(coin=symbol)
            success = bool(response) and response.get("status") == "ok"
            return OrderResult(
                success=success,
                message=(
                    f"Emergency close executed for {symbol}"
                    if success
                    else f"Emergency close failed for {symbol}: {response}"
                ),
                details=response,
            )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Emergency close exception for {symbol}: {e}",
            )

    def _place_take_profit(self, setup: TradeSetup, is_buy: bool) -> OrderResult:
        """Place take profit trigger order."""
        try:
            safe_tp_price = self.client.normalize_price(setup.symbol, setup.take_profit)
            safe_size = self.client.normalize_size(setup.symbol, setup.size)
            tp_order_type = {
                "trigger": {
                    "triggerPx": safe_tp_price,
                    "isMarket": True,
                    "tpsl": "tp",
                }
            }
            response = self.client.exchange.order(
                name=setup.symbol,
                is_buy=not is_buy,  # Opposite direction to close
                sz=safe_size,
                limit_px=safe_tp_price,
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

    def update_chandelier_exit(
        self, 
        symbol: str, 
        current_high: float, 
        current_low: float,
        current_atr: float | None = None,
    ) -> OrderResult | None:
        """Update Chandelier Exit trailing stop.

        Chandelier Exit Logic:
        - LONG: New SL = max(Prev SL, Highest_High - ATR * atr_trailing_multiplier)
        - SHORT: New SL = min(Prev SL, Lowest_Low + ATR * atr_trailing_multiplier)

        Args:
            symbol: Trading symbol.
            current_high: Current candle high (used to update highest_price for LONG).
            current_low: Current candle low (used to update lowest_price for SHORT).
            current_atr: Current ATR value (optional - uses stored ATR if not provided).

        Returns:
            OrderResult if SL was updated, None otherwise.
        """
        if symbol not in self.active_positions:
            return None

        pos = self.active_positions[symbol]
        
        # Use provided ATR or fall back to stored value
        atr_value = current_atr if current_atr is not None else pos.atr_value

        if pos.side == Side.LONG:
            # Track highest high for long positions
            if current_high > pos.highest_price:
                pos.highest_price = current_high

            # Calculate Chandelier Exit: Highest_High - ATR * multiplier
            trail_distance = atr_value * pos.atr_trailing_multiplier
            new_sl = pos.highest_price - trail_distance

            # Only move SL up, never down (ratchet effect)
            if new_sl > pos.current_sl:
                return self._update_sl_order(pos, new_sl)

        else:  # SHORT
            # Track lowest low for short positions
            if current_low < pos.lowest_price:
                pos.lowest_price = current_low

            # Calculate Chandelier Exit: Lowest_Low + ATR * multiplier
            trail_distance = atr_value * pos.atr_trailing_multiplier
            new_sl = pos.lowest_price + trail_distance

            # Only move SL down, never up (ratchet effect)
            if new_sl < pos.current_sl:
                return self._update_sl_order(pos, new_sl)

        return None

    def adopt_external_position(
        self,
        symbol: str,
        side_str: str,
        entry_price: float,
        size: float,
        current_high: float,
        current_low: float,
        current_atr: float | None = None,
    ) -> OrderResult:
        """Adopt an externally opened position and ensure it has a protective SL."""
        side = Side.LONG if side_str.lower() == "long" else Side.SHORT
        atr_value = current_atr if current_atr is not None and current_atr > 0 else entry_price * 0.02

        # Detect existing SL first.
        sl_check = self.has_existing_sl_order(symbol)
        if sl_check.get("has_sl") and sl_check.get("sl_price") is not None:
            sl_price = float(sl_check["sl_price"])
            sl_order_id = sl_check.get("sl_order_id")
        else:
            sl_price = self.calculate_atr_stop_loss(entry_price, side, atr_value)
            is_buy_close = side == Side.SHORT
            safe_sl_price = self.client.normalize_price(symbol, sl_price)
            safe_size = self.client.normalize_size(symbol, size)
            sl_order_type = {
                "trigger": {
                    "triggerPx": safe_sl_price,
                    "isMarket": True,
                    "tpsl": "sl",
                }
            }
            response = self.client.exchange.order(
                name=symbol,
                is_buy=is_buy_close,
                sz=safe_size,
                limit_px=safe_sl_price,
                order_type=sl_order_type,
                reduce_only=True,
            )
            sl_order_id = self._extract_order_id(response)
            if response.get("status") != "ok" or sl_order_id is None:
                return OrderResult(
                    success=False,
                    message=f"Failed to place SL while adopting external position: {response}",
                    details=response,
                )
            verified = self._verify_trigger_order_visible(
                symbol=symbol,
                order_id=sl_order_id,
                expected_trigger_px=safe_sl_price,
            )
            if not verified:
                return OrderResult(
                    success=False,
                    order_id=sl_order_id,
                    message="External position adopted, but SL is not visible in trigger orders",
                    details=response,
                )
            sl_price = safe_sl_price

        if side == Side.LONG:
            highest_price = max(entry_price, current_high)
            lowest_price = entry_price
        else:
            highest_price = entry_price
            lowest_price = min(entry_price, current_low)

        self.active_positions[symbol] = ActivePosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            current_sl=sl_price,
            atr_value=atr_value,
            atr_trailing_multiplier=self.atr_trailing_multiplier,
            highest_price=highest_price,
            lowest_price=lowest_price,
            sl_order_id=sl_order_id,
        )

        return OrderResult(
            success=True,
            order_id=sl_order_id,
            message=f"External position adopted and protected with SL @ ${sl_price:,.2f}",
        )

    def _update_sl_order(self, pos: ActivePosition, new_sl: float) -> OrderResult:
        """Cancel current SL and recreate a fresh trigger SL at new price."""
        old_sl = pos.current_sl
        try:
            is_buy = pos.side == Side.SHORT  # Opposite direction to close
            safe_new_sl = self.client.normalize_price(pos.symbol, new_sl)
            safe_size = self.client.normalize_size(pos.symbol, pos.size)
            sl_order_type = {
                "trigger": {
                    "triggerPx": safe_new_sl,
                    "isMarket": True,
                    "tpsl": "sl",
                }
            }

            if pos.sl_order_id:
                cancel_response = self.client.cancel_order(pos.symbol, pos.sl_order_id)
                if cancel_response.get("status") != "ok":
                    return OrderResult(
                        success=False,
                        order_id=pos.sl_order_id,
                        message=f"Failed to cancel previous Chandelier SL: {cancel_response}",
                        details=cancel_response,
                    )

            response = self.client.exchange.order(
                name=pos.symbol,
                is_buy=is_buy,
                sz=safe_size,
                limit_px=safe_new_sl,
                order_type=sl_order_type,
                reduce_only=True,
            )

            if response.get("status") == "ok":
                new_order_id = self._extract_order_id(response)
                if new_order_id is None:
                    return OrderResult(
                        success=False,
                        message=f"Failed to update Chandelier Exit SL: invalid response {response}",
                    )
                verified = self._verify_trigger_order_visible(
                    symbol=pos.symbol,
                    order_id=new_order_id,
                    expected_trigger_px=safe_new_sl,
                )
                if not verified:
                    return OrderResult(
                        success=False,
                        order_id=new_order_id,
                        message="Trailing SL recreated but not visible in trigger orders",
                        details=response,
                    )
                pos.current_sl = safe_new_sl
                pos.sl_order_id = new_order_id
                return OrderResult(
                    success=True,
                    order_id=pos.sl_order_id,
                    message=f"Chandelier Exit SL updated: ${old_sl:,.2f} → ${new_sl:,.2f}",
                    details=response,
                )
            else:
                return OrderResult(
                    success=False,
                    message=f"Failed to update Chandelier Exit SL: {response}",
                )
        except Exception as e:
            return OrderResult(
                success=False,
                message=f"Error updating Chandelier Exit SL: {e}",
            )

    def emergency_close_symbol(self, symbol: str, reason: str) -> OrderResult:
        """Public emergency close helper for callers that need symbol-level fail-safe."""
        return self._emergency_close_position(symbol=symbol, reason=reason)

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
                    close_result = self._close_position_with_verification(
                        symbol=pos.symbol,
                        size=pos.size,
                    )
                    results.append(close_result)
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

    def _close_position_with_verification(self, symbol: str, size: float, max_attempts: int = 3) -> OrderResult:
        """Close a position and verify it is actually gone on exchange."""
        last_response = None
        for attempt in range(1, max_attempts + 1):
            response = self.client.exchange.market_close(
                coin=symbol,
                sz=size,
            )
            last_response = response

            if not response or response.get("status") != "ok":
                logger.error("Market close failed | symbol=%s attempt=%d response=%s", symbol, attempt, response)
                continue

            # Hyperliquid state can lag briefly after close.
            time.sleep(0.4)
            if not self._is_position_still_open(symbol):
                return OrderResult(
                    success=True,
                    message=f"Closed position {symbol} (verified)",
                    details=response,
                )

            logger.warning("Position still open after close attempt | symbol=%s attempt=%d", symbol, attempt)

        return OrderResult(
            success=False,
            message=f"Failed to fully close {symbol} after {max_attempts} attempts",
            details=last_response,
        )

    def _is_position_still_open(self, symbol: str) -> bool:
        """Check if a position remains open for a symbol."""
        try:
            open_positions = self.client.get_open_positions()
            return any(pos.symbol == symbol for pos in open_positions)
        except Exception as e:
            logger.error("Failed to verify position status for %s: %s", symbol, e)
            return True

    def emergency_shutdown(self) -> dict:
        """Execute emergency shutdown (kill switch).

        1. Cancel all pending orders
        2. Close all open positions at market
        3. Clear all position tracking

        Returns:
            Dict with results for cancellations and closures.
        """
        cancelled_orders = self.cancel_all_orders()
        close_attempts = self.close_all_positions()
        # Force verification: panic should fail loudly if positions remain open.
        remaining_positions = self.client.get_open_positions()
        self.active_positions.clear()

        return {
            "cancelled_orders": cancelled_orders,
            "closed_positions": close_attempts,
            "remaining_positions": remaining_positions,
            "panic_success": len(remaining_positions) == 0,
        }
