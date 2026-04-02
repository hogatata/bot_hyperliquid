"""Risk manager for position sizing, SL/TP, and leverage."""

from dataclasses import dataclass
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


@dataclass
class OrderResult:
    """Result from order execution."""

    success: bool
    order_id: Optional[int] = None
    message: str = ""
    details: Optional[dict] = None


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
        """
        self.client = client
        self.position_size_percent = position_size_percent
        self.default_leverage = default_leverage
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.use_atr_for_sl = use_atr_for_sl
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier

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
        )

    def execute_trade(
        self,
        setup: TradeSetup,
        dry_run: bool = False,
    ) -> dict[str, OrderResult]:
        """Execute a complete trade with SL/TP orders.

        Flow:
        1. Set leverage (isolated margin)
        2. Place main market order
        3. Place stop loss order
        4. Place take profit order

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

        # Step 2: Place main market order
        if dry_run:
            results["entry"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would open {setup.side.value.upper()} {setup.size} {setup.symbol} @ ~${setup.entry_price:,.2f}",
            )
        else:
            try:
                response = self.client.exchange.market_open(
                    name=setup.symbol,
                    is_buy=is_buy,
                    sz=setup.size,
                )
                # Extract order ID from response
                order_id = None
                if response.get("status") == "ok":
                    statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                    if statuses and "resting" in statuses[0]:
                        order_id = statuses[0]["resting"]["oid"]
                    elif statuses and "filled" in statuses[0]:
                        order_id = statuses[0]["filled"]["oid"]

                results["entry"] = OrderResult(
                    success=response.get("status") == "ok",
                    order_id=order_id,
                    message=f"Opened {setup.side.value.upper()} {setup.size} {setup.symbol}",
                    details=response,
                )
            except Exception as e:
                results["entry"] = OrderResult(
                    success=False,
                    message=f"Failed to place entry order: {e}",
                )
                return results  # Abort if entry fails

        # Step 3: Place Stop Loss order (trigger order, reduce_only)
        if dry_run:
            results["stop_loss"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would place SL @ ${setup.stop_loss:,.2f}",
            )
        else:
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
                order_id = None
                if response.get("status") == "ok":
                    statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                    if statuses and "resting" in statuses[0]:
                        order_id = statuses[0]["resting"]["oid"]

                results["stop_loss"] = OrderResult(
                    success=response.get("status") == "ok",
                    order_id=order_id,
                    message=f"SL placed @ ${setup.stop_loss:,.2f}",
                    details=response,
                )
            except Exception as e:
                results["stop_loss"] = OrderResult(
                    success=False,
                    message=f"Failed to place SL: {e}",
                )

        # Step 4: Place Take Profit order (trigger order, reduce_only)
        if dry_run:
            results["take_profit"] = OrderResult(
                success=True,
                message=f"[DRY RUN] Would place TP @ ${setup.take_profit:,.2f}",
            )
        else:
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
                order_id = None
                if response.get("status") == "ok":
                    statuses = response.get("response", {}).get("data", {}).get("statuses", [])
                    if statuses and "resting" in statuses[0]:
                        order_id = statuses[0]["resting"]["oid"]

                results["take_profit"] = OrderResult(
                    success=response.get("status") == "ok",
                    order_id=order_id,
                    message=f"TP placed @ ${setup.take_profit:,.2f}",
                    details=response,
                )
            except Exception as e:
                results["take_profit"] = OrderResult(
                    success=False,
                    message=f"Failed to place TP: {e}",
                )

        return results

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

        Returns:
            Dict with results for cancellations and closures.
        """
        return {
            "cancelled_orders": self.cancel_all_orders(),
            "closed_positions": self.close_all_positions(),
        }
