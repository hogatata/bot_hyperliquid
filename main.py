"""Main entry point for the Hyperliquid trading bot.

This bot implements a multi-timeframe strategy:
- Daily trend: Price vs SMA determines direction (LONG/SHORT only)
- Intraday signals: VWAP crossover + RSI for entry timing

Features:
- Configurable via config.json and .env
- Isolated margin with bracket orders (SL/TP)
- Limit orders for maker rebates (optional)
- Trailing stop loss (optional)
- Macro filters: funding rate, volatility (optional)
- Kill switch (Ctrl+C) for emergency shutdown
- Live terminal dashboard
"""

import signal
import sys
import time
from datetime import datetime

from src.config import load_settings
from src.exchange import HyperliquidClient
from src.risk import RiskManager, Side
from src.strategy import Signal, SignalGenerator, add_atr
from src.utils.logger import setup_logger
from src.utils.notifier import TelegramNotifier

# Global flag for graceful shutdown
shutdown_requested = False
risk_manager_global = None  # For signal handler access
notifier_global = None  # For signal handler access


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def print_dashboard(
    symbol: str,
    price: float,
    trend: str,
    rsi: float,
    rsi_zone: str,
    vwap: float,
    ma_value: float,
    signal_reason: str,
    position_info: str,
    balance: float,
    is_testnet: bool,
    last_update: str,
    funding_rate: float = 0.0,
    trailing_sl: float | None = None,
    features: dict | None = None,
):
    """Print a clean terminal dashboard."""
    # Color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    trend_color = GREEN if trend == "bullish" else RED if trend == "bearish" else YELLOW
    rsi_color = RED if rsi_zone == "overbought" else GREEN if rsi_zone == "oversold" else RESET
    funding_color = RED if abs(funding_rate) > 0.01 else RESET

    mode = f"{RED}⚠ MAINNET{RESET}" if not is_testnet else f"{GREEN}TESTNET{RESET}"

    # Build features string
    features = features or {}
    feature_list = []
    if features.get("limit_orders"):
        feature_list.append("LIMIT")
    if features.get("trailing_stop"):
        feature_list.append("TRAIL")
    if features.get("funding_filter"):
        feature_list.append("FUND")
    if features.get("volatility_filter"):
        feature_list.append("VOL")
    features_str = " | ".join(feature_list) if feature_list else "MARKET"

    # Trailing SL info
    trailing_info = f" (Trail: ${trailing_sl:,.2f})" if trailing_sl else ""

    print(f"""
{BOLD}
╔══════════════════════════════════════════════════════════════╗
║          HYPERLIQUID TRADING BOT                             ║
╠══════════════════════════════════════════════════════════════╣{RESET}
║ Mode: {mode}  Features: {CYAN}{features_str:15}{RESET}  Last: {last_update}  ║
╠══════════════════════════════════════════════════════════════╣
║ {BOLD}MARKET DATA{RESET}                                                  ║
║  Symbol:     {CYAN}{symbol:10}{RESET}                                      ║
║  Price:      {BOLD}${price:>12,.2f}{RESET}                                   ║
║  Trend:      {trend_color}{trend.upper():10}{RESET} (vs MA: ${ma_value:,.2f})                  ║
║  VWAP:       ${vwap:>12,.2f}                                   ║
║  RSI:        {rsi_color}{rsi:>6.2f}{RESET} ({rsi_zone})                                ║
║  Funding:    {funding_color}{funding_rate:>+7.4f}%{RESET}                                       ║
╠══════════════════════════════════════════════════════════════╣
║ {BOLD}STRATEGY{RESET}                                                     ║
║  {signal_reason:<60}║
╠══════════════════════════════════════════════════════════════╣
║ {BOLD}ACCOUNT{RESET}                                                      ║
║  Balance:    ${balance:>12,.2f}                                   ║
║  Position:   {position_info:<47} ║{trailing_info}
╠══════════════════════════════════════════════════════════════╣
║ {YELLOW}Press Ctrl+C for EMERGENCY SHUTDOWN (cancels all orders){RESET}     ║
╚══════════════════════════════════════════════════════════════╝
""")


def kill_switch_handler(signum, frame):
    """Handle Ctrl+C for emergency shutdown."""
    global shutdown_requested, risk_manager_global, notifier_global

    shutdown_requested = True

    print("\n")
    print("=" * 60)
    print("🚨 KILL SWITCH ACTIVATED - EMERGENCY SHUTDOWN 🚨")
    print("=" * 60)

    if risk_manager_global:
        print("\n[1/2] Cancelling all pending orders...")
        try:
            results = risk_manager_global.emergency_shutdown()

            cancelled = results.get("cancelled_orders", [])
            closed = results.get("closed_positions", [])

            for result in cancelled:
                status = "✓" if result.success else "✗"
                print(f"   {status} {result.message}")

            print("\n[2/2] Closing all open positions...")
            for result in closed:
                status = "✓" if result.success else "✗"
                print(f"   {status} {result.message}")

            print(f"\n✓ Cancelled {len(cancelled)} orders")
            print(f"✓ Closed {len(closed)} positions")

        except Exception as e:
            print(f"✗ Error during shutdown: {e}")

    # Send Telegram notification
    if notifier_global:
        notifier_global.notify_shutdown("🚨 Emergency Shutdown Triggered!")

    print("\n" + "=" * 60)
    print("Shutdown complete. Goodbye!")
    print("=" * 60)
    sys.exit(0)


def run_bot():
    """Main bot loop."""
    global risk_manager_global, notifier_global

    # Register kill switch (Ctrl+C handler)
    signal.signal(signal.SIGINT, kill_switch_handler)
    signal.signal(signal.SIGTERM, kill_switch_handler)

    # Load configuration
    try:
        settings = load_settings()
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Configuration error: {e}")
        print("Please check your .env and config.json files.")
        sys.exit(1)

    # Setup logger
    logger = setup_logger("bot", settings.bot.log_level)

    # Initialize client
    logger.info("Initializing Hyperliquid client...")
    try:
        client = HyperliquidClient(
            private_key=settings.private_key,
            wallet_address=settings.wallet_address,
            is_testnet=settings.is_testnet,
        )

        if not client.is_connected():
            logger.error("Failed to connect to Hyperliquid API")
            sys.exit(1)

        logger.info(f"✓ Connected to {'TESTNET' if settings.is_testnet else 'MAINNET'}")

    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        sys.exit(1)

    # Initialize signal generator with macro filters
    signal_gen = SignalGenerator(
        daily_ma_type=settings.strategy.daily_ma_type,
        daily_ma_period=settings.strategy.daily_ma_period,
        rsi_period=settings.strategy.rsi_period,
        rsi_oversold=settings.strategy.rsi_oversold,
        rsi_overbought=settings.strategy.rsi_overbought,
        vwap_enabled=settings.strategy.vwap_enabled,
        # Macro filters
        funding_filter_enabled=settings.filters.funding_filter_enabled,
        funding_threshold=settings.filters.funding_threshold,
        volatility_filter_enabled=settings.filters.volatility_filter_enabled,
        volatility_atr_period=settings.filters.volatility_atr_period,
        volatility_lookback=settings.filters.volatility_lookback,
        volatility_threshold=settings.filters.volatility_threshold,
    )
    logger.info(f"✓ Signal generator initialized ({settings.strategy.daily_ma_type}{settings.strategy.daily_ma_period})")

    # Initialize risk manager with new features
    risk_manager = RiskManager(
        client=client,
        position_size_percent=settings.trading.position_size_percent,
        default_leverage=settings.trading.leverage,
        stop_loss_percent=settings.risk.stop_loss_percent,
        take_profit_percent=settings.risk.take_profit_percent,
        use_atr_for_sl=settings.risk.use_atr_for_sl,
        atr_sl_multiplier=settings.risk.atr_multiplier,
        atr_tp_multiplier=settings.risk.atr_multiplier * 2,  # 2:1 R:R by default
        # New features
        use_limit_orders=settings.risk.use_limit_orders,
        limit_order_timeout=settings.risk.limit_order_timeout,
        trailing_stop_enabled=settings.risk.trailing_stop_enabled,
        trailing_atr_multiplier=settings.risk.trailing_atr_multiplier,
    )
    risk_manager_global = risk_manager  # For signal handler
    logger.info(f"✓ Risk manager initialized ({settings.trading.leverage}x leverage)")

    # ==========================================================================
    # STATE RECOVERY - Check for existing positions on startup
    # ==========================================================================
    logger.info("Checking for existing positions...")
    recovery_results = risk_manager.recover_state_on_startup(
        symbols=settings.trading.symbols,
        atr_period=settings.strategy.rsi_period,  # Use same as RSI for consistency
        candle_interval=settings.strategy.intraday_timeframe,
    )

    recovered_positions = 0
    for symbol, result in recovery_results.items():
        if result.get("recovered"):
            recovered_positions += 1
            pos = result.get("position")
            active = result.get("active_position")
            logger.info(f"  🔄 {symbol}: {result['message']}")
            logger.info(f"     Size: {pos.size} | PnL: ${pos.unrealized_pnl:,.2f}")
            if active:
                if active.side == Side.LONG:
                    logger.info(f"     Highest: ${active.highest_price:,.2f} | Current SL: ${active.current_sl:,.2f}")
                else:
                    logger.info(f"     Lowest: ${active.lowest_price:,.2f} | Current SL: ${active.current_sl:,.2f}")
                if result.get("existing_sl_order"):
                    logger.info(f"     ✓ Existing SL order synced (ID: {result['existing_sl_order']})")
                if result.get("existing_tp_order"):
                    logger.info(f"     ✓ Existing TP order synced (ID: {result['existing_tp_order']})")
        elif result.get("position"):
            # Position exists but recovery failed
            logger.warning(f"  ⚠ {symbol}: {result['message']}")
    
    if recovered_positions > 0:
        logger.info(f"✓ Recovered {recovered_positions} existing position(s)")
    else:
        logger.info("✓ No existing positions found - clean start")

    # Build features dict for dashboard
    active_features = {
        "limit_orders": settings.risk.use_limit_orders,
        "trailing_stop": settings.risk.trailing_stop_enabled,
        "funding_filter": settings.filters.funding_filter_enabled,
        "volatility_filter": settings.filters.volatility_filter_enabled,
    }

    # Initialize Telegram notifier
    notifier = TelegramNotifier(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
        enabled=settings.notifications.enable_telegram_alerts,
    )
    notifier_global = notifier  # For signal handler
    
    if notifier.enabled:
        logger.info("✓ Telegram notifications enabled")
    else:
        logger.info("ℹ Telegram notifications disabled")

    # Display startup info
    logger.info("")
    logger.info("=" * 50)
    logger.info("BOT CONFIGURATION")
    logger.info("=" * 50)
    logger.info(f"  Symbols:        {', '.join(settings.trading.symbols)}")
    logger.info(f"  Timeframe:      {settings.strategy.intraday_timeframe}")
    logger.info(f"  Leverage:       {settings.trading.leverage}x (ISOLATED)")
    logger.info(f"  Position Size:  {settings.trading.position_size_percent}% of account")
    logger.info(f"  Stop Loss:      {settings.risk.stop_loss_percent}%")
    logger.info(f"  Take Profit:    {settings.risk.take_profit_percent}%")
    logger.info(f"  Loop Interval:  {settings.bot.loop_interval_seconds}s")
    logger.info("-" * 50)
    logger.info("ADVANCED FEATURES:")
    logger.info(f"  Limit Orders:   {'✓ ENABLED' if settings.risk.use_limit_orders else '✗ Disabled'}")
    logger.info(f"  Trailing Stop:  {'✓ ENABLED' if settings.risk.trailing_stop_enabled else '✗ Disabled'}")
    logger.info(f"  Funding Filter: {'✓ ENABLED' if settings.filters.funding_filter_enabled else '✗ Disabled'}")
    logger.info(f"  Vol Filter:     {'✓ ENABLED' if settings.filters.volatility_filter_enabled else '✗ Disabled'}")
    logger.info(f"  Telegram:       {'✓ ENABLED' if notifier.enabled else '✗ Disabled'}")
    logger.info("=" * 50)
    logger.info("Press Ctrl+C to activate KILL SWITCH")
    logger.info("Starting main loop in 3 seconds...")
    
    # Send startup notification
    notifier.notify_startup(
        is_testnet=settings.is_testnet,
        symbols=settings.trading.symbols,
        leverage=settings.trading.leverage,
        features=active_features,
    )
    
    time.sleep(3)

    # Main loop
    loop_count = 0
    while not shutdown_requested:
        loop_count += 1

        try:
            for symbol in settings.trading.symbols:
                # Fetch candle data
                # We need enough candles for indicators (MA period + buffer)
                candle_limit = max(settings.strategy.daily_ma_period * 2, 100)

                df = client.get_candles(
                    symbol=symbol,
                    interval=settings.strategy.intraday_timeframe,
                    limit=candle_limit,
                )

                if df.empty:
                    logger.warning(f"[{symbol}] No candle data received")
                    continue

                # Add indicators
                df = signal_gen.add_indicators(df)

                # Add ATR if using for SL or trailing stop
                if settings.risk.use_atr_for_sl or settings.risk.trailing_stop_enabled:
                    df = add_atr(df, period=settings.risk.atr_period)

                # Get funding rate for macro filter
                funding_rate = 0.0
                if settings.filters.funding_filter_enabled:
                    funding_rate = client.get_funding_rate(symbol)

                # Analyze for signals (with funding rate for macro filter)
                result = signal_gen.analyze(df, funding_rate=funding_rate)

                # Get account info
                balance = client.get_account_balance()
                positions = client.get_open_positions()

                # Check if we have an open position for this symbol
                current_position = None
                for pos in positions:
                    if pos.symbol == symbol:
                        current_position = pos
                        break

                # Handle trailing stop for existing positions
                trailing_sl_price = None
                if current_position and settings.risk.trailing_stop_enabled:
                    current_price = result.current_price
                    trail_result = risk_manager.update_trailing_stop(symbol, current_price)
                    if trail_result and trail_result.success:
                        logger.info(f"📈 {trail_result.message}")
                    # Get current trailing SL for display
                    if symbol in risk_manager.active_positions:
                        trailing_sl_price = risk_manager.active_positions[symbol].current_sl
                elif not current_position and symbol in risk_manager.active_positions:
                    # Position closed (hit SL/TP), send notification and clean up tracking
                    closed_pos = risk_manager.active_positions[symbol]
                    
                    # Determine exit reason and calculate PnL
                    exit_reason = "stop_loss"  # Default, could be SL or TP
                    exit_price = result.current_price  # Approximate
                    
                    if closed_pos.side == Side.LONG:
                        pnl = (exit_price - closed_pos.entry_price) * closed_pos.size
                        pnl_percent = ((exit_price - closed_pos.entry_price) / closed_pos.entry_price) * 100
                        if exit_price >= closed_pos.entry_price * 1.01:  # Likely TP
                            exit_reason = "take_profit"
                    else:
                        pnl = (closed_pos.entry_price - exit_price) * closed_pos.size
                        pnl_percent = ((closed_pos.entry_price - exit_price) / closed_pos.entry_price) * 100
                        if exit_price <= closed_pos.entry_price * 0.99:  # Likely TP for short
                            exit_reason = "take_profit"
                    
                    if settings.risk.trailing_stop_enabled and closed_pos.current_sl != closed_pos.initial_sl:
                        exit_reason = "trailing_stop"
                    
                    # Send notification
                    notifier.notify_trade_closed(
                        symbol=symbol,
                        side=closed_pos.side.value.upper(),
                        entry_price=closed_pos.entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        exit_reason=exit_reason,
                    )
                    
                    logger.info(f"📊 Position closed: {symbol} | PnL: ${pnl:+,.2f}")
                    risk_manager.clear_position_tracking(symbol)

                # Format position info
                if current_position:
                    pnl_sign = "+" if current_position.unrealized_pnl >= 0 else ""
                    position_info = (
                        f"{current_position.side.upper()} {current_position.size} @ "
                        f"${current_position.entry_price:,.2f} "
                        f"(PnL: {pnl_sign}${current_position.unrealized_pnl:,.2f})"
                    )
                else:
                    position_info = "No open position"

                # Print dashboard
                clear_screen()
                print_dashboard(
                    symbol=symbol,
                    price=result.current_price,
                    trend=result.trend,
                    rsi=result.rsi_value,
                    rsi_zone=result.rsi_zone,
                    vwap=result.vwap_value,
                    ma_value=result.ma_value,
                    signal_reason=result.reason,
                    position_info=position_info,
                    balance=balance.account_value,
                    is_testnet=settings.is_testnet,
                    last_update=datetime.now().strftime("%H:%M:%S"),
                    funding_rate=funding_rate,
                    trailing_sl=trailing_sl_price,
                    features=active_features,
                )

                # Execute trade if signal and no position
                if result.signal != Signal.NO_SIGNAL and current_position is None:
                    side = Side.LONG if result.signal == Signal.LONG else Side.SHORT
                    order_type = "LIMIT" if settings.risk.use_limit_orders else "MARKET"

                    print(f"\n🚀 SIGNAL DETECTED: {order_type} {side.value.upper()} {symbol}")
                    print("-" * 40)

                    # Prepare trade
                    try:
                        setup = risk_manager.prepare_trade(
                            symbol=symbol,
                            side=side,
                            leverage=settings.trading.leverage,
                            df_with_indicators=df if (settings.risk.use_atr_for_sl or settings.risk.trailing_stop_enabled) else None,
                        )

                        print(f"   Entry:       ${setup.entry_price:,.2f}")
                        print(f"   Size:        {setup.size} {symbol}")
                        print(f"   Stop Loss:   ${setup.stop_loss:,.2f}")
                        print(f"   Take Profit: ${setup.take_profit:,.2f}")
                        print(f"   Risk:        ${setup.risk_amount:,.2f}")
                        print(f"   Reward:      ${setup.potential_profit:,.2f}")
                        if settings.risk.trailing_stop_enabled:
                            print(f"   Trailing:    ✓ ATR×{settings.risk.trailing_atr_multiplier}")
                        print("-" * 40)

                        # Execute trade
                        results = risk_manager.execute_trade(setup, dry_run=False)

                        for step, order_result in results.items():
                            status = "✓" if order_result.success else "✗"
                            print(f"   {status} {step}: {order_result.message}")

                        if all(r.success for r in results.values()):
                            logger.info(f"✓ Trade executed: {order_type} {side.value.upper()} {symbol}")
                            
                            # Send Telegram notification for trade opened
                            notifier.notify_trade_opened(
                                symbol=symbol,
                                side=side.value.upper(),
                                entry_price=setup.entry_price,
                                size=setup.size,
                                stop_loss=setup.stop_loss,
                                take_profit=setup.take_profit,
                                leverage=setup.leverage,
                                order_type=order_type,
                            )
                        else:
                            logger.error(f"✗ Trade partially failed for {symbol}")

                    except Exception as e:
                        logger.error(f"Failed to execute trade: {e}")
                        notifier.notify_error(str(e), context=f"Trade execution for {symbol}")

                    # Pause after trade execution
                    time.sleep(5)

            # Wait for next iteration
            time.sleep(settings.bot.loop_interval_seconds)

        except KeyboardInterrupt:
            # This shouldn't trigger due to signal handler, but just in case
            kill_switch_handler(None, None)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(10)  # Wait before retry


def main():
    """Entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     HYPERLIQUID ALGORITHMIC TRADING BOT                   ║
    ║     ─────────────────────────────────────                 ║
    ║     Multi-timeframe trend following strategy              ║
    ║     Daily SMA + Intraday VWAP/RSI signals                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    try:
        run_bot()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
