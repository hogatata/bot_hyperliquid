"""Main entry point for the Hyperliquid trading bot.

This bot implements a multi-timeframe strategy:
- Daily trend: Price vs SMA determines direction (LONG/SHORT only)
- Intraday signals: VWAP crossover + RSI for entry timing

Features:
- Configurable via config.json and .env
- Isolated margin with bracket orders (SL/TP)
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

# Global flag for graceful shutdown
shutdown_requested = False
risk_manager_global = None  # For signal handler access


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

    mode = f"{RED}⚠ MAINNET{RESET}" if not is_testnet else f"{GREEN}TESTNET{RESET}"

    print(f"""
{BOLD}
╔══════════════════════════════════════════════════════════════╗
║          HYPERLIQUID TRADING BOT                             ║
╠══════════════════════════════════════════════════════════════╣{RESET}
║ Mode: {mode}                          Last: {last_update}        ║
╠══════════════════════════════════════════════════════════════╣
║ {BOLD}MARKET DATA{RESET}                                                  ║
║  Symbol:     {CYAN}{symbol:10}{RESET}                                      ║
║  Price:      {BOLD}${price:>12,.2f}{RESET}                                   ║
║  Trend:      {trend_color}{trend.upper():10}{RESET} (vs MA: ${ma_value:,.2f})                  ║
║  VWAP:       ${vwap:>12,.2f}                                   ║
║  RSI:        {rsi_color}{rsi:>6.2f}{RESET} ({rsi_zone})                                ║
╠══════════════════════════════════════════════════════════════╣
║ {BOLD}STRATEGY{RESET}                                                     ║
║  {signal_reason:<60}║
╠══════════════════════════════════════════════════════════════╣
║ {BOLD}ACCOUNT{RESET}                                                      ║
║  Balance:    ${balance:>12,.2f}                                   ║
║  Position:   {position_info:<47} ║
╠══════════════════════════════════════════════════════════════╣
║ {YELLOW}Press Ctrl+C for EMERGENCY SHUTDOWN (cancels all orders){RESET}     ║
╚══════════════════════════════════════════════════════════════╝
""")


def kill_switch_handler(signum, frame):
    """Handle Ctrl+C for emergency shutdown."""
    global shutdown_requested, risk_manager_global

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

    print("\n" + "=" * 60)
    print("Shutdown complete. Goodbye!")
    print("=" * 60)
    sys.exit(0)


def run_bot():
    """Main bot loop."""
    global risk_manager_global

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

    # Initialize signal generator
    signal_gen = SignalGenerator(
        daily_ma_type=settings.strategy.daily_ma_type,
        daily_ma_period=settings.strategy.daily_ma_period,
        rsi_period=settings.strategy.rsi_period,
        rsi_oversold=settings.strategy.rsi_oversold,
        rsi_overbought=settings.strategy.rsi_overbought,
        vwap_enabled=settings.strategy.vwap_enabled,
    )
    logger.info(f"✓ Signal generator initialized ({settings.strategy.daily_ma_type}{settings.strategy.daily_ma_period})")

    # Initialize risk manager
    risk_manager = RiskManager(
        client=client,
        position_size_percent=settings.trading.position_size_percent,
        default_leverage=settings.trading.leverage,
        stop_loss_percent=settings.risk.stop_loss_percent,
        take_profit_percent=settings.risk.take_profit_percent,
        use_atr_for_sl=settings.risk.use_atr_for_sl,
        atr_sl_multiplier=settings.risk.atr_multiplier,
        atr_tp_multiplier=settings.risk.atr_multiplier * 2,  # 2:1 R:R by default
    )
    risk_manager_global = risk_manager  # For signal handler
    logger.info(f"✓ Risk manager initialized ({settings.trading.leverage}x leverage)")

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
    logger.info("=" * 50)
    logger.info("Press Ctrl+C to activate KILL SWITCH")
    logger.info("Starting main loop in 3 seconds...")
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

                # Add ATR if using for SL
                if settings.risk.use_atr_for_sl:
                    df = add_atr(df, period=settings.risk.atr_period)

                # Analyze for signals
                result = signal_gen.analyze(df)

                # Get account info
                balance = client.get_account_balance()
                positions = client.get_open_positions()

                # Check if we have an open position for this symbol
                current_position = None
                for pos in positions:
                    if pos.symbol == symbol:
                        current_position = pos
                        break

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
                )

                # Execute trade if signal and no position
                if result.signal != Signal.NO_SIGNAL and current_position is None:
                    side = Side.LONG if result.signal == Signal.LONG else Side.SHORT

                    print(f"\n🚀 SIGNAL DETECTED: {side.value.upper()} {symbol}")
                    print("-" * 40)

                    # Prepare trade
                    try:
                        setup = risk_manager.prepare_trade(
                            symbol=symbol,
                            side=side,
                            leverage=settings.trading.leverage,
                            df_with_indicators=df if settings.risk.use_atr_for_sl else None,
                        )

                        print(f"   Entry:       ${setup.entry_price:,.2f}")
                        print(f"   Size:        {setup.size} {symbol}")
                        print(f"   Stop Loss:   ${setup.stop_loss:,.2f}")
                        print(f"   Take Profit: ${setup.take_profit:,.2f}")
                        print(f"   Risk:        ${setup.risk_amount:,.2f}")
                        print(f"   Reward:      ${setup.potential_profit:,.2f}")
                        print("-" * 40)

                        # Execute trade
                        results = risk_manager.execute_trade(setup, dry_run=False)

                        for step, order_result in results.items():
                            status = "✓" if order_result.success else "✗"
                            print(f"   {status} {step}: {order_result.message}")

                        if all(r.success for r in results.values()):
                            logger.info(f"✓ Trade executed: {side.value.upper()} {symbol}")
                        else:
                            logger.error(f"✗ Trade partially failed for {symbol}")

                    except Exception as e:
                        logger.error(f"Failed to execute trade: {e}")

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
