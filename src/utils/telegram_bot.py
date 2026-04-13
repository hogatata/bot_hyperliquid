"""Interactive Telegram Bot for trading bot control.

Provides commands to monitor and control the trading bot remotely:
- /status - Account balance, positions, daily PnL
- /config - Current configuration summary
- /pause - Pause new trade entries (keeps managing existing positions)
- /resume - Resume trading
- /panic - Emergency close all positions

SECURITY: Only responds to the authorized TELEGRAM_CHAT_ID.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BotState:
    """Shared state between main loop and Telegram bot."""
    
    is_paused: bool = False
    client: object = None  # HyperliquidClient
    risk_manager: object = None  # RiskManager
    settings: object = None  # Settings
    notifier: object = None  # TelegramNotifier
    last_algo_update: str = "N/A"
    last_market_state: dict = None  # Per-symbol snapshot from main loop


# Global state shared with main loop
bot_state = BotState()


def is_authorized(chat_id: int, authorized_chat_id: str) -> bool:
    """Check if the message is from the authorized chat ID."""
    return str(chat_id) == str(authorized_chat_id)


async def cmd_status(update, context) -> None:
    """Handle /status command - Show account balance and positions."""
    from telegram import Update
    
    chat_id = update.effective_chat.id
    authorized_id = context.bot_data.get("authorized_chat_id", "")
    
    if not is_authorized(chat_id, authorized_id):
        logger.warning(f"Unauthorized /status attempt from chat_id: {chat_id}")
        return  # Silently ignore unauthorized users
    
    try:
        client = bot_state.client
        if client is None:
            await update.message.reply_text("❌ Bot not fully initialized yet.")
            return
        
        # Get account info
        balance = client.get_account_balance()
        positions = client.get_open_positions()
        risk_manager = bot_state.risk_manager
        market_state = bot_state.last_market_state or {}
        
        # Build message
        pause_status = "⏸️ PAUSED" if bot_state.is_paused else "▶️ RUNNING"
        
        msg = (
            f"<b>📊 Bot Status: {pause_status}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>💰 Account Balance:</b>\n"
            f"  • Value: <b>${balance.account_value:,.2f}</b>\n"
            f"  • Margin Used: ${balance.total_margin_used:,.2f}\n"
            f"  • Withdrawable: ${balance.withdrawable:,.2f}\n\n"
        )
        
        if positions:
            msg += "<b>📈 Open Positions:</b>\n"
            total_pnl = 0.0
            for pos in positions:
                pnl_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                side_emoji = "📈" if pos.side == "long" else "📉"
                pnl_percent = (pos.unrealized_pnl / balance.account_value * 100) if balance.account_value > 0 else 0.0

                sl_text = "N/A"
                if risk_manager is not None:
                    if pos.symbol in risk_manager.active_positions:
                        sl_text = f"${risk_manager.active_positions[pos.symbol].current_sl:,.2f}"
                    else:
                        sl_check = risk_manager.has_existing_sl_order(pos.symbol)
                        if sl_check.get("has_sl") and sl_check.get("sl_price") is not None:
                            sl_text = f"${float(sl_check['sl_price']):,.2f}"
                if sl_text == "N/A":
                    sl_text = "N/A"

                msg += (
                    f"\n{side_emoji} <b>{pos.symbol}</b> ({pos.side.upper()})\n"
                    f"  • Entry: ${pos.entry_price:,.2f}\n"
                    f"  • Size: {pos.size}\n"
                    f"  • PnL: {pnl_emoji} ${pos.unrealized_pnl:+,.2f} ({pnl_percent:+.2f}%)\n"
                    f"  • Stop Loss: {sl_text}\n"
                )
                total_pnl += pos.unrealized_pnl
            
            msg += f"\n<b>Total Unrealized PnL: ${total_pnl:+,.2f}</b>\n"
        else:
            msg += "<i>No open positions</i>\n"

        msg += (
            "\n<b>🕒 Last Algorithm Update:</b>\n"
            f"  • Time: {bot_state.last_algo_update}\n"
        )

        if market_state:
            msg += "\n<b>🌍 Market State (Last Cycle):</b>\n"
            for symbol, snap in market_state.items():
                msg += (
                    f"\n<b>{symbol}</b>\n"
                    f"  • Price: ${snap.get('price', 0):,.2f}\n"
                    f"  • Trend: {str(snap.get('trend', 'N/A')).upper()}\n"
                    f"  • RSI: {snap.get('rsi', 'N/A')}\n"
                    f"  • Signal: {snap.get('signal_reason', 'N/A')}\n"
                )
        
        await update.message.reply_text(msg, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in /status command: {e}")
        await update.message.reply_text(f"❌ Error fetching status: {e}")


async def cmd_config(update, context) -> None:
    """Handle /config command - Show current configuration."""
    from telegram import Update
    
    chat_id = update.effective_chat.id
    authorized_id = context.bot_data.get("authorized_chat_id", "")
    
    if not is_authorized(chat_id, authorized_id):
        logger.warning(f"Unauthorized /config attempt from chat_id: {chat_id}")
        return
    
    try:
        settings = bot_state.settings
        if settings is None:
            await update.message.reply_text("❌ Bot not fully initialized yet.")
            return
        
        # Build config summary
        msg = (
            "<b>⚙️ Current Configuration</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "<b>📊 Strategy:</b>\n"
            f"  • MA: {settings.strategy.daily_ma_type}{settings.strategy.daily_ma_period}\n"
            f"  • RSI: {settings.strategy.rsi_period} ({settings.strategy.rsi_oversold}/{settings.strategy.rsi_overbought})\n"
            f"  • VWAP: {'✓' if settings.strategy.vwap_enabled else '✗'}\n"
            f"  • Timeframe: {settings.strategy.intraday_timeframe}\n\n"
            
            "<b>📐 Risk Management (ATR Dynamic):</b>\n"
            f"  • Risk/Trade: {settings.risk.risk_percent_per_trade}%\n"
            f"  • Max Leverage: {settings.trading.max_leverage}x\n"
            f"  • ATR Period: {settings.risk.atr_period}\n"
            f"  • ATR SL Multiplier: {settings.risk.atr_sl_multiplier}x\n"
            f"  • ATR Trail Multiplier: {settings.risk.atr_trailing_multiplier}x\n"
            f"  • Exit Strategy: Chandelier Exit (dynamic)\n\n"
            
            "<b>🛡️ Filters:</b>\n"
            f"  • Funding Filter: {'✓' if settings.filters.funding_filter_enabled else '✗'}\n"
            f"  • Volatility Filter: {'✓' if settings.filters.volatility_filter_enabled else '✗'}\n\n"
            
            "<b>🔧 Trading:</b>\n"
            f"  • Symbols: {', '.join(settings.trading.symbols)}\n"
            f"  • Limit Orders: {'✓' if settings.risk.use_limit_orders else '✗'}\n"
            f"  • Loop Interval: {settings.bot.loop_interval_seconds}s\n"
        )
        
        await update.message.reply_text(msg, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error in /config command: {e}")
        await update.message.reply_text(f"❌ Error fetching config: {e}")


async def cmd_pause(update, context) -> None:
    """Handle /pause command - Pause new trade entries."""
    from telegram import Update
    
    chat_id = update.effective_chat.id
    authorized_id = context.bot_data.get("authorized_chat_id", "")
    
    if not is_authorized(chat_id, authorized_id):
        logger.warning(f"Unauthorized /pause attempt from chat_id: {chat_id}")
        return
    
    if bot_state.is_paused:
        await update.message.reply_text("⏸️ Bot is already paused.")
        return
    
    bot_state.is_paused = True
    logger.info("Bot PAUSED via Telegram command")
    
    msg = (
        "<b>⏸️ Bot PAUSED</b>\n\n"
        "New trade entries are now disabled.\n"
        "Existing positions will continue to be managed.\n\n"
        "Use /resume to resume trading."
    )
    await update.message.reply_text(msg, parse_mode="HTML")
    
    # Also send notification via notifier
    if bot_state.notifier and bot_state.notifier.enabled:
        bot_state.notifier._send_message("⏸️ Bot PAUSED via Telegram command")


async def cmd_resume(update, context) -> None:
    """Handle /resume command - Resume trading."""
    from telegram import Update
    
    chat_id = update.effective_chat.id
    authorized_id = context.bot_data.get("authorized_chat_id", "")
    
    if not is_authorized(chat_id, authorized_id):
        logger.warning(f"Unauthorized /resume attempt from chat_id: {chat_id}")
        return
    
    if not bot_state.is_paused:
        await update.message.reply_text("▶️ Bot is already running.")
        return
    
    bot_state.is_paused = False
    logger.info("Bot RESUMED via Telegram command")
    
    msg = (
        "<b>▶️ Bot RESUMED</b>\n\n"
        "Trading is now active.\n"
        "The bot will look for new entry signals."
    )
    await update.message.reply_text(msg, parse_mode="HTML")
    
    # Also send notification via notifier
    if bot_state.notifier and bot_state.notifier.enabled:
        bot_state.notifier._send_message("▶️ Bot RESUMED via Telegram command")


async def cmd_panic(update, context) -> None:
    """Handle /panic command - Emergency close all positions."""
    from telegram import Update
    
    chat_id = update.effective_chat.id
    authorized_id = context.bot_data.get("authorized_chat_id", "")
    
    if not is_authorized(chat_id, authorized_id):
        logger.warning(f"Unauthorized /panic attempt from chat_id: {chat_id}")
        return
    
    await update.message.reply_text("🚨 <b>PANIC MODE ACTIVATED</b>\n\nClosing all positions...", parse_mode="HTML")
    
    try:
        risk_manager = bot_state.risk_manager
        if risk_manager is None:
            await update.message.reply_text("❌ Risk manager not initialized.")
            return
        
        # Execute emergency shutdown
        results = risk_manager.emergency_shutdown()
        
        cancelled = results.get("cancelled_orders", [])
        closed = results.get("closed_positions", [])
        remaining = results.get("remaining_positions", [])
        panic_success = results.get("panic_success", len(remaining) == 0)
        
        # Set bot to paused
        bot_state.is_paused = True
        
        # Build response
        if panic_success:
            msg = (
                "<b>🚨 PANIC SHUTDOWN COMPLETE</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"• Cancelled {len(cancelled)} orders\n"
                f"• Close attempts: {len(closed)}\n"
                "• Remaining positions: 0\n\n"
                "<b>⏸️ Bot is now PAUSED</b>\n"
                "Use /resume to restart trading."
            )
        else:
            msg = (
                "<b>⚠️ PANIC PARTIAL FAILURE</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"• Cancelled {len(cancelled)} orders\n"
                f"• Close attempts: {len(closed)}\n"
                f"• Remaining positions: {len(remaining)}\n\n"
                "<b>⏸️ Bot is now PAUSED</b>\n"
                "Check exchange/account mapping and logs before /resume."
            )
        
        # Add details
        if closed:
            msg += "\n\n<b>Closed Positions:</b>\n"
            for result in closed:
                status = "✓" if result.success else "✗"
                msg += f"  {status} {result.message}\n"
        if remaining:
            msg += "\n<b>Still Open on Hyperliquid:</b>\n"
            for pos in remaining:
                msg += f"  • {pos.side.upper()} {pos.size} {pos.symbol}\n"
        
        await update.message.reply_text(msg, parse_mode="HTML")
        
        logger.warning("PANIC SHUTDOWN executed via Telegram")
        
        # Also send notification via notifier
        if bot_state.notifier and bot_state.notifier.enabled:
            bot_state.notifier._send_message(
                "🚨 <b>PANIC SHUTDOWN</b>\n\n"
                f"Cancelled {len(cancelled)} orders\n"
                f"Close attempts: {len(closed)}\n"
                f"Remaining positions: {len(remaining)}\n"
                "Bot is now PAUSED"
            )
        
    except Exception as e:
        logger.error(f"Error in /panic command: {e}")
        await update.message.reply_text(f"❌ Error during panic shutdown: {e}")


async def cmd_help(update, context) -> None:
    """Handle /help or /start command."""
    from telegram import Update
    
    chat_id = update.effective_chat.id
    authorized_id = context.bot_data.get("authorized_chat_id", "")
    
    if not is_authorized(chat_id, authorized_id):
        logger.warning(f"Unauthorized access attempt from chat_id: {chat_id}")
        return
    
    msg = (
        "<b>🤖 Hyperliquid Trading Bot</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "<b>Available Commands:</b>\n\n"
        "/status - Account balance & positions\n"
        "/config - Current configuration\n"
        "/pause - Pause new entries\n"
        "/resume - Resume trading\n"
        "/panic - ⚠️ Close all positions NOW\n"
        "/help - Show this help message\n"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


class TelegramBotController:
    """Controller for the interactive Telegram bot.
    
    Runs the bot in a background thread so it doesn't block the main trading loop.
    """
    
    def __init__(
        self,
        bot_token: str,
        authorized_chat_id: str,
        client=None,
        risk_manager=None,
        settings=None,
        notifier=None,
    ):
        """Initialize the Telegram bot controller.
        
        Args:
            bot_token: Telegram Bot API token.
            authorized_chat_id: Only respond to this chat ID.
            client: HyperliquidClient instance.
            risk_manager: RiskManager instance.
            settings: Settings instance.
            notifier: TelegramNotifier instance.
        """
        self.bot_token = bot_token
        self.authorized_chat_id = authorized_chat_id
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._application = None
        self._running = False
        
        # Set up shared state
        bot_state.client = client
        bot_state.risk_manager = risk_manager
        bot_state.settings = settings
        bot_state.notifier = notifier
    
    def update_state(
        self,
        client=None,
        risk_manager=None,
        settings=None,
        notifier=None,
        last_algo_update=None,
        last_market_state=None,
    ):
        """Update the shared state (call after initialization)."""
        if client is not None:
            bot_state.client = client
        if risk_manager is not None:
            bot_state.risk_manager = risk_manager
        if settings is not None:
            bot_state.settings = settings
        if notifier is not None:
            bot_state.notifier = notifier
        if last_algo_update is not None:
            bot_state.last_algo_update = last_algo_update
        if last_market_state is not None:
            bot_state.last_market_state = last_market_state
    
    @property
    def is_paused(self) -> bool:
        """Check if bot is paused (for use in main loop)."""
        return bot_state.is_paused
    
    @is_paused.setter
    def is_paused(self, value: bool):
        """Set paused state."""
        bot_state.is_paused = value
    
    def _run_bot(self):
        """Run the Telegram bot in a separate thread."""
        from telegram.ext import Application, CommandHandler
        
        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            # Build application
            self._application = (
                Application.builder()
                .token(self.bot_token)
                .build()
            )
            
            # Store authorized chat ID in bot_data for handlers
            self._application.bot_data["authorized_chat_id"] = self.authorized_chat_id
            
            # Add command handlers
            self._application.add_handler(CommandHandler("start", cmd_help))
            self._application.add_handler(CommandHandler("help", cmd_help))
            self._application.add_handler(CommandHandler("status", cmd_status))
            self._application.add_handler(CommandHandler("config", cmd_config))
            self._application.add_handler(CommandHandler("pause", cmd_pause))
            self._application.add_handler(CommandHandler("resume", cmd_resume))
            self._application.add_handler(CommandHandler("panic", cmd_panic))
            
            # Run the bot
            self._running = True
            logger.info("Telegram bot controller started (polling)")
            
            self._loop.run_until_complete(
                self._application.run_polling(
                    allowed_updates=["message"],
                    drop_pending_updates=True,
                    stop_signals=None,
                )
            )
            
        except Exception as e:
            logger.error(f"Telegram bot error: {e}")
        finally:
            self._running = False
            if self._loop:
                self._loop.close()
    
    def start(self):
        """Start the Telegram bot in a background thread."""
        if not self.bot_token or self.bot_token == "your_bot_token_here":
            logger.warning("Telegram bot token not configured - interactive bot disabled")
            return False
        
        if not self.authorized_chat_id or self.authorized_chat_id == "your_chat_id_here":
            logger.warning("Telegram chat ID not configured - interactive bot disabled")
            return False
        
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Telegram bot already running")
            return True
        
        self._thread = threading.Thread(
            target=self._run_bot,
            name="TelegramBotThread",
            daemon=True,
        )
        self._thread.start()
        
        # Give it a moment to start
        import time
        time.sleep(1)
        
        return self._running
    
    def stop(self):
        """Stop the Telegram bot."""
        if self._application and self._running:
            try:
                if self._loop and self._loop.is_running():
                    # Schedule stop in the bot's event loop
                    asyncio.run_coroutine_threadsafe(
                        self._application.stop(),
                        self._loop
                    )
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {e}")
        
        self._running = False
