"""Telegram notification system for trade alerts and bot status."""

import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""
    
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


class TelegramNotifier:
    """Send notifications via Telegram Bot API.
    
    Handles all notification types for the trading bot:
    - Bot startup/shutdown
    - Trade opened
    - Trade closed (TP/SL/Trailing)
    - Errors and warnings
    
    All methods are designed to fail gracefully - network issues
    will not crash the bot, only log warnings.
    """
    
    TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
    ):
        """Initialize the Telegram notifier.
        
        Args:
            bot_token: Telegram Bot API token (from @BotFather).
            chat_id: Chat ID to send messages to (user or group).
            enabled: Whether notifications are enabled.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._validated = False
        
        # Validate configuration on init
        if enabled:
            self._validate_config()
    
    def _validate_config(self) -> bool:
        """Validate that bot token and chat ID are configured."""
        if not self.bot_token or self.bot_token == "your_bot_token_here":
            logger.warning("Telegram bot token not configured - notifications disabled")
            self.enabled = False
            return False
        
        if not self.chat_id or self.chat_id == "your_chat_id_here":
            logger.warning("Telegram chat ID not configured - notifications disabled")
            self.enabled = False
            return False
        
        self._validated = True
        return True
    
    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram Bot API.
        
        Args:
            text: Message text (supports HTML formatting).
            parse_mode: Parse mode for formatting ("HTML" or "Markdown").
            
        Returns:
            True if message sent successfully, False otherwise.
        """
        if not self.enabled:
            return False
        
        if not self._validated:
            return False
        
        url = self.TELEGRAM_API_URL.format(token=self.bot_token)
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"Telegram notification sent successfully")
                return True
            else:
                logger.warning(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.warning("Telegram notification timed out - continuing without alert")
            return False
            
        except requests.exceptions.ConnectionError:
            logger.warning("No internet connection - Telegram notification skipped")
            return False
            
        except Exception as e:
            logger.warning(f"Failed to send Telegram notification: {e}")
            return False
    
    # =========================================================================
    # Bot Status Notifications
    # =========================================================================
    
    def notify_startup(
        self,
        is_testnet: bool,
        symbols: list[str],
        leverage: int,
        features: Optional[dict] = None,
    ) -> bool:
        """Notify that the bot has started.
        
        Args:
            is_testnet: Whether running on testnet.
            symbols: List of trading symbols.
            leverage: Configured leverage.
            features: Dict of enabled features.
        """
        mode = "TESTNET" if is_testnet else "⚠️ MAINNET"
        symbols_str = ", ".join(symbols)
        
        # Build features list
        features = features or {}
        feature_list = []
        if features.get("limit_orders"):
            feature_list.append("Limit Orders")
        if features.get("trailing_stop"):
            feature_list.append("Trailing Stop")
        if features.get("funding_filter"):
            feature_list.append("Funding Filter")
        if features.get("volatility_filter"):
            feature_list.append("Volatility Filter")
        
        features_str = ", ".join(feature_list) if feature_list else "Standard"
        
        message = f"""🤖 <b>Bot Started on {mode}</b>

📊 <b>Configuration:</b>
• Symbols: {symbols_str}
• Leverage: {leverage}x
• Features: {features_str}

Bot is now monitoring the markets."""
        
        return self._send_message(message)
    
    def notify_shutdown(self, reason: str = "Emergency shutdown triggered") -> bool:
        """Notify that the bot has shut down.
        
        Args:
            reason: Reason for shutdown.
        """
        message = f"""🛑 <b>Bot Shutdown</b>

{reason}

All orders cancelled. Positions closed."""
        
        return self._send_message(message)
    
    # =========================================================================
    # Trade Notifications
    # =========================================================================
    
    def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        leverage: int,
        order_type: str = "MARKET",
    ) -> bool:
        """Notify that a trade has been opened.
        
        Args:
            symbol: Trading symbol (e.g., "BTC").
            side: Trade side ("LONG" or "SHORT").
            entry_price: Entry price.
            size: Position size.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            leverage: Leverage used.
            order_type: Order type ("MARKET" or "LIMIT").
        """
        emoji = "🟢" if side.upper() == "LONG" else "🔴"
        
        # Calculate risk/reward
        if side.upper() == "LONG":
            risk_pct = ((entry_price - stop_loss) / entry_price) * 100
            reward_pct = ((take_profit - entry_price) / entry_price) * 100
        else:
            risk_pct = ((stop_loss - entry_price) / entry_price) * 100
            reward_pct = ((entry_price - take_profit) / entry_price) * 100
        
        message = f"""{emoji} <b>{side.upper()} {symbol} Executed</b>

📍 Entry: ${entry_price:,.2f}
📦 Size: {size:.4f} {symbol}
🎯 TP: ${take_profit:,.2f} (+{reward_pct:.1f}%)
🛡️ SL: ${stop_loss:,.2f} (-{risk_pct:.1f}%)
⚡ Leverage: {leverage}x
📝 Type: {order_type}"""
        
        return self._send_message(message)
    
    def notify_trade_closed(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        exit_reason: str = "manual",
    ) -> bool:
        """Notify that a trade has been closed.
        
        Args:
            symbol: Trading symbol.
            side: Trade side.
            entry_price: Original entry price.
            exit_price: Exit price.
            pnl: Profit/loss in USD.
            pnl_percent: Profit/loss percentage.
            exit_reason: Reason for exit ("take_profit", "stop_loss", "trailing_stop", "manual").
        """
        # Choose emoji based on outcome
        if pnl >= 0:
            emoji = "💰" if pnl > 0 else "⚖️"
        else:
            emoji = "📉"
        
        # Format reason
        reason_map = {
            "take_profit": "✅ Take Profit Hit",
            "stop_loss": "❌ Stop Loss Hit",
            "trailing_stop": "📈 Trailing Stop Hit",
            "manual": "🖐️ Manual Close",
            "emergency": "🚨 Emergency Close",
        }
        reason_str = reason_map.get(exit_reason, exit_reason.capitalize())
        
        # Format PnL
        pnl_sign = "+" if pnl >= 0 else ""
        
        message = f"""{emoji} <b>Trade Closed | {symbol} {side.upper()}</b>

{reason_str}

📍 Entry: ${entry_price:,.2f}
🏁 Exit: ${exit_price:,.2f}
💵 PnL: <b>{pnl_sign}${pnl:,.2f}</b> ({pnl_sign}{pnl_percent:.1f}%)"""
        
        return self._send_message(message)
    
    def notify_trailing_stop_updated(
        self,
        symbol: str,
        side: str,
        old_sl: float,
        new_sl: float,
        current_price: float,
    ) -> bool:
        """Notify that trailing stop has been updated.
        
        Args:
            symbol: Trading symbol.
            side: Trade side.
            old_sl: Previous stop loss price.
            new_sl: New stop loss price.
            current_price: Current market price.
        """
        if side.upper() == "LONG":
            locked_profit = ((new_sl - old_sl) / old_sl) * 100
        else:
            locked_profit = ((old_sl - new_sl) / old_sl) * 100
        
        message = f"""📈 <b>Trailing Stop Updated | {symbol}</b>

📍 Price: ${current_price:,.2f}
🛡️ SL: ${old_sl:,.2f} → ${new_sl:,.2f}
🔒 Locked: +{locked_profit:.1f}% profit"""
        
        return self._send_message(message)
    
    # =========================================================================
    # Error Notifications
    # =========================================================================
    
    def notify_error(self, error_message: str, context: str = "") -> bool:
        """Notify about an error.
        
        Args:
            error_message: The error message.
            context: Additional context about where the error occurred.
        """
        context_str = f"\n📍 Context: {context}" if context else ""
        
        message = f"""⚠️ <b>Bot Error</b>

{error_message}{context_str}

Bot is still running."""
        
        return self._send_message(message)
    
    def notify_signal_blocked(
        self,
        symbol: str,
        side: str,
        reason: str,
    ) -> bool:
        """Notify that a signal was blocked by a filter.
        
        Args:
            symbol: Trading symbol.
            side: Blocked signal side.
            reason: Reason for blocking (e.g., "high funding rate").
        """
        message = f"""🚫 <b>Signal Blocked | {symbol}</b>

Direction: {side.upper()}
Reason: {reason}

Signal skipped due to macro filter."""
        
        return self._send_message(message)
    
    # =========================================================================
    # Test Method
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Send a test message to verify configuration.
        
        Returns:
            True if test message sent successfully.
        """
        message = """✅ <b>Telegram Connection Test</b>

This is a test message from your Hyperliquid trading bot.
Notifications are working correctly!"""
        
        return self._send_message(message)


# Convenience function to create notifier from settings
def create_notifier_from_settings(settings) -> TelegramNotifier:
    """Create a TelegramNotifier from Settings object.
    
    Args:
        settings: Settings object with telegram configuration.
        
    Returns:
        Configured TelegramNotifier instance.
    """
    return TelegramNotifier(
        bot_token=getattr(settings, 'telegram_bot_token', ''),
        chat_id=getattr(settings, 'telegram_chat_id', ''),
        enabled=getattr(settings.notifications, 'enable_telegram_alerts', False) if hasattr(settings, 'notifications') else False,
    )
