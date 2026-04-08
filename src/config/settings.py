"""Settings loader for environment variables and config.json."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class TradingConfig:
    symbols: list[str]
    margin_type: str
    max_leverage: int = 10


@dataclass
class StrategyConfig:
    daily_ma_type: str
    daily_ma_period: int
    intraday_timeframe: str
    rsi_period: int
    rsi_oversold: int
    rsi_overbought: int
    vwap_enabled: bool


@dataclass
class RiskConfig:
    risk_percent_per_trade: float
    atr_period: int
    atr_sl_multiplier: float
    atr_trailing_multiplier: float
    use_limit_orders: bool = False
    limit_order_timeout: int = 60


@dataclass
class FiltersConfig:
    funding_filter_enabled: bool = False
    funding_threshold: float = 0.01
    volatility_filter_enabled: bool = False
    volatility_atr_period: int = 14
    volatility_lookback: int = 20
    volatility_threshold: float = 0.5


@dataclass
class BotConfig:
    loop_interval_seconds: int
    log_level: str


@dataclass
class NotificationsConfig:
    """Telegram notification configuration."""
    enable_telegram_alerts: bool = False


@dataclass
class Settings:
    # Environment variables
    private_key: str
    wallet_address: str
    is_testnet: bool
    
    # Telegram credentials (from .env)
    telegram_bot_token: str
    telegram_chat_id: str

    # Config file sections
    trading: TradingConfig
    strategy: StrategyConfig
    risk: RiskConfig
    filters: FiltersConfig
    bot: BotConfig
    notifications: NotificationsConfig


def load_settings(config_path: str = "config.json") -> Settings:
    """Load settings from .env and config.json files."""
    # Load environment variables
    load_dotenv()

    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
    is_testnet = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"
    
    # Telegram credentials (optional)
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not private_key or private_key == "your_private_key_here":
        raise ValueError("HYPERLIQUID_PRIVATE_KEY not set in .env file")
    if not wallet_address or wallet_address == "0xYourWalletAddressHere":
        raise ValueError("HYPERLIQUID_WALLET_ADDRESS not set in .env file")

    # Load config.json
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = json.load(f)

    # Handle optional filters section (backwards compatibility)
    filters_config = config.get("filters", {})
    
    # Handle optional notifications section (backwards compatibility)
    notifications_config = config.get("notifications", {})

    return Settings(
        private_key=private_key,
        wallet_address=wallet_address,
        is_testnet=is_testnet,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        trading=TradingConfig(**config["trading"]),
        strategy=StrategyConfig(**config["strategy"]),
        risk=RiskConfig(**config["risk_management"]),
        filters=FiltersConfig(**filters_config),
        bot=BotConfig(**config["bot"]),
        notifications=NotificationsConfig(**notifications_config),
    )
