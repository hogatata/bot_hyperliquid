"""Settings loader for environment variables and config.json."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class TradingConfig:
    symbols: list[str]
    leverage: int
    margin_type: str
    position_size_percent: float


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
    stop_loss_percent: float
    take_profit_percent: float
    use_atr_for_sl: bool
    atr_period: int
    atr_multiplier: float


@dataclass
class BotConfig:
    loop_interval_seconds: int
    log_level: str


@dataclass
class Settings:
    # Environment variables
    private_key: str
    wallet_address: str
    is_testnet: bool

    # Config file sections
    trading: TradingConfig
    strategy: StrategyConfig
    risk: RiskConfig
    bot: BotConfig


def load_settings(config_path: str = "config.json") -> Settings:
    """Load settings from .env and config.json files."""
    # Load environment variables
    load_dotenv()

    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
    wallet_address = os.getenv("HYPERLIQUID_WALLET_ADDRESS", "")
    is_testnet = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"

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

    return Settings(
        private_key=private_key,
        wallet_address=wallet_address,
        is_testnet=is_testnet,
        trading=TradingConfig(**config["trading"]),
        strategy=StrategyConfig(**config["strategy"]),
        risk=RiskConfig(**config["risk_management"]),
        bot=BotConfig(**config["bot"]),
    )
