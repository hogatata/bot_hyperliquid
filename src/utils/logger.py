"""Logger utility for terminal display."""

import logging
import sys
from datetime import datetime


def setup_logger(name: str = "bot", level: str = "INFO") -> logging.Logger:
    """Configure and return a logger with formatted output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_trade_info(
    logger: logging.Logger,
    symbol: str,
    price: float,
    trend: str,
    status: str,
) -> None:
    """Log formatted trade information."""
    logger.info(f"[{symbol}] Price: ${price:,.2f} | Trend: {trend} | {status}")


def log_position(
    logger: logging.Logger,
    symbol: str,
    side: str,
    size: float,
    entry_price: float,
    pnl: float,
) -> None:
    """Log open position details."""
    pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    logger.info(
        f"[{symbol}] {side.upper()} | Size: {size} | Entry: ${entry_price:,.2f} | PnL: {pnl_str}"
    )
