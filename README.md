# 🤖 Hyperliquid Algorithmic Trading Bot

A Python-based algorithmic trading bot for the [Hyperliquid](https://hyperliquid.xyz) decentralized perpetuals exchange. The bot implements a multi-timeframe trend-following strategy with **ATR-based volatility targeting** and **Chandelier Exit** risk management.

## 📋 Features

- **Multi-Timeframe Analysis**: Daily trend (SMA/EMA) + Intraday signals (VWAP/RSI)
- **Automated Trading**: Fully automated entry with dynamic position sizing
- **ATR Volatility Targeting**: Position size calculated from risk % and ATR-based SL distance
- **Chandelier Exit**: Dynamic trailing stop that follows price using ATR multiplier
- **Kill Switch**: Emergency shutdown via Ctrl+C (cancels all orders, closes positions)
- **Live Dashboard**: Real-time terminal display of market data and positions

### Advanced Features

- **Limit Orders for Entry**: Earn maker rebates by placing limit orders at best bid/ask
- **Macro Filters**: Block trades based on funding rate or low volatility conditions
- **State Recovery**: Automatically recovers position tracking after bot restarts
- **Telegram Alerts**: Real-time notifications for trades and bot status
- **Interactive Telegram Bot**: Remote control via Telegram commands

## 📱 Telegram Bot Commands

The bot includes an interactive Telegram interface for remote monitoring and control:

| Command | Description |
|---------|-------------|
| `/status` | Account balance, positions, and unrealized PnL |
| `/config` | Current configuration (strategy, ATR parameters, risk) |
| `/pause` | Pause new trade entries (keeps managing existing positions) |
| `/resume` | Resume trading |
| `/panic` | ⚠️ Emergency: Close all positions immediately |
| `/help` | Show available commands |

**Security**: The bot only responds to the `TELEGRAM_CHAT_ID` defined in your `.env` file.

## 🎯 Strategy

The bot follows a trend-continuation strategy:

| Condition | Action |
|-----------|--------|
| Price > Daily SMA + VWAP cross up + RSI exiting oversold | **LONG** |
| Price < Daily SMA + VWAP cross down + RSI exiting overbought | **SHORT** |

**Entry Logic:**
1. **Daily Trend**: Price above/below SMA determines direction (only LONG when bullish, only SHORT when bearish)
2. **VWAP Confirmation**: Price must cross VWAP in the trend direction
3. **RSI Timing**: RSI must be exiting extreme zone (< 30 for longs, > 70 for shorts)

**Exit Logic (Chandelier Exit):**
- No fixed Take Profit - positions trail with ATR-based stop
- LONG: `SL = max(prev_SL, highest_high - ATR × trail_multiplier)`
- SHORT: `SL = min(prev_SL, lowest_low + ATR × trail_multiplier)`

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Hyperliquid account with API access

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bot_hyperliquide.git
cd bot_hyperliquide

# Install dependencies
uv sync
```

### Configuration

1. **Set up your API credentials** (copy the example and fill in your details):

```bash
cp .env.example .env
nano .env
```

```env
HYPERLIQUID_PRIVATE_KEY=your_private_key_here
HYPERLIQUID_WALLET_ADDRESS=0xYourWalletAddressHere
HYPERLIQUID_TESTNET=true

# Optional: Telegram notifications
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

2. **Adjust trading parameters** in `config.json`:

```json
{
  "trading": {
    "symbols": ["BTC", "ETH"],
    "margin_type": "isolated",
    "max_leverage": 10
  },
  "strategy": {
    "daily_ma_type": "SMA",
    "daily_ma_period": 50,
    "intraday_timeframe": "15m",
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70
  },
  "risk_management": {
    "risk_percent_per_trade": 2.0,
    "atr_period": 14,
    "atr_sl_multiplier": 1.5,
    "atr_trailing_multiplier": 2.0,
    "use_limit_orders": false,
    "limit_order_timeout": 60
  },
  "notifications": {
    "enable_telegram_alerts": false
  }
}
```

### Telegram Notifications (Optional)

To receive trade alerts on your phone:

1. **Create a Telegram Bot**:
   - Message [@BotFather](https://t.me/BotFather) on Telegram
   - Send `/newbot` and follow the prompts
   - Copy the bot token

2. **Get your Chat ID**:
   - Start a chat with your new bot
   - Visit: `https://api.telegram.org/bot<YourBotToken>/getUpdates`
   - Look for `"chat":{"id":XXXXXXXX}` - that's your chat ID

3. **Configure**:
   - Add credentials to `.env`
   - Set `"enable_telegram_alerts": true` in `config.json`

**Notification types:**
- 🤖 Bot startup/shutdown
- 🟢/🔴 Trade opened (LONG/SHORT) with ATR-based SL
- 💰 Trade closed via Chandelier Exit with PnL
- 📈 Trailing stop updates
- ⚠️ Error alerts

### Running the Bot

```bash
uv run python main.py
```

## ⚠️ Emergency Shutdown (Kill Switch)

Press **`Ctrl+C`** at any time to trigger an emergency shutdown:

1. ❌ Cancels all pending orders
2. 📉 Closes all open positions at market price
3. 🛑 Exits gracefully

## 🔬 Parameter Optimization (Backtesting)

The `backtest/` folder contains a grid search optimizer to find the best ATR parameters:

```bash
uv run python backtest/run_optimization.py
```

This will:
1. Fetch last 30 days of BTC 15m candles from Hyperliquid
2. Test ATR multiplier combinations (SL: 1.0-2.5, Trail: 1.5-3.0)
3. Print a detailed performance report
4. **Directly update `config.json`** with optimal parameters
5. Send Telegram notification with results (if configured)

**Metrics available:**
- `total_pnl` - Total profit/loss (default)
- `sharpe_ratio` - Risk-adjusted returns
- `profit_factor` - Gross profit / gross loss
- `win_rate` - Percentage of winning trades

## 📁 Project Structure

```
bot_hyperliquide/
├── .env                    # API credentials (NEVER commit!)
├── .env.example            # Template for .env
├── config.json             # Trading parameters
├── pyproject.toml          # Project dependencies
│
├── main.py                 # Entry point (run this)
│
├── backtest/               # Parameter optimization
│   ├── backtester.py       # ATR volatility targeting simulation
│   ├── optimizer.py        # Grid search optimizer
│   └── run_optimization.py # Run optimization script
│
├── src/
│   ├── config/
│   │   └── settings.py     # Configuration loader
│   │
│   ├── exchange/
│   │   └── client.py       # Hyperliquid SDK wrapper
│   │
│   ├── strategy/
│   │   ├── indicators.py   # Technical indicators (SMA, RSI, VWAP, ATR)
│   │   └── signals.py      # Entry signal logic
│   │
│   ├── risk/
│   │   └── manager.py      # ATR volatility targeting, Chandelier Exit
│   │
│   └── utils/
│       ├── logger.py       # Terminal logging
│       ├── notifier.py     # Telegram notifications
│       └── telegram_bot.py # Interactive Telegram commands
│
└── tests/                  # Test scripts
```

## ⚙️ Configuration Options

### Trading Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `symbols` | Trading pairs | `["BTC", "ETH"]` |
| `max_leverage` | Maximum allowed leverage | `10` |
| `margin_type` | Margin mode | `isolated` |

### Strategy Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `daily_ma_type` | Moving average type | `SMA` |
| `daily_ma_period` | MA period | `50` |
| `intraday_timeframe` | Candle timeframe | `15m` |
| `rsi_period` | RSI calculation period | `14` |
| `rsi_oversold` | Oversold threshold | `30` |
| `rsi_overbought` | Overbought threshold | `70` |

### Risk Management (ATR Dynamic)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `risk_percent_per_trade` | % of account to risk per trade | `2.0` |
| `atr_period` | ATR calculation period | `14` |
| `atr_sl_multiplier` | ATR multiplier for initial SL | `1.5` |
| `atr_trailing_multiplier` | ATR multiplier for Chandelier Exit | `2.0` |
| `use_limit_orders` | Use limit orders for entry | `false` |
| `limit_order_timeout` | Seconds to wait for limit fill | `60` |

### Macro Filters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `funding_filter_enabled` | Block trades when funding is extreme | `false` |
| `funding_threshold` | Funding rate threshold (%) | `0.01` |
| `volatility_filter_enabled` | Block trades in low volatility | `false` |
| `volatility_threshold` | ATR ratio threshold | `0.5` |

## 🧪 Running Tests

```bash
# Test the Hyperliquid client (requires valid .env)
uv run python tests/test_client.py

# Test indicators (no API required)
uv run python tests/test_indicators.py

# Test risk manager (no API required)
uv run python tests/test_risk_manager.py

# Test signal generation (no API required)
uv run python tests/test_signals.py
```

## 🔧 Extending the Bot

### Adding New Indicators

Edit `src/strategy/indicators.py`:

```python
def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    macd = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    df = pd.concat([df, macd], axis=1)
    return df
```

### Modifying Entry Logic

Edit `src/strategy/signals.py` in the `analyze()` method to customize entry conditions.

## ⚠️ Disclaimer

**USE AT YOUR OWN RISK.** This bot is provided for educational purposes only.

- Trading cryptocurrencies involves substantial risk of loss
- Past performance does not guarantee future results
- Never trade with money you cannot afford to lose
- Always test on **testnet** before using real funds
- The authors are not responsible for any financial losses

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.