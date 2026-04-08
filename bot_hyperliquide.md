# Hyperliquid Daily Trading Bot Specifications

## 1. General Objective
Create an algorithmic trading bot running locally, interacting with the decentralized Hyperliquid platform. The bot must take intraday Long or Short positions based on daily trend analysis and technical indicators, with strict risk management.

## 2. Recommended Tech Stack
* **Language:** Python 3.13 ✅ (using `uv` package manager)
* **Key Libraries:**
    * `hyperliquid-python-sdk` - Official SDK for exchange interaction ✅
    * `pandas` and `pandas-ta` - For indicator calculations ✅
    * `python-dotenv` - To secure private keys ✅
* **Testnet Compatibility:** Monkey-patch applied to fix broken `spotMeta` endpoint on testnet ✅

## 3. Trading Strategy and Indicators
The bot analyzes two timeframes:
* **Daily Trend:** Moving Average (SMA 50 or EMA 20) to determine overall trend ✅
    * *Rule:* Price > daily MA → only Longs. Price < daily MA → only Shorts.
* **Intraday Signals (configurable: 5m, 15m, etc.):**
    * **VWAP (Volume Weighted Average Price):** Value zone reference ✅
    * **RSI (Relative Strength Index):** Overbought/oversold detection ✅
    * **ATR (Average True Range):** Volatility measurement for dynamic SL/TP ✅
    * **Entry Logic (Long Example):** Bullish trend + VWAP cross up + RSI exiting oversold (<30 → rising) ✅
* **Modular Design:** All indicators implemented as standalone functions in `src/strategy/indicators.py` ✅

## 4. Risk Management - CRITICAL
* **Margin Type:** ISOLATED MARGIN mandatory for each trade ✅
* **Position Size:** Configurable % of total capital (e.g., 5% per trade) ✅
* **Leverage:** Configurable (e.g., 5x or 10x), set via API before order placement ✅
* **Stop Loss (SL) and Take Profit (TP):**
    * **Fixed Mode:** SL/TP as fixed percentage of entry price ✅
    * **Dynamic ATR Mode:** SL = entry ± (ATR × multiplier), TP = entry ± (ATR × TP multiplier) ✅
    * Both sent as trigger orders immediately after market entry ✅
* **Implementation:** `src/risk/manager.py` with `RiskManager` class ✅

## 5. Infrastructure, Interface, and Control
* **Execution:** Script runs locally via `uv run python main.py` ✅
* **Configuration:**
    * `.env` file for secrets (private key, wallet address, testnet flag) ✅
    * `config.json` for trading parameters (symbols, leverage, indicators, SL/TP) ✅
* **Terminal Dashboard:** Live display with current price, trend, RSI, VWAP, balance, position status ✅
* **Kill Switch (Ctrl+C):** Signal handler that IMMEDIATELY: ✅
    1. Cancels all pending orders
    2. Closes all open positions at market price
    3. Shuts down gracefully
* **Implementation:** `main.py` with SIGINT/SIGTERM handlers and `emergency_shutdown()` ✅

## 5.1 Backtesting & Parameter Optimization ✅ (NEW)
* **Location:** `backtest/` folder
* **Components:**
    * `backtester.py` - Simulates strategy on historical OHLCV data with realistic fees (0.05% taker)
    * `optimizer.py` - Grid search over 8,100+ parameter combinations
    * `run_optimization.py` - Fetches real data from Hyperliquid and runs optimization
* **Parameters Optimized:**
    * MA Type (SMA/EMA) and Period (20/50/100)
    * RSI Period (7/14/21), Oversold (25/30/35), Overbought (65/70/75)
    * Stop Loss % (1.0-3.0), Take Profit % (2.0-6.0)
    * VWAP filter (enabled/disabled)
* **Metrics Calculated:** Total PnL, Win Rate, Profit Factor, Max Drawdown, Sharpe Ratio
* **Output:** `config_optimized.json` ready to use with best parameters

## 6. Future Enhancements

### 6.1 WebSocket Migration (Real-Time Data)
**Impact:** High | **Complexity:** Medium

Currently the bot polls the REST API every X seconds, introducing latency between price movements and signal detection. Migrating to WebSockets provides:
- **Instant price updates** - React to market moves in milliseconds, not seconds
- **Reduced API load** - Single persistent connection vs. repeated HTTP requests
- **Faster signal detection** - Critical for volatile markets and tight SL/TP levels
- **Order book streaming** - Enable real-time depth analysis

**Implementation:** Use `hyperliquid-python-sdk` WebSocket subscriptions for `allMids`, `trades`, and `l2Book`.

---

### 6.2 Trailing Stop Loss
**Impact:** High | **Complexity:** Low

Replace fixed SL with a trailing mechanism that follows price movement:
- **Lock in profits** - As price moves in your favor, SL adjusts upward (long) or downward (short)
- **Let winners run** - Capture extended trends instead of exiting at fixed TP
- **ATR-based trail** - Trail distance = X × ATR, adapting to current volatility

**Example:** Long entry at $65,000, initial SL at $64,000. Price rises to $67,000 → SL trails to $66,000. Price reverses → exit at $66,000 with $1,000 profit instead of original $1,000 loss.

---

### 6.3 Partial Take Profits (Scaling Out)
**Impact:** High | **Complexity:** Medium

Instead of all-or-nothing exits, scale out of positions:
- **TP1 (50% of position):** Close at 2×ATR profit, lock in guaranteed gains
- **TP2 (remaining 50%):** Let it run with trailing stop to capture extended moves
- **Risk-free trade:** After TP1, move SL to breakeven

**Benefit:** Balances the need to secure profits with the opportunity to capture larger trends. Reduces regret from early exits.

---

### 6.4 Limit Orders for Entry (Maker Rebates)
**Impact:** Medium | **Complexity:** Medium

Hyperliquid fee structure:
- **Taker fee:** ~0.035% (market orders)
- **Maker rebate:** ~0.01% (limit orders that add liquidity)

**Strategy:**
- Place limit buy orders slightly below VWAP for long entries
- Place limit sell orders slightly above VWAP for short entries
- Set GTT (Good Till Time) expiry to avoid stale orders
- Fall back to market order if limit doesn't fill within X seconds

**Savings:** On a $10,000 position, switching from taker to maker saves ~$4.50 per trade. Over 100 trades/month = $450 saved.

---

### 6.5 Macro/Regime Filters
**Impact:** High | **Complexity:** Medium

Add pre-trade filters to avoid unfavorable market conditions:

#### A. Funding Rate Filter
- **Logic:** Extreme positive funding (>0.05%) = crowded long, avoid new longs
- **Logic:** Extreme negative funding (<-0.05%) = crowded short, avoid new shorts
- **Benefit:** Avoid entering trades where everyone is already positioned (crowded trades often reverse)

#### B. Order Book Imbalance
- **Logic:** Check bid/ask depth before entry
- **Long confirmation:** Bid volume > Ask volume (buyers stronger)
- **Short confirmation:** Ask volume > Bid volume (sellers stronger)
- **Benefit:** Additional confirmation signal, reduces false entries

#### C. Volatility Filter (ATR Percentile)
- **Logic:** Compare current ATR to its 20-period average
- **Skip trades when:** ATR < 50% of average (low volatility, choppy market)
- **Take trades when:** ATR is normal or elevated (trending market)
- **Benefit:** Avoid whipsaws during consolidation periods

---

### 6.6 Implementation Priority Matrix

| Enhancement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| Trailing Stop Loss | High | Low | 🔴 P1 |
| Partial Take Profits | High | Medium | 🔴 P1 |
| Funding Rate Filter | High | Low | 🔴 P1 |
| WebSocket Migration | High | Medium | 🟡 P2 |
| Limit Order Entries | Medium | Medium | 🟡 P2 |
| Order Book Imbalance | Medium | Medium | 🟢 P3 |
| Volatility Filter | Medium | Low | 🟢 P3 |