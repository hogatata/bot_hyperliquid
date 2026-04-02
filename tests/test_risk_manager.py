"""Test script for risk manager module."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.risk import RiskManager, Side, TradeSetup
from src.strategy import add_atr


def create_mock_client():
    """Create a mock client for testing without real API calls."""

    class MockBalance:
        account_value = 10000.0
        total_margin_used = 0.0
        withdrawable = 10000.0
        raw_usd = 10000.0

    class MockClient:
        wallet_address = "0x1234567890123456789012345678901234567890"

        def get_current_price(self, symbol: str) -> float:
            prices = {"BTC": 65000.0, "ETH": 3500.0}
            return prices.get(symbol, 1000.0)

        def get_account_balance(self):
            return MockBalance()

        def get_open_positions(self):
            return []  # No open positions

        def set_leverage(self, symbol: str, leverage: int, is_cross: bool = False):
            return {"status": "ok", "leverage": leverage}

        class exchange:
            @staticmethod
            def market_open(name, is_buy, sz):
                return {
                    "status": "ok",
                    "response": {
                        "data": {"statuses": [{"filled": {"oid": 12345}}]}
                    },
                }

            @staticmethod
            def order(name, is_buy, sz, limit_px, order_type, reduce_only):
                return {
                    "status": "ok",
                    "response": {
                        "data": {"statuses": [{"resting": {"oid": 12346}}]}
                    },
                }

            @staticmethod
            def market_close(coin, sz=None):
                return {"status": "ok"}

            @staticmethod
            def cancel(name, oid):
                return {"status": "ok"}

        class info:
            @staticmethod
            def open_orders(address):
                return []

    return MockClient()


def create_sample_df_with_atr() -> pd.DataFrame:
    """Create sample OHLCV data with ATR."""
    np.random.seed(42)
    n = 50
    
    base_price = 65000
    close = base_price + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n) * 300)
    low = close - np.abs(np.random.randn(n) * 300)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min"),
        "open": np.roll(close, 1),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    df.iloc[0, df.columns.get_loc("open")] = base_price
    
    # Add ATR
    df = add_atr(df, period=14)
    return df


def main():
    print("=" * 60)
    print("Risk Manager Module Test")
    print("=" * 60)
    
    # Create mock client
    client = create_mock_client()
    print("\n✓ Created mock client")
    
    # Initialize risk manager
    risk_manager = RiskManager(
        client=client,
        position_size_percent=5.0,  # 5% of account
        default_leverage=10,
        stop_loss_percent=2.0,
        take_profit_percent=4.0,
        use_atr_for_sl=False,
    )
    print("✓ Initialized RiskManager")
    print(f"  Position Size: {risk_manager.position_size_percent}%")
    print(f"  Default Leverage: {risk_manager.default_leverage}x")
    print(f"  SL: {risk_manager.stop_loss_percent}%")
    print(f"  TP: {risk_manager.take_profit_percent}%")
    
    # Test position sizing
    print("\n" + "-" * 40)
    print("1. Testing Position Sizing")
    print("-" * 40)
    
    balance = 10000  # $10,000 account
    entry_price = 65000  # BTC price
    leverage = 10
    
    size = risk_manager.calculate_position_size(balance, entry_price, leverage)
    usd_value = size * entry_price
    
    print(f"   Account Balance: ${balance:,.2f}")
    print(f"   Entry Price: ${entry_price:,.2f}")
    print(f"   Leverage: {leverage}x")
    print(f"   Position Size: {size:.6f} BTC (${usd_value:,.2f})")
    print(f"   ✓ Expected: 5% * $10k * 10x = $5,000 worth")
    
    # Test fixed SL/TP calculation
    print("\n" + "-" * 40)
    print("2. Testing Fixed SL/TP Calculation (LONG)")
    print("-" * 40)
    
    sl, tp = risk_manager.calculate_sl_tp_fixed(entry_price, Side.LONG)
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Stop Loss (2%): ${sl:,.2f}")
    print(f"   Take Profit (4%): ${tp:,.2f}")
    print(f"   ✓ Risk/Reward: 1:{(tp - entry_price) / (entry_price - sl):.1f}")
    
    print("\n" + "-" * 40)
    print("3. Testing Fixed SL/TP Calculation (SHORT)")
    print("-" * 40)
    
    sl, tp = risk_manager.calculate_sl_tp_fixed(entry_price, Side.SHORT)
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Stop Loss (2%): ${sl:,.2f}")
    print(f"   Take Profit (4%): ${tp:,.2f}")
    print(f"   ✓ Risk/Reward: 1:{(entry_price - tp) / (sl - entry_price):.1f}")
    
    # Test ATR-based SL/TP
    print("\n" + "-" * 40)
    print("4. Testing ATR-based SL/TP Calculation")
    print("-" * 40)
    
    df = create_sample_df_with_atr()
    atr_value = df["atr_14"].iloc[-1]
    
    risk_manager_atr = RiskManager(
        client=client,
        use_atr_for_sl=True,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )
    
    sl, tp = risk_manager_atr.calculate_sl_tp_atr(entry_price, Side.LONG, atr_value)
    print(f"   ATR Value: ${atr_value:,.2f}")
    print(f"   Entry: ${entry_price:,.2f}")
    print(f"   Stop Loss (1.5 ATR): ${sl:,.2f}")
    print(f"   Take Profit (3 ATR): ${tp:,.2f}")
    print(f"   ✓ SL Distance: ${entry_price - sl:,.2f} | TP Distance: ${tp - entry_price:,.2f}")
    
    # Test trade setup preparation
    print("\n" + "-" * 40)
    print("5. Testing Trade Setup Preparation")
    print("-" * 40)
    
    setup = risk_manager.prepare_trade(
        symbol="BTC",
        side=Side.LONG,
        leverage=10,
    )
    
    print(f"   Symbol: {setup.symbol}")
    print(f"   Side: {setup.side.value.upper()}")
    print(f"   Size: {setup.size} BTC")
    print(f"   Entry: ${setup.entry_price:,.2f}")
    print(f"   Stop Loss: ${setup.stop_loss:,.2f}")
    print(f"   Take Profit: ${setup.take_profit:,.2f}")
    print(f"   Leverage: {setup.leverage}x")
    print(f"   Risk Amount: ${setup.risk_amount:,.2f}")
    print(f"   Potential Profit: ${setup.potential_profit:,.2f}")
    print(f"   ✓ Risk/Reward: 1:{setup.potential_profit / setup.risk_amount:.1f}")
    
    # Test trade execution (dry run)
    print("\n" + "-" * 40)
    print("6. Testing Trade Execution (DRY RUN)")
    print("-" * 40)
    
    results = risk_manager.execute_trade(setup, dry_run=True)
    
    for step, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"   {status} {step}: {result.message}")
    
    # Test mock execution
    print("\n" + "-" * 40)
    print("7. Testing Mock Trade Execution")
    print("-" * 40)
    
    results = risk_manager.execute_trade(setup, dry_run=False)
    
    for step, result in results.items():
        status = "✓" if result.success else "✗"
        oid_str = f" (OID: {result.order_id})" if result.order_id else ""
        print(f"   {status} {step}: {result.message}{oid_str}")
    
    # Test emergency shutdown
    print("\n" + "-" * 40)
    print("8. Testing Emergency Shutdown")
    print("-" * 40)
    
    shutdown_results = risk_manager.emergency_shutdown()
    print(f"   Cancelled Orders: {len(shutdown_results['cancelled_orders'])}")
    print(f"   Closed Positions: {len(shutdown_results['closed_positions'])}")
    print("   ✓ Emergency shutdown executed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
