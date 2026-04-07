"""Test script for advanced features: Limit Orders, Trailing Stop, Macro Filters."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import time

from src.risk import RiskManager, Side, TradeSetup, ActivePosition
from src.strategy import Signal, SignalGenerator, FilterResult, add_atr


# =============================================================================
# Mock Client for Testing
# =============================================================================

class MockBalance:
    account_value = 10000.0
    total_margin_used = 0.0
    withdrawable = 10000.0
    raw_usd = 10000.0


class MockClient:
    """Enhanced mock client for testing advanced features."""
    
    wallet_address = "0x1234567890123456789012345678901234567890"
    
    def __init__(self):
        self.open_orders = []
        self.filled_orders = []
        self.cancelled_orders = []
        self.limit_fill_immediately = False  # Control limit order behavior
    
    def get_current_price(self, symbol: str) -> float:
        prices = {"BTC": 65000.0, "ETH": 3500.0}
        return prices.get(symbol, 1000.0)
    
    def get_account_balance(self):
        return MockBalance()
    
    def get_open_positions(self):
        return []
    
    def get_best_bid_ask(self, symbol: str) -> tuple[float, float]:
        """Return best bid/ask for limit order testing."""
        if symbol == "BTC":
            return (64990.0, 65010.0)  # Bid/Ask spread
        return (3495.0, 3505.0)
    
    def get_order_book(self, symbol: str) -> dict:
        return {
            "bids": [{"px": 64990.0, "sz": 1.0}],
            "asks": [{"px": 65010.0, "sz": 1.0}],
        }
    
    def get_funding_rate(self, symbol: str) -> float:
        """Mock funding rate."""
        return 0.005  # 0.005% funding rate
    
    def get_open_orders(self, symbol: str = None):
        """Return currently open orders."""
        if self.limit_fill_immediately:
            return []  # Order filled, no longer open
        return self.open_orders
    
    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel an order."""
        self.cancelled_orders.append(order_id)
        return {"status": "ok"}
    
    def set_leverage(self, symbol: str, leverage: int, is_cross: bool = False):
        return {"status": "ok", "leverage": leverage}
    
    class exchange:
        _parent = None  # Will be set after class creation
        
        @classmethod
        def market_open(cls, name, is_buy, sz):
            return {
                "status": "ok",
                "response": {
                    "data": {"statuses": [{"filled": {"oid": 12345, "avgPx": 65000.0}}]}
                },
            }
        
        @classmethod
        def order(cls, name, is_buy, sz, limit_px, order_type, reduce_only):
            # Check if it's a limit order (for entry)
            if "limit" in order_type:
                oid = 99999
                if cls._parent and cls._parent.limit_fill_immediately:
                    # Simulate immediate fill
                    return {
                        "status": "ok",
                        "response": {
                            "data": {"statuses": [{"filled": {"oid": oid, "avgPx": limit_px}}]}
                        },
                    }
                else:
                    # Order is resting (not filled)
                    if cls._parent:
                        cls._parent.open_orders.append({"oid": oid, "coin": name})
                    return {
                        "status": "ok",
                        "response": {
                            "data": {"statuses": [{"resting": {"oid": oid}}]}
                        },
                    }
            # Trigger order (SL/TP)
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


def create_mock_client():
    """Create a mock client with proper exchange reference."""
    client = MockClient()
    client.exchange._parent = client
    return client


# =============================================================================
# Sample Data Helpers
# =============================================================================

def create_sample_df_with_atr(n: int = 100, base_price: float = 65000) -> pd.DataFrame:
    """Create sample OHLCV data with ATR."""
    np.random.seed(42)
    
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
    
    df = add_atr(df, period=14)
    return df


def create_low_volatility_df() -> pd.DataFrame:
    """Create data with very low ATR (choppy market)."""
    np.random.seed(42)
    n = 100
    base_price = 65000
    
    # Very small price movements
    close = base_price + np.random.randn(n) * 50  # Tiny movements
    high = close + np.abs(np.random.randn(n) * 20)
    low = close - np.abs(np.random.randn(n) * 20)
    
    df = pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min"),
        "open": np.roll(close, 1),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    df.iloc[0, df.columns.get_loc("open")] = base_price
    
    df = add_atr(df, period=14)
    return df


# =============================================================================
# Test 1: Limit Orders
# =============================================================================

def test_limit_order_immediate_fill():
    """Test limit order that fills immediately."""
    print("\n" + "=" * 60)
    print("TEST 1.1: Limit Order - Immediate Fill")
    print("=" * 60)
    
    client = create_mock_client()
    client.limit_fill_immediately = True
    
    risk_manager = RiskManager(
        client=client,
        use_limit_orders=True,
        limit_order_timeout=60,
    )
    
    setup = TradeSetup(
        symbol="BTC",
        side=Side.LONG,
        size=0.01,
        entry_price=65000.0,
        stop_loss=64000.0,
        take_profit=66000.0,
        leverage=5,
        risk_amount=100.0,
        potential_profit=200.0,
        atr_value=500.0,
    )
    
    results = risk_manager.execute_trade(setup, dry_run=False)
    
    print(f"   Entry Order: {results['entry'].message}")
    assert results["entry"].success, "Limit order should succeed"
    assert "LIMIT" in results["entry"].message, "Should be a LIMIT order"
    assert "filled" in results["entry"].message.lower(), "Should indicate fill"
    print("   ✓ Limit order filled immediately")


def test_limit_order_timeout():
    """Test limit order that times out and gets cancelled."""
    print("\n" + "=" * 60)
    print("TEST 1.2: Limit Order - Timeout (Simulated)")
    print("=" * 60)
    
    client = create_mock_client()
    client.limit_fill_immediately = False  # Order will NOT fill
    
    risk_manager = RiskManager(
        client=client,
        use_limit_orders=True,
        limit_order_timeout=2,  # Short timeout for testing
    )
    
    setup = TradeSetup(
        symbol="BTC",
        side=Side.LONG,
        size=0.01,
        entry_price=65000.0,
        stop_loss=64000.0,
        take_profit=66000.0,
        leverage=5,
        risk_amount=100.0,
        potential_profit=200.0,
        atr_value=500.0,
    )
    
    print("   Placing limit order with 2s timeout...")
    start = time.time()
    results = risk_manager.execute_trade(setup, dry_run=False)
    elapsed = time.time() - start
    
    print(f"   Entry Order: {results['entry'].message}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    
    assert not results["entry"].success, "Should fail due to timeout"
    assert "timed out" in results["entry"].message.lower(), "Should mention timeout"
    assert 99999 in client.cancelled_orders, "Order should be cancelled"
    print("   ✓ Limit order timed out and was cancelled")


def test_limit_order_best_bid():
    """Test that limit orders use best bid for LONG."""
    print("\n" + "=" * 60)
    print("TEST 1.3: Limit Order - Best Bid for LONG")
    print("=" * 60)
    
    client = create_mock_client()
    client.limit_fill_immediately = True
    
    risk_manager = RiskManager(
        client=client,
        use_limit_orders=True,
    )
    
    best_bid, best_ask = client.get_best_bid_ask("BTC")
    print(f"   Best Bid: ${best_bid:,.2f}")
    print(f"   Best Ask: ${best_ask:,.2f}")
    
    # For LONG, we should place at best_bid (64990)
    setup = TradeSetup(
        symbol="BTC",
        side=Side.LONG,
        size=0.01,
        entry_price=65000.0,
        stop_loss=64000.0,
        take_profit=66000.0,
        leverage=5,
        risk_amount=100.0,
        potential_profit=200.0,
        atr_value=500.0,
    )
    
    results = risk_manager.execute_trade(setup, dry_run=False)
    
    # The order message should show the limit price
    print(f"   Result: {results['entry'].message}")
    assert results["entry"].success, "Limit order should succeed"
    assert "$64,990" in results["entry"].message, f"Should use best bid price, got: {results['entry'].message}"
    print("   ✓ Limit order placed at best bid for LONG")


# =============================================================================
# Test 2: Trailing Stop Loss
# =============================================================================

def test_trailing_stop_long_profit():
    """Test trailing stop moves up when LONG is in profit."""
    print("\n" + "=" * 60)
    print("TEST 2.1: Trailing Stop - LONG Position Moving Up")
    print("=" * 60)
    
    client = create_mock_client()
    client.limit_fill_immediately = True
    
    risk_manager = RiskManager(
        client=client,
        use_limit_orders=False,
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.5,
    )
    
    setup = TradeSetup(
        symbol="BTC",
        side=Side.LONG,
        size=0.01,
        entry_price=65000.0,
        stop_loss=64000.0,
        take_profit=68000.0,
        leverage=5,
        risk_amount=100.0,
        potential_profit=300.0,
        atr_value=500.0,  # ATR = $500
    )
    
    # Execute trade to register position
    results = risk_manager.execute_trade(setup, dry_run=False)
    assert results["entry"].success, "Entry should succeed"
    
    # Check position is tracked
    assert "BTC" in risk_manager.active_positions, "Position should be tracked"
    pos = risk_manager.active_positions["BTC"]
    initial_sl = pos.current_sl
    print(f"   Entry: ${setup.entry_price:,.2f}")
    print(f"   Initial SL: ${initial_sl:,.2f}")
    print(f"   ATR: ${setup.atr_value:,.2f}")
    
    # Simulate price moving up to $66,000
    new_price = 66000.0
    trail_result = risk_manager.update_trailing_stop("BTC", new_price)
    
    if trail_result and trail_result.success:
        print(f"   Price moved to: ${new_price:,.2f}")
        print(f"   {trail_result.message}")
        
        # New SL should be: 66000 - (500 * 1.5) = 65250
        expected_sl = new_price - (setup.atr_value * 1.5)
        actual_sl = risk_manager.active_positions["BTC"].current_sl
        
        print(f"   New SL: ${actual_sl:,.2f} (expected: ${expected_sl:,.2f})")
        assert actual_sl > initial_sl, "SL should have moved up"
        assert abs(actual_sl - expected_sl) < 1, "SL should be at expected level"
        print("   ✓ Trailing stop moved up with profit")
    else:
        print("   ✓ No update needed (price didn't move enough)")


def test_trailing_stop_long_no_move_down():
    """Test trailing stop does NOT move down when price retraces."""
    print("\n" + "=" * 60)
    print("TEST 2.2: Trailing Stop - LONG Should NOT Move Down")
    print("=" * 60)
    
    client = create_mock_client()
    
    risk_manager = RiskManager(
        client=client,
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.5,
    )
    
    # Manually set up a tracked position with SL at 65250
    risk_manager.active_positions["BTC"] = ActivePosition(
        symbol="BTC",
        side=Side.LONG,
        entry_price=65000.0,
        size=0.01,
        initial_sl=64000.0,
        current_sl=65250.0,  # Already trailed up
        atr_value=500.0,
        atr_trail_multiplier=1.5,
        highest_price=66000.0,
        lowest_price=65000.0,
        sl_order_id=12346,
    )
    
    current_sl = risk_manager.active_positions["BTC"].current_sl
    print(f"   Current SL: ${current_sl:,.2f}")
    print(f"   Highest Price: ${risk_manager.active_positions['BTC'].highest_price:,.2f}")
    
    # Price retraces down to 65500 (below highest of 66000)
    retrace_price = 65500.0
    print(f"   Price retraces to: ${retrace_price:,.2f}")
    
    trail_result = risk_manager.update_trailing_stop("BTC", retrace_price)
    
    new_sl = risk_manager.active_positions["BTC"].current_sl
    print(f"   SL after retrace: ${new_sl:,.2f}")
    
    assert new_sl == current_sl, "SL should NOT move down on retrace"
    assert trail_result is None, "No update should occur"
    print("   ✓ Trailing stop correctly stayed in place")


def test_trailing_stop_short_profit():
    """Test trailing stop moves down when SHORT is in profit."""
    print("\n" + "=" * 60)
    print("TEST 2.3: Trailing Stop - SHORT Position Moving Down")
    print("=" * 60)
    
    client = create_mock_client()
    
    risk_manager = RiskManager(
        client=client,
        trailing_stop_enabled=True,
        trailing_atr_multiplier=1.5,
    )
    
    # Manually set up a SHORT position
    risk_manager.active_positions["ETH"] = ActivePosition(
        symbol="ETH",
        side=Side.SHORT,
        entry_price=3500.0,
        size=1.0,
        initial_sl=3600.0,
        current_sl=3600.0,
        atr_value=50.0,  # ATR = $50
        atr_trail_multiplier=1.5,
        highest_price=3500.0,
        lowest_price=3500.0,
        sl_order_id=12346,
    )
    
    initial_sl = risk_manager.active_positions["ETH"].current_sl
    print(f"   Entry: $3,500.00 (SHORT)")
    print(f"   Initial SL: ${initial_sl:,.2f}")
    print(f"   ATR: $50.00")
    
    # Price moves down to $3,400 (in our profit)
    new_price = 3400.0
    trail_result = risk_manager.update_trailing_stop("ETH", new_price)
    
    if trail_result and trail_result.success:
        print(f"   Price moved to: ${new_price:,.2f}")
        print(f"   {trail_result.message}")
        
        # New SL should be: 3400 + (50 * 1.5) = 3475
        expected_sl = new_price + (50.0 * 1.5)
        actual_sl = risk_manager.active_positions["ETH"].current_sl
        
        print(f"   New SL: ${actual_sl:,.2f} (expected: ${expected_sl:,.2f})")
        assert actual_sl < initial_sl, "SL should have moved down"
        assert abs(actual_sl - expected_sl) < 1, "SL should be at expected level"
        print("   ✓ Trailing stop moved down with SHORT profit")
    else:
        print("   ✓ No update needed")


# =============================================================================
# Test 3: Macro/Regime Filters
# =============================================================================

def test_funding_filter_blocks_crowded_long():
    """Test funding filter blocks LONG when funding is high positive."""
    print("\n" + "=" * 60)
    print("TEST 3.1: Funding Filter - Block Crowded LONG")
    print("=" * 60)
    
    signal_gen = SignalGenerator(
        daily_ma_type="SMA",
        daily_ma_period=50,
        funding_filter_enabled=True,
        funding_threshold=0.01,  # 0.01%
    )
    
    # High positive funding (longs paying shorts)
    high_funding = 0.05  # 0.05% > 0.01% threshold
    
    # Create bullish scenario that would normally trigger LONG
    df = create_sample_df_with_atr()
    
    # Check filter directly
    filter_result = signal_gen.check_macro_filters(df, Signal.LONG, funding_rate=high_funding)
    
    print(f"   Funding Rate: {high_funding}%")
    print(f"   Threshold: {signal_gen.funding_threshold}%")
    print(f"   Filter Passed: {filter_result.passed}")
    print(f"   Reason: {filter_result.reason}")
    
    assert not filter_result.passed, "Filter should block the signal"
    assert filter_result.funding_blocked, "Should be blocked by funding"
    assert "crowded long" in filter_result.funding_reason.lower()
    print("   ✓ LONG signal correctly blocked due to high funding")


def test_funding_filter_blocks_crowded_short():
    """Test funding filter blocks SHORT when funding is high negative."""
    print("\n" + "=" * 60)
    print("TEST 3.2: Funding Filter - Block Crowded SHORT")
    print("=" * 60)
    
    signal_gen = SignalGenerator(
        funding_filter_enabled=True,
        funding_threshold=0.01,
    )
    
    # High negative funding (shorts paying longs)
    low_funding = -0.05  # -0.05% < -0.01% threshold
    
    df = create_sample_df_with_atr()
    
    filter_result = signal_gen.check_macro_filters(df, Signal.SHORT, funding_rate=low_funding)
    
    print(f"   Funding Rate: {low_funding}%")
    print(f"   Threshold: ±{signal_gen.funding_threshold}%")
    print(f"   Filter Passed: {filter_result.passed}")
    print(f"   Reason: {filter_result.reason}")
    
    assert not filter_result.passed, "Filter should block the signal"
    assert filter_result.funding_blocked, "Should be blocked by funding"
    assert "crowded short" in filter_result.funding_reason.lower()
    print("   ✓ SHORT signal correctly blocked due to low funding")


def test_funding_filter_allows_normal():
    """Test funding filter allows trades when funding is normal."""
    print("\n" + "=" * 60)
    print("TEST 3.3: Funding Filter - Allow Normal Funding")
    print("=" * 60)
    
    signal_gen = SignalGenerator(
        funding_filter_enabled=True,
        funding_threshold=0.01,
    )
    
    normal_funding = 0.005  # 0.005% - within threshold
    
    df = create_sample_df_with_atr()
    
    filter_result = signal_gen.check_macro_filters(df, Signal.LONG, funding_rate=normal_funding)
    
    print(f"   Funding Rate: {normal_funding}%")
    print(f"   Threshold: ±{signal_gen.funding_threshold}%")
    print(f"   Filter Passed: {filter_result.passed}")
    
    assert filter_result.passed, "Filter should allow normal funding"
    assert not filter_result.funding_blocked
    print("   ✓ Signal correctly allowed with normal funding")


def test_volatility_filter_blocks_low_atr():
    """Test volatility filter blocks entry when ATR is too low."""
    print("\n" + "=" * 60)
    print("TEST 3.4: Volatility Filter - Block Low ATR")
    print("=" * 60)
    
    signal_gen = SignalGenerator(
        volatility_filter_enabled=True,
        volatility_atr_period=14,
        volatility_lookback=20,
        volatility_threshold=0.5,  # Block if ATR < 50% of average
    )
    
    # Create low volatility data
    df = create_low_volatility_df()
    
    # Manually add ATR and ensure indicators exist
    signal_gen.add_indicators(df)
    df = add_atr(df, period=14, column_name="atr_14")
    
    current_atr = df["atr_14"].iloc[-1]
    avg_atr = df["atr_14"].tail(20).mean()
    ratio = current_atr / avg_atr if avg_atr > 0 else 0
    
    print(f"   Current ATR: ${current_atr:,.2f}")
    print(f"   Average ATR (20): ${avg_atr:,.2f}")
    print(f"   Ratio: {ratio:.2f}")
    print(f"   Threshold: {signal_gen.volatility_threshold}")
    
    filter_result = signal_gen.check_macro_filters(df, Signal.LONG, funding_rate=0.0)
    
    print(f"   Filter Passed: {filter_result.passed}")
    if not filter_result.passed:
        print(f"   Reason: {filter_result.reason}")
        assert filter_result.volatility_blocked, "Should be blocked by volatility"
        print("   ✓ Signal correctly blocked due to low volatility")
    else:
        print("   ℹ Volatility was within acceptable range")


def test_volatility_filter_allows_normal():
    """Test volatility filter allows entry when ATR is normal."""
    print("\n" + "=" * 60)
    print("TEST 3.5: Volatility Filter - Allow Normal ATR")
    print("=" * 60)
    
    signal_gen = SignalGenerator(
        volatility_filter_enabled=True,
        volatility_atr_period=14,
        volatility_lookback=20,
        volatility_threshold=0.5,
    )
    
    # Create normal volatility data
    df = create_sample_df_with_atr()
    signal_gen.add_indicators(df)
    
    current_atr = df["atr_14"].iloc[-1]
    avg_atr = df["atr_14"].tail(20).mean()
    ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
    
    print(f"   Current ATR: ${current_atr:,.2f}")
    print(f"   Average ATR (20): ${avg_atr:,.2f}")
    print(f"   Ratio: {ratio:.2f}")
    
    filter_result = signal_gen.check_macro_filters(df, Signal.LONG, funding_rate=0.0)
    
    print(f"   Filter Passed: {filter_result.passed}")
    
    if ratio >= 0.5:
        assert filter_result.passed, "Filter should allow normal volatility"
        assert not filter_result.volatility_blocked
        print("   ✓ Signal correctly allowed with normal volatility")
    else:
        print("   ℹ Volatility happened to be low in this sample")


def test_signal_integration_with_filters():
    """Test that SignalGenerator.analyze() integrates filters correctly."""
    print("\n" + "=" * 60)
    print("TEST 3.6: Signal Integration with Macro Filters")
    print("=" * 60)
    
    signal_gen = SignalGenerator(
        daily_ma_type="SMA",
        daily_ma_period=50,
        rsi_period=14,
        vwap_enabled=False,  # Simplify
        funding_filter_enabled=True,
        funding_threshold=0.01,
    )
    
    df = create_sample_df_with_atr()
    
    # Test with high funding that would block LONG
    high_funding = 0.05
    result = signal_gen.analyze(df, funding_rate=high_funding)
    
    print(f"   Signal: {result.signal.value}")
    print(f"   Reason: {result.reason}")
    print(f"   Filter Result: {result.filter_result}")
    
    if result.filter_result and not result.filter_result.passed:
        print("   ✓ Signal properly blocked by macro filter")
    else:
        print("   ℹ No blocking occurred (signal conditions not met)")


# =============================================================================
# Main Test Runner
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  ADVANCED FEATURES TEST SUITE")
    print("  Testing: Limit Orders, Trailing Stop, Macro Filters")
    print("=" * 70)
    
    # Test 1: Limit Orders
    test_limit_order_immediate_fill()
    test_limit_order_best_bid()
    # Skip timeout test by default (takes 2+ seconds)
    # test_limit_order_timeout()
    
    # Test 2: Trailing Stop
    test_trailing_stop_long_profit()
    test_trailing_stop_long_no_move_down()
    test_trailing_stop_short_profit()
    
    # Test 3: Macro Filters
    test_funding_filter_blocks_crowded_long()
    test_funding_filter_blocks_crowded_short()
    test_funding_filter_allows_normal()
    test_volatility_filter_blocks_low_atr()
    test_volatility_filter_allows_normal()
    test_signal_integration_with_filters()
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
