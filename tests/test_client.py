"""Test script for Hyperliquid client - run manually to verify connection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_settings
from src.exchange import HyperliquidClient


def main():
    """Test the Hyperliquid client functionality."""
    print("=" * 50)
    print("Hyperliquid Client Test")
    print("=" * 50)

    # Load settings
    try:
        settings = load_settings()
        print(f"✓ Settings loaded (testnet: {settings.is_testnet})")
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        print("  Make sure to fill in .env with your credentials")
        return

    # Initialize client
    print("\nInitializing client...")
    client = HyperliquidClient(
        private_key=settings.private_key,
        wallet_address=settings.wallet_address,
        is_testnet=settings.is_testnet,
    )

    # Test connection
    print("\n1. Testing connection...")
    if client.is_connected():
        print("   ✓ Connected to Hyperliquid API")
    else:
        print("   ✗ Failed to connect")
        return

    # Test get current price
    print("\n2. Testing get_current_price('BTC')...")
    try:
        price = client.get_current_price("BTC")
        print(f"   ✓ BTC price: ${price:,.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test get account balance
    print("\n3. Testing get_account_balance()...")
    try:
        balance = client.get_account_balance()
        print(f"   ✓ Account Value: ${balance.account_value:,.2f}")
        print(f"   ✓ Margin Used: ${balance.total_margin_used:,.2f}")
        print(f"   ✓ Withdrawable: ${balance.withdrawable:,.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test get open positions
    print("\n4. Testing get_open_positions()...")
    try:
        positions = client.get_open_positions()
        if positions:
            for pos in positions:
                print(f"   ✓ {pos.symbol} {pos.side.upper()}: {pos.size} @ ${pos.entry_price:,.2f}")
        else:
            print("   ✓ No open positions")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test get candles
    print("\n5. Testing get_candles('BTC', '1h', limit=10)...")
    try:
        candles = client.get_candles("BTC", "1h", limit=10)
        print(f"   ✓ Retrieved {len(candles)} candles")
        if not candles.empty:
            print(f"   Latest: {candles.iloc[-1]['timestamp']} | Close: ${candles.iloc[-1]['close']:,.2f}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
