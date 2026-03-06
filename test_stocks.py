"""
Simple driver script: run the pipeline and print only final predictions.

Run:
    python test_stocks.py
"""
import argparse

from backend import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Show stock predictions (price + risk)")
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        help="Tickers to evaluate",
    )
    args = parser.parse_args()

    horizons = [1, 5, 10]

    print("=" * 60)
    print("STOCK ANALYSIS – final predictions only")
    print("=" * 60)

    for symbol in args.symbols:
        print(f"\n{'─' * 60}")
        print(f"  {symbol}")
        print("─" * 60)

        for h in horizons:
            try:
                result = run_pipeline(symbol, horizon_years=h)
            except Exception as e:
                print(f"  Error for {symbol} ({h}y): {e}")
                continue

            if result.get("error"):
                print(f"  Error for {symbol} ({h}y): {result['error']}")
                continue

            pp = result.get("predicted_price") or {}
            risk = result.get("risk") or {}

            current_price = result.get("current_price") or pp.get("current_price")
            print(f"\n  Horizon: {h} year(s)")
            print(f"    Current price:   ${current_price}")
            print(f"    Predicted price: ${pp.get('value')}")
            print(f"    Risk level:      {risk.get('level')} (confidence: {risk.get('confidence')})")

    print("\n" + "=" * 60)
    print("End of predictions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
