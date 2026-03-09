"""
Simple driver script: run the pipeline and print only final predictions.

Run:
    python test_stocks.py
"""
import argparse
import json
import time
import uuid

from backend import run_pipeline


def _debug_log(message: str, data: dict, *, hypothesis_id: str, run_id: str) -> None:
    # region agent log
    try:
        ts = int(time.time() * 1000)
        payload = {
            "sessionId": "dfedf4",
            "id": f"log_{ts}_{uuid.uuid4().hex[:8]}",
            "timestamp": ts,
            "location": "test_stocks.py",
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with open(
            "/Users/jkb/Desktop/info442wi-group-prototype/.cursor/debug-dfedf4.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # endregion


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
            run_id = f"repro-issue_{symbol}_{h}"
            try:
                _debug_log(
                    "run_pipeline call",
                    {"symbol": symbol, "horizon_years": h},
                    hypothesis_id="H1",
                    run_id=run_id,
                )
                result = run_pipeline(symbol, horizon_years=h)
            except Exception as e:
                _debug_log(
                    "run_pipeline exception",
                    {"symbol": symbol, "horizon_years": h, "exc_type": type(e).__name__, "exc": str(e)},
                    hypothesis_id="H2",
                    run_id=run_id,
                )
                print(f"  Error for {symbol} ({h}y): {e}")
                continue

            if result.get("error"):
                _debug_log(
                    "run_pipeline returned error",
                    {"symbol": symbol, "horizon_years": h, "error": result.get("error")},
                    hypothesis_id="H3",
                    run_id=run_id,
                )
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
