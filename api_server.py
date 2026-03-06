"""
Optional Flask API so the HTML frontend can call the backend.
Run: pip install flask && python api_server.py
Then: GET /api/analyze?symbol=AAPL&years=1
"""
from flask import Flask, jsonify, request

app = Flask(__name__)

from backend import run_pipeline
from price_model import HORIZONS
from fetcher import client as finnhub_client

_WATCHLIST_CACHE = None


def _build_watchlist():
    """
    Build a simple curated watchlist.
    - If Finnhub client is available, pull common US stocks and filter to tech/large names.
    - Otherwise, fall back to a small static list.
    """
    static = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corp."},
        {"symbol": "NVDA", "name": "NVIDIA Corp."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "GOOGL", "name": "Alphabet Inc. (Class A)"},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
    ]

    if finnhub_client is None:
        return static

    try:
        symbols = finnhub_client.stock_symbols("US")
    except Exception:
        return static

    tech_like = []
    for s in symbols:
        if s.get("type") != "Common Stock":
            continue
        if s.get("currency") != "USD":
            continue
        exchange = (s.get("exchange") or "").upper()
        if "NASDAQ" not in exchange and "NYSE" not in exchange:
            continue
        tech_like.append(
            {
                "symbol": s.get("symbol"),
                "name": s.get("description") or s.get("symbol"),
            }
        )

    # Take the first 100 as a simple universe (can be refined later)
    if not tech_like:
        return static
    return tech_like[:100]


@app.route("/api/analyze", methods=["GET"])
def analyze():
    """Return predicted price and risk level for symbol. Query: symbol, years (1, 5, or 10)."""
    symbol = (request.args.get("symbol") or "AAPL").strip().upper()
    try:
        years = int(request.args.get("years", 1))
    except (TypeError, ValueError):
        years = 1
    if years not in (1, 5, 10):
        years = 1
    result = run_pipeline(symbol, horizon_years=years)
    return jsonify(result)


@app.route("/api/horizons", methods=["GET"])
def horizons():
    """Return allowed prediction horizons (1, 5, 10 years)."""
    return jsonify({"horizons": list(HORIZONS)})


@app.route("/api/watchlist", methods=["GET"])
def watchlist():
    """
    Return a curated watchlist of symbols.
    Backed by Finnhub when available, with a static fallback.
    """
    global _WATCHLIST_CACHE
    if _WATCHLIST_CACHE is None:
        _WATCHLIST_CACHE = _build_watchlist()
    return jsonify({"watchlist": _WATCHLIST_CACHE})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
