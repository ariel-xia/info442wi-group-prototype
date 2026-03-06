"""
Optional Flask API so the HTML frontend can call the backend.
Run: pip install flask && python api_server.py
Then: GET /api/analyze?symbol=AAPL&years=1
"""
from flask import Flask, jsonify, request

app = Flask(__name__)

from backend import run_pipeline
from price_model import HORIZONS


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
