"""
Unified backend: data fetch → process → risk prediction (Random Forest) + price prediction (1/5/10y).
Returns: { "predicted_price": {...}, "risk": {...} } for use by the frontend.
"""
from fetcher import get_candles_for_model, get_quote
from processor import add_features, RISK_FEATURES
from risk_model import train as train_risk_model, predict_risk
from price_model import predict_future_price, HORIZONS


def run_pipeline(symbol: str, horizon_years: int = 1) -> dict:
    """
    Run full pipeline for one symbol and chosen horizon (1, 5, or 10 years).
    Returns predicted price for that horizon and risk level (Low/Medium/High).
    """
    horizon_years = int(horizon_years)
    if horizon_years not in HORIZONS:
        horizon_years = min(HORIZONS, key=lambda h: abs(h - horizon_years))

    # 1) Fetch historical data (yfinance is primary DataFrame source)
    df_raw = get_candles_for_model(symbol, horizon_years=horizon_years)
    if df_raw is None or df_raw.empty or len(df_raw) < 50:
        return {
            "symbol": symbol,
            "predicted_price": None,
            "risk": {"level": "Unknown", "confidence": 0.0, "error": "Insufficient history"},
            "error": "Not enough historical data for this symbol.",
        }

    # 2) Process and add features
    try:
        df = add_features(df_raw)
    except Exception as e:
        return {
            "symbol": symbol,
            "predicted_price": None,
            "risk": {"level": "Unknown", "confidence": 0.0},
            "error": f"Feature processing failed: {e}",
        }

    if df.empty or len(df) < 30:
        return {
            "symbol": symbol,
            "predicted_price": None,
            "risk": {"level": "Unknown", "confidence": 0.0},
            "error": "Insufficient data after feature engineering.",
        }

    # 3) Train risk model on this symbol's history and predict latest
    risk_model = train_risk_model(df)
    latest = df.iloc[[-1]]
    risk_out = predict_risk(risk_model, latest)

    # 4) Price prediction for chosen horizon
    price_out = predict_future_price(df, years=horizon_years)

    # Optional: attach current quote from Finnhub if available
    quote = get_quote(symbol)
    current_price = None
    if quote and quote.get("c") is not None:
        current_price = quote["c"]
    if current_price is None and price_out.get("current_price") is not None:
        current_price = price_out["current_price"]

    return {
        "symbol": symbol,
        "current_price": current_price,
        "predicted_price": {
            "value": price_out.get("predicted_price"),
            "years": horizon_years,
            "annualized_return_used": price_out.get("annualized_return_used"),
            "current_price": price_out.get("current_price"),
            "method": price_out.get("method"),
        },
        "risk": {
            "level": risk_out["level"],
            "confidence": risk_out["confidence"],
            "probabilities": risk_out.get("probabilities"),
        },
    }


if __name__ == "__main__":
    import json
    for sym in ["AAPL", "NVDA"]:
        for horizon in [1, 5, 10]:
            result = run_pipeline(sym, horizon_years=horizon)
            print(json.dumps(result, indent=2))
            print("---")
