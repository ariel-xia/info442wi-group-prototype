"""
Long-horizon stock price prediction (1, 5, or 10 years).
Uses historical annualized return for compound-growth prediction and
optional linear regression on log-price trend for robustness.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from processor import add_features, get_feature_summary


# Supported horizons (years)
HORIZONS = (1, 5, 10)


def _annualized_return_from_returns(daily_returns: pd.Series) -> float:
    """Compound annualized return from daily returns (~252 trading days/year)."""
    if daily_returns is None or len(daily_returns.dropna()) < 2:
        return 0.0
    total = (1 + daily_returns).prod()
    n_days = len(daily_returns.dropna())
    if n_days <= 0:
        return 0.0
    years = n_days / 252.0
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def predict_price_with_growth(
    latest_close: float,
    annualized_return: float,
    years: int,
) -> dict:
    """
    Predict future price using compound growth: P_future = P_now * (1 + r)^years.
    Optionally cap r to a sane range for display (e.g. -50% to +50% annual).
    """
    r = np.clip(annualized_return, -0.5, 0.5)
    predicted = latest_close * ((1 + r) ** years)
    return {
        "predicted_price": round(float(predicted), 2),
        "annualized_return_used": round(float(r), 4),
        "years": years,
        "current_price": round(float(latest_close), 2),
    }


def predict_price_with_regression(processed_df: pd.DataFrame, years: int) -> float | None:
    """
    Simple trend: regress log(close) on time index, then extrapolate.
    Returns predicted price at (last_date + years) or None if insufficient data.
    """
    if processed_df is None or processed_df.empty or "close" not in processed_df.columns:
        return None
    df = processed_df.copy()
    df = df.dropna(subset=["close"])
    if len(df) < 30:
        return None
    y = np.log(df["close"].values)
    X = np.arange(len(y)).reshape(-1, 1)
    # Last date is roughly "today"; we want today + years (approx 252 * years trading days)
    steps_ahead = int(252 * years)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    future_idx = len(y) + steps_ahead - 1
    log_future = model.predict([[future_idx]])[0]
    return float(np.exp(log_future))


def predict_future_price(
    processed_df: pd.DataFrame,
    years: int,
    method: str = "growth",
) -> dict:
    """
    Main entry: predict price in `years` (1, 5, or 10).
    method: "growth" (compound growth from history) or "regression" (log-price trend).
    """
    if years not in HORIZONS:
        years = min(HORIZONS, key=lambda h: abs(h - years))

    summary = get_feature_summary(processed_df)
    if not summary or summary.get("latest_close") is None:
        return {
            "predicted_price": None,
            "current_price": None,
            "years": years,
            "error": "Insufficient data",
        }

    latest_close = summary["latest_close"]
    ann_return = summary.get("annualized_return", 0.0) or 0.0

    out = predict_price_with_growth(latest_close, ann_return, years)
    out["method"] = "compound_growth"

    if method == "regression":
        reg_price = predict_price_with_regression(processed_df, years)
        if reg_price is not None:
            out["predicted_price_regression"] = round(reg_price, 2)
            # Blend or prefer regression if you want
            out["predicted_price"] = round(
                0.5 * out["predicted_price"] + 0.5 * reg_price, 2
            )
            out["method"] = "blend_growth_and_regression"

    return out
