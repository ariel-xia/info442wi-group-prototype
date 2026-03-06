"""
Long-horizon stock price prediction (1, 5, or 10 years).

Simplified design:
- use this stock's past daily data
- train a single regression model (RandomForestRegressor) to predict 1‑year forward return
- use that predicted 1‑year return as the annual growth rate
  for 1, 5, or 10‑year price forecasts.

We only expose the final predicted prices, not accuracy/R² metrics.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from processor import get_feature_summary, RISK_FEATURES


# Supported horizons (years)
HORIZONS = (1, 5, 10)

# Reuse the same feature set as the risk model
PRICE_FEATURES = RISK_FEATURES


def predict_price_with_growth(
    latest_close: float,
    annualized_return: float,
    years: int,
) -> dict:
    """
    Predict future price using compound growth: P_future = P_now * (1 + r)^years.
    Caps r to [-50%, +50%] per year to avoid extreme outputs.
    """
    r = np.clip(annualized_return, -0.5, 0.5)
    predicted = latest_close * ((1 + r) ** years)
    return {
        "predicted_price": round(float(predicted), 2),
        "annualized_return_used": round(float(r), 4),
        "years": years,
        "current_price": round(float(latest_close), 2),
    }


def _train_return_model(
    processed_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train a RandomForestRegressor to predict 1‑year forward return.

    Target: future 1‑year return = close(t+252 trading days) / close(t) - 1.
    """
    if processed_df is None or processed_df.empty or "close" not in processed_df.columns:
        return None

    df = processed_df.copy()
    steps_ahead = 252  # ~1 trading year
    df["future_1y_return"] = df["close"].shift(-steps_ahead) / df["close"] - 1

    cols_needed = list(PRICE_FEATURES) + ["future_1y_return"]
    train_df = df.dropna(subset=cols_needed)
    if len(train_df) < 80:
        return None

    X = train_df[PRICE_FEATURES]
    y = train_df["future_1y_return"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False,  # validate on later data
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_future_price(
    processed_df: pd.DataFrame,
    years: int,
) -> dict:
    """
    Predict price in `years` (1, 5, or 10) using a single regression model on past data.

    Steps:
      1. Train a RandomForestRegressor to predict 1‑year forward return.
      2. Use its predicted 1‑year return as the annualized rate r.
      3. Forecast price for 1, 5, or 10 years via compound growth.

    If there is not enough data to train the regressor, fall back to using the
    historical annualized return from `get_feature_summary`.
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

    # 1) Try to train a regression model on 1‑year forward returns
    model = _train_return_model(processed_df)

    if model is not None:
        latest_row = processed_df.dropna(subset=PRICE_FEATURES).iloc[[-1]]
        pred_ret_1y = float(model.predict(latest_row[PRICE_FEATURES])[0])
        ann_return = pred_ret_1y
        out = predict_price_with_growth(latest_close, ann_return, years)
        out["method"] = "learned_regression_1y"
        return out

    # 2) Fallback: simple historical growth if we cannot train a model
    ann_return_hist = summary.get("annualized_return", 0.0) or 0.0
    out = predict_price_with_growth(latest_close, ann_return_hist, years)
    out["method"] = "historical_growth_fallback"
    return out

