"""
Feature engineering for stock DataFrame (OHLCV from fetcher).
Produces features used by the risk model and summary stats for price prediction.
"""
import pandas as pd
import numpy as np

# Features used by the Random Forest risk model (must exist after add_features)
RISK_FEATURES = [
    "volatility_20",
    "ma_5",
    "ma_20",
    "rsi",
    "volume",
    "daily_return",
]

# Raw OHLCV columns we expect from fetcher
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical features and risk labels from OHLCV.
    Returns DataFrame with RISK_FEATURES + risk_level (0=Low, 1=Medium, 2=High).
    """
    df = df.copy()

    # Ensure we have required columns (yfinance uses lowercase after normalize)
    for col in ["close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}'. Got: {list(df.columns)}")

    df["daily_return"] = df["close"].pct_change()
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    df["bb_upper"] = df["ma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["ma_20"] - 2 * df["close"].rolling(20).std()

    # Next-day return (for training only; last row will be NaN)
    df["next_return"] = df["daily_return"].shift(-1)

    # 3-class risk label from next day return (for Random Forest)
    # High=2: next return < -2%; Medium=1: next return in [-2%, 0.5%]; Low=0: else
    df["high_risk"] = (df["next_return"] < -0.02).astype(int)
    df["risk_level"] = np.where(
        df["next_return"] < -0.02, 2, np.where(df["next_return"] < 0.005, 1, 0)
    )

    # Keep rows with all features (last row has features but NaN risk_level; keep it for prediction)
    return df.dropna(subset=RISK_FEATURES)


def get_feature_summary(df: pd.DataFrame) -> dict:
    """
    Summary stats from processed data for price prediction:
    annualized return, annualized volatility, latest close.
    """
    if df.empty or "daily_return" not in df.columns:
        return {}
    # ~252 trading days per year
    daily_returns = df["daily_return"].dropna()
    n = len(daily_returns)
    if n < 2:
        return {}
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    if std_daily and not np.isnan(std_daily):
        annual_return = (1 + mean_daily) ** 252 - 1
        annual_vol = std_daily * np.sqrt(252)
    else:
        annual_return = 0.0
        annual_vol = 0.0
    latest_close = float(df["close"].iloc[-1]) if "close" in df.columns else None
    return {
        "annualized_return": annual_return,
        "annualized_volatility": annual_vol,
        "latest_close": latest_close,
        "trading_days": n,
    }
