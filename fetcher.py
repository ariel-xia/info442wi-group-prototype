"""
Data fetching for stock analysis. Uses Finnhub for real-time quotes (optional)
and yfinance for historical OHLCV data (the main DataFrame source for models).
"""
import os
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

try:
    import finnhub
    _finnhub_key = os.getenv("FINNHUB_API_KEY")
    client = finnhub.Client(api_key=_finnhub_key) if _finnhub_key else None
except ImportError:
    client = None

def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Current quote from Finnhub. Returns None if API key missing or request fails."""
    if client is None:
        return None
    try:
        return client.quote(symbol)
    except Exception:
        return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to lowercase single-level names (Open, High, Low, Close, Volume)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # With group_by="ticker", level 0 is ticker (e.g. AAPL), level 1 is metric (Open, Close, ...)
        level_0 = df.columns.get_level_values(0)
        level_1 = df.columns.get_level_values(1)
        # Use level that looks like OHLCV (has 'open'/'close' etc), else level 0
        if len(level_1) and str(level_1[0]).lower() in ("open", "high", "low", "close", "volume", "adj close"):
            df.columns = [str(c).lower() for c in level_1]
        else:
            df.columns = [str(c).lower() for c in level_0]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df


def get_candles(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """
    Historical OHLCV from yfinance (primary DataFrame for analysis).
    period: '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'
    """
    df = yf.download(
        symbol,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if df.empty:
        return df
    # Single ticker can still come with MultiIndex; flatten to Open, High, Low, Close, Volume
    df = _normalize_columns(df)
    # yfinance sometimes uses 'adj close' – prefer 'close' for consistency
    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]
    return df


def get_candles_for_model(symbol: str, horizon_years: int = 10) -> pd.DataFrame:
    """Fetch enough history for modeling and long-horizon prediction (e.g. 1/5/10 year)."""
    period = "10y" if horizon_years >= 5 else "2y"
    return get_candles(symbol, period=period)


if __name__ == "__main__":
    q = get_quote("AAPL")
    print("Quote (Finnhub):", q if q else "(no API key or error)")
    df = get_candles("AAPL", period="1y")
    print("Candles columns:", list(df.columns))
    print(get_candles("AAPL").tail())
