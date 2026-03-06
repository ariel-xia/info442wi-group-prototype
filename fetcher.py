import os
import finnhub
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

def get_quote(symbol: str) -> dict:
    return client.quote(symbol)

def get_candles(symbol: str, period='6mo') -> pd.DataFrame:
    df = yf.download(symbol, period=period, auto_adjust=True)
    df.columns = [col[0].lower() for col in df.columns]
    return df

if __name__ == "__main__":
    print(get_quote("AAPL"))
    print(get_candles("AAPL").tail())