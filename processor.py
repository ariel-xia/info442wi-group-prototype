import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['daily_return'] = df['close'].pct_change()

    df['volatility_20'] = df['daily_return'].rolling(20).std()

    df['ma_5']  = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()

    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    df['bb_upper'] = df['ma_20'] + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['ma_20'] - 2 * df['close'].rolling(20).std()

    df['next_return'] = df['daily_return'].shift(-1)
    df['high_risk']   = (df['next_return'] < -0.02).astype(int)

    return df.dropna()