# Retail Radar – Stock analysis backend

Beginner-friendly stock evaluation: **risk level** (Random Forest) and **price prediction** (1 / 5 / 10 years).

## Data & connection

- **Historical OHLCV (main data for models):** from **yfinance** – no API key.  
  `fetcher.get_candles(symbol, period)` returns the DataFrame used for features and modeling.
- **Live quote (optional):** from **Finnhub**. Set `FINNHUB_API_KEY` in `.env` to enable; app still works without it using the latest close from history.

So the app is **correctly connected**: Finnhub is optional for live price; the “database” for the pipeline is yfinance.

## Backend layout

| File | Role |
|------|------|
| `fetcher.py` | Get quote (Finnhub) + historical candles (yfinance), normalize columns |
| `processor.py` | Build features (returns, volatility, MA, RSI, etc.) and risk labels |
| `risk_model.py` | Random Forest classifier → risk level **Low / Medium / High** |
| `price_model.py` | Compound-growth (and optional regression) → **predicted price** at 1, 5, or 10 years |
| `backend.py` | Single entry: `run_pipeline(symbol, horizon_years)` → both outputs |

## Outputs

1. **Predicted price**  
   At 1, 5, or 10 years (user choice), from historical annualized return and compound growth.

2. **Risk level**  
   Low / Medium / High from a 3-class Random Forest on:  
   `volatility_20`, `ma_5`, `ma_20`, `rsi`, `volume`, `daily_return`.

## Usage

```bash
pip install -r requirements.txt
```

```python
from backend import run_pipeline

result = run_pipeline("AAPL", horizon_years=5)
# result["predicted_price"]  → { "value", "years", "current_price", ... }
# result["risk"]             → { "level": "Low"|"Medium"|"High", "confidence", "probabilities" }
```

Optional API server (for the HTML frontend):

```bash
pip install flask
python api_server.py
# GET /api/analyze?symbol=AAPL&years=1  → JSON with predicted_price and risk
```

## References

- [Finnhub](https://finnhub.io/) – optional live quote
