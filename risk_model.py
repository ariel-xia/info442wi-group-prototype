"""
Random Forest classifier for stock risk level: Low (0), Medium (1), High (2).

Simplified design:
- use this stock's past days with a known `risk_level`
- train a single RandomForestClassifier
- use it to predict today's risk level.

We do NOT return accuracy or other metrics – only the trained model.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from processor import RISK_FEATURES

TARGET = "risk_level"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest on past labeled days for this stock.
    """
    for col in RISK_FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing column for risk model: {col}")

    # Only rows with known label (last row is NaN for next_return / risk_level)
    train_df = df.dropna(subset=[TARGET])
    if len(train_df) < 50:
        raise ValueError("Not enough rows with risk_level for training")

    X = train_df[RISK_FEATURES]
    y = train_df[TARGET]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False,  # respect time order
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_risk(model: RandomForestClassifier, latest_row: pd.DataFrame) -> dict:
    """
    Predict risk level for the latest row. Returns label and confidence.
    latest_row: single row DataFrame with RISK_FEATURES.
    """
    if latest_row is None or latest_row.empty:
        return {"level": "Unknown", "confidence": 0.0, "probabilities": None}

    X = latest_row[RISK_FEATURES]
    proba = model.predict_proba(X)[0]
    pred_class = int(model.predict(X)[0])
    confidence = float(proba[pred_class])

    return {
        "level": RISK_LABELS.get(pred_class, "Unknown"),
        "confidence": round(confidence, 3),
        "probabilities": {
            RISK_LABELS[i]: round(float(p), 3) for i, p in enumerate(proba)
        },
    }
