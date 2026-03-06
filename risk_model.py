"""
Random Forest classifier for stock risk level: Low (0), Medium (1), High (2).
Uses features from processor.RISK_FEATURES.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from processor import RISK_FEATURES

TARGET = "risk_level"
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def train(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, verbose: bool = False):
    """
    Train Random Forest on processed DataFrame. Requires risk_level and RISK_FEATURES.
    """
    for col in RISK_FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Missing column for risk model: {col}")

    # Train only on rows with known risk_level (last row is NaN for next_return/risk_level)
    train_df = df.dropna(subset=[TARGET])
    if len(train_df) < 20:
        raise ValueError("Not enough rows with risk_level for training")

    X = train_df[RISK_FEATURES]
    y = train_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    if verbose:
        y_pred = model.predict(X_test)
        print("Risk model (3-class) classification report:")
        print(classification_report(y_test, y_pred, target_names=[RISK_LABELS[i] for i in range(3)]))
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
