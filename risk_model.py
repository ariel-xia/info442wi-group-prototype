import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

FEATURES = ['volatility_20', 'ma_5', 'ma_20', 'rsi', 'volume', 'daily_return']
TARGET   = 'high_risk'

def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def predict_risk(model, latest_row: pd.DataFrame) -> str:
    prob = model.predict_proba(latest_row[FEATURES])[0][1]
    label = "HIGH RISK" if prob > 0.5 else "LOW RISK"
    return f"{label}  (confidence: {prob:.1%})"