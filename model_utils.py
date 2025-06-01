#helper functions for features and models here

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def add_features(df):
    df["Return"] = df["Close"].pct_change()

    df["Target_1d"] = df["Close"].shift(-1) > df["Close"]
    df["Target_3d"] = df["Close"].shift(-3) > df["Close"]
    df["Target_7d"] = df["Close"].shift(-7) > df["Close"]
    df["Target_30d"] = df["Close"].shift(-30) > df["Close"]

    df["Target_1d"] = df["Target_1d"].astype(int)
    df["Target_3d"] = df["Target_3d"].astype(int)
    df["Target_7d"] = df["Target_7d"].astype(int)
    df["Target_30d"] = df["Target_30d"].astype(int)

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df = df.dropna()
    return df

def train_model(df):
    features = ["Close", "MA5", "MA10"]

    models={}
    for target in ["Target_1d", "Target_3d", "Target_7d", "Target_30d"]:
        X = df[features]
        y = df[target]
        model = RandomForestClassifier(n_estimators=100, random_state=1)
        model.fit(X, y)
        models[target] = model

    return models


def predict_for_date(df, models, date):
    df = df.sort_index()

    if date not in df.index:
        earlier_dates = df.index[df.index <= date]
        if len(earlier_dates) == 0:
            print(f"No data available on or before {date.date()}")
            return {}
        date = earlier_dates[-1]

    latest_data = df.loc[[date]]
    features = ["Close", "MA5", "MA10"]

    if latest_data[features].isnull().any().any():
        print(f"Missing data in features for {date.date()}")
        return{}

    results = {}
    for target, model in models.items():
        prediction = model.predict(latest_data[features])
        results[target] = "Up" if prediction[0] == 1 else "Down"

    return results

def evaluate_accuracy(df, models):
    features = ["Close", "MA5", "MA10"]
    accuracy_scores = {}

    for target, model in models.items():
        X = df[features]
        y = df[target]
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        accuracy_scores[target] = round(accuracy, 2)

    return accuracy_scores
