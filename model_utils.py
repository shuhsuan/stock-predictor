#helper functions for features and models here

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int) #as in, 1=up, 0=down
    df["Trend"] = (df["Close"].shift(-3) > df["Close"]).astype(int) #1 is a upward shirt in the next 3 days, 0 is otherwise
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    return df

def train_model(df):
    features = ['Return', 'MA5', 'MA10']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return model, accuracy

def predict_next_day(model, latest_row):
    features = ['Return', 'MA5', 'MA10']
    return model.predict(latest_row[features])[0]

