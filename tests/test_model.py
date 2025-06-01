#automated tests for model here

import pandas as pd
from model_utils import add_features, train_model
import yfinance as yf

def test_model_accuracy_above_threshold():

    df = yf.download("AAPL", start="2022-01-01", end="2024-01-01")
    df = df[["Close"]]
    df = add_features(df)
    model, accuracy = train_model(df)
    assert accuracy > 0.5, f"Accuracy too low {accuracy}"
    print(f"Model accuracy: {accuracy}")