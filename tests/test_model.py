#automated tests for model here

import pandas as pd
from model_utils import add_features, train_model
import yfinance as yf
from model_utils import add_features, train_model, evaluate_accuracy

def test_target_logic():
    df = pd.DataFrame({"Close": [100, 105, 103, 110, 108, 115, 120, 121, 119, 122]})
    df = add_features(df)
    assert not df.empty
    row = df.iloc[0]
    assert "Target_1d" in df.columns
    assert row["Target_1d"] in (0, 1)

def test_model_accuracy_above_threshold():

    df = yf.download("AAPL", start="2022-01-01", end="2024-01-01")[["Close"]]
    df = add_features(df)
    models = train_model(df)
    accuracies = evaluate_accuracy(df, models)
    for acc in accuracies.values():
        assert acc >= 0.5 #expected minimum accuracy

def test_training_time_under_limit():
    import time
    df = yf.download("AAPL", start = "2022-01-01", end = "2024-01-01")[["Close"]]
    df = add_features(df)
    start = time.time()
    train_model(df)
    assert(time.time() - start) < 5.0