import pytest
import pandas as pd
from model_utils import add_features, train_model, evaluate_accuracy

def test_add_features_output():
    data = pd.DataFrame({
            "Close": [100, 102, 101, 105, 110, 108, 111, 115, 120, 125, 130]
    })
    df = add_features(data)
    assert "MA5" in df.columns
    assert not df["MA5"].isnull().all()

def test_train_model_creates_models():
    data = pd.DataFrame({
        "Close": [100 + i for i in range(60)]
    })
    df = add_features(data)
    models = train_model(df)
    accuracies = evaluate_accuracy(df, models)
    for acc in accuracies.values():
        assert acc >= 0.5
