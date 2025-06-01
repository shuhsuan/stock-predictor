#get data, train, and predict here

import yfinance as yf
import pandas as pd
from model_utils import add_features, train_model, predict_next_day

#ze data
data = yf.download("AAPL", start="2023-01-01", end="2025-01-01")
data = data[['Close']]

#features and target
df = add_features(data)

#train model
model, accuracy = train_model(df)

#predict tomorrow
latest_data = df.iloc[-1:]
prediction = predict_next_day(model, latest_data)


print(f"Predicted next movement: {prediction}, Accuracy: {accuracy:.2f}")



