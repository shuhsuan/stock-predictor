#get data, train, and predict here

import yfinance as yf
import pandas as pd
from model_utils import add_features, train_model, predict_for_date, evaluate_accuracy, run_etl
import logging


logging.basicConfig(filename='predictions.log', level=logging.INFO)

ticker = input("Enter stock ticker (eg. AAPL): ")
date_input = input("Enter date (YYYY-MM-DD): ")

data = yf.download(ticker, start="2010-01-01", end="2024-12-31")[["Close"]]
data.columns = ["Close"]
data = add_features(data)

models = train_model(data)
predictions = predict_for_date(data, models, pd.to_datetime(date_input))
accuracies = evaluate_accuracy(data, models)

if predictions:
    vol = predictions.pop("5 day volatility", None)
    print(f"\nPredictions for {ticker.upper()} after {date_input}: (based on data from Jan 2010 - Dec 2024)")
    for time_frame, result in predictions.items():
        if time_frame in accuracies:
            print(f" {time_frame}: {result} (Accuracy: {accuracies[time_frame]})")
        else: 
            print(f" {time_frame}: {result}")
        logging.info(f"{ticker} on {date_input} - {time_frame}: {result}")
    if vol:
        print(f"\n5 day Volatility on {date_input}: {vol}")
else:
    print("No predictions could be made for the selected date :(")
    print("Available dates:", data.index[-5:])

    print("\nModel Accuracies")
    for k, v in accuracies.items():
        print(f" {k}: {v}")








