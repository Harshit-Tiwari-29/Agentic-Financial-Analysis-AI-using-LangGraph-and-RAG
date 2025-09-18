# ml_model.py
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def train_and_predict(ticker: str):
    """
    Trains a simple linear regression model on historical stock data
    and predicts the price for the next 7 days.
    """
    try:
        # 1. Fetch historical data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="2y") # Use 2 years of data for training
        
        if hist_data.empty:
            return f"Could not fetch historical data for {ticker}. The ticker might be invalid."

        # 2. Feature Engineering
        # We'll use the number of days from the start as our feature
        hist_data = hist_data.reset_index()
        hist_data['days'] = (hist_data['Date'] - hist_data['Date'].min()).dt.days
        
        # 3. Model Training
        X = hist_data[['days']]
        y = hist_data['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 4. Prediction
        last_day = X['days'].max()
        future_days = np.array(range(last_day + 1, last_day + 8)).reshape(-1, 1)
        predictions = model.predict(future_days)
        
        # Format the output
        last_close_price = y.iloc[-1]
        prediction_summary = {
            "ticker": ticker.upper(),
            "last_close_price": f"${last_close_price:.2f}",
            "prediction_next_7_days": [f"${price:.2f}" for price in predictions]
        }
        
        return prediction_summary

    except Exception as e:
        return f"An error occurred during model prediction for {ticker}: {e}"

def predict_stock_price(ticker: str):
    """Wrapper function to be called by the tool."""
    return train_and_predict(ticker)
