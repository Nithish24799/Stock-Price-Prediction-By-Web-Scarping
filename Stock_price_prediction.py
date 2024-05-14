import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def get_stock_data(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}/history"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find_all("table")[0]
    rows = table.find_all("tr")[1:]
    data = []
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        data.append(cols)
    return data

def preprocess_data(stock_data):
    df = pd.DataFrame(stock_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
    df.set_index('Date', inplace=True)
    return df

def create_features(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    return df.dropna()


def compute_rsi(close, window=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def main():
    stock_symbol = "AAPL"
    stock_data = get_stock_data(stock_symbol)
    df = preprocess_data(stock_data)
    df = create_features(df)
    X = df.drop(columns=['Close'])
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    rmse = evaluate_model(model, X_test, y_test)
    print(f"Root Mean Squared Error: {rmse}")
    future_dates = pd.date_range(df.index[-1], periods=5, freq='B')
    future_features = pd.DataFrame(index=future_dates, columns=X.columns)
    future_features.fillna(method='ffill', inplace=True)
    future_predictions = model.predict(future_features)
    print("Predicted closing prices for the next 5 days:")
    print(future_predictions)

if __name__ == "__main__":
    main()
