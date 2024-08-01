import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Function to load and preprocess data
def load_data():
    data = pd.read_csv("DOGE-USD.csv")
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
    data.set_index('Date', inplace=True)
    data = data.dropna()
    data["gap"] = (data["High"] - data["Low"]) * data["Volume"]
    data["y"] = data["High"] / data["Volume"]
    data["z"] = data["Low"] / data["Volume"]
    data["a"] = data["High"] / data["Low"]
    data["b"] = (data["High"] / data["Low"]) * data["Volume"]
    data = data[["Close", "Volume", "gap", "a", "b"]]
    return data

# Function to fit SARIMAX model and get predictions
def fit_model(data):
    df2 = data.tail(30)
    train = df2[:11]
    test = df2[-19:]
    
    model = SARIMAX(endog=train["Close"], exog=train.drop("Close", axis=1), order=(2, 1, 1))
    results = model.fit()
    
    start = 11
    end = 29
    predictions = results.predict(start=start, end=end, exog=test.drop("Close", axis=1))
    
    mae = mean_absolute_error(test["Close"], predictions)
    mse = mean_squared_error(test["Close"], predictions)
    rmse = sqrt(mse)
    
    forecast = results.get_forecast(steps=len(test), exog=test.drop("Close", axis=1))
    forecast_conf_int = forecast.conf_int()
    
    return train, test, predictions, mae, mse, rmse, forecast_conf_int

# Streamlit UI
def main():
    st.title("DOGE-USD Analysis and Forecasting")

    st.sidebar.header("Options")
    show_data = st.sidebar.checkbox("Show Raw Data")
    show_correlation = st.sidebar.checkbox("Show Correlation Matrix")
    show_plot = st.sidebar.checkbox("Show Time Series Plot")
    show_model_results = st.sidebar.checkbox("Show Model Results")
    
    data = load_data()
    
    if show_data:
        st.subheader("Raw Data")
        st.write(data.head())
    
    if show_correlation:
        st.subheader("Correlation Matrix")
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    
    if show_plot:
        st.subheader("Time Series Plot")
        x = data.groupby(data.index)['Close'].mean()
        fig, ax = plt.subplots(figsize=(20, 7))
        x.plot(linewidth=2.5, color='b', ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title("Date vs Close of 2021")
        st.pyplot(fig)
    
    train, test, predictions, mae, mse, rmse, forecast_conf_int = fit_model(data)
    
    if show_model_results:
        st.subheader("Model Results")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")
        
        st.subheader("Predictions vs Actual")
        fig, ax = plt.subplots(figsize=(12, 6))
        test["Close"].plot(legend=True, color='blue', label='Actual Close', ax=ax)
        predictions.plot(label='Predicted Close', color='red', ax=ax)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Actual vs Predicted Close Prices')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Residuals")
        residuals = test["Close"] - predictions
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(residuals, color='green')
        ax.set_title('Residuals of the SARIMAX Model')
        ax.set_xlabel('Date')
        ax.set_ylabel('Residuals')
        st.pyplot(fig)
        
        st.subheader("Forecast with Confidence Intervals")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test.index, test["Close"], label='Actual Close', color='blue')
        ax.plot(predictions.index, predictions, label='Predicted Close', color='red')
        ax.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='red', alpha=0.3)
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Forecast with Confidence Intervals')
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
