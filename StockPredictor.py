import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime, timedelta

# Title
st.title("Stock Prediction Web App")

# Sidebar inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Function to load stock data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Load and validate data
try:
    stock_data = load_data(ticker, start_date, end_date)

    # Validate date rangae
    if len(stock_data) < 15:  # Require at least 15 data points
        st.error("Error: Selected date range is too short. Please choose a longer date range (at least 1 month).")
        st.stop()

    st.subheader(f"{ticker} Stock Data")
    st.write(stock_data.tail())

    # Plot historical prices
    st.subheader("Closing Price Trend")
    fig, ax = plt.subplots()
    ax.plot(stock_data['Close'], label="Closing Price")
    plt.legend()
    st.pyplot(fig)

    # Prepare data for LSTM
    st.subheader("Prediction Model")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']])

    # Dynamically adjust time_step based on available data
    time_step = min(10, len(stock_data) - 1)

    # Create dataset
    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    # Make future predictions
    st.subheader("Future Predictions")
    future_days = st.sidebar.slider("Predict next N days", 1, 30, 7)

    # Generate future predictions
    last_data = scaled_data[-time_step:]
    future_preds = []

    for _ in range(future_days):
        input_data = last_data.reshape(1, time_step, 1)
        pred = model.predict(input_data, verbose=0)[0][0]
        future_preds.append(pred)
        last_data = np.append(last_data[1:], [[pred]], axis=0)

    # Rescale predictions back to original values
    future_preds_rescaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

    # Display predictions
    future_dates = pd.date_range(stock_data.index[-1] + timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_preds_rescaled.flatten()})
    st.write(future_df)

    # Plot predictions
    st.subheader("Future Predictions Plot")
    fig2, ax2 = plt.subplots()
    ax2.plot(stock_data['Close'], label="Historical Prices", color='blue')
    ax2.plot(future_df['Date'], future_df['Predicted_Close'], label="Future Predictions", linestyle='--', color='red')
    plt.legend()
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("**Created by Aaron Aberasturia**")
