from flask import Flask, render_template,request,redirect
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# Initialize Flask app
app = Flask(__name__)

#email and password
users = {
    "admin": "admin123",
    "user": "user123",
    "test": "test123",
    "shaiksharifa": "sharifa123",
    
    }
# --- Main Stock App ---
def stock_prediction_app():

    st.title("📈 Stock Price Prediction")
   
# User input: Ticker and date range
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT)", value='AAPL')
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
#---predtict button---
predict_button = st.button("Predict Stock Prices")
if ticker:
    st.write(f"Fetching data for {ticker}...")

    # --- Fetch Stock Data ---
    data_load_state = st.text("Loading data...")
    data = yf.download(ticker,  start=start_date, end=end_date)
    st.write("Data Sample:")
    st.dataframe(data.tail())
    
    # --- Prepare Data ---
    close_prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # --- Create Time-Series Dataset ---
    time_steps = 60
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_LSTM = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_LSTM = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # --- Train LSTM Model ---
    st.write("Training LSTM model...")
    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_LSTM.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train_LSTM, y_train, epochs=10, batch_size=32, verbose=0)

    # --- Train Random Forest ---
    st.write("Training Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # --- Train XGBoost ---
    st.write("Training XGBoost model...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    xgb.fit(X_train, y_train)

    # --- Make Predictions ---
    pred_lstm = scaler.inverse_transform(model_lstm.predict(X_test_LSTM))
    pred_rf = scaler.inverse_transform(rf.predict(X_test).reshape(-1, 1))
    pred_xgb = scaler.inverse_transform(xgb.predict(X_test).reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    st.success("prediction completed successfully!")

    #---- Graphs Models ----
    model_choice = st.selectbox("Select Model to View Predictions", ("LSTM", "Random Forest", "XGBoost",  "All Models"))   
    if model_choice == "LSTM":
        st.subheader("LSTM Model Predictions")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(actual, color='black', label='Actual')
        ax.plot(pred_lstm, color='orange', linestyle='--', label='LSTM Prediction')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)  
    elif model_choice == "Random Forest":
        st.subheader("Random Forest Model Predictions")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(actual, color='black', label='Actual')
        ax.plot(pred_rf, color='green', linestyle='--', label='Random Forest Prediction')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price ")
        ax.legend()
        st.pyplot(fig)
    elif model_choice == "XGBoost":
        st.subheader("XGBoost Model Predictions")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(actual, color='black', label='Actual')
        ax.plot(pred_xgb, color='red', linestyle='--', label='XGBoost Prediction')
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

    # --- Optional: Ensemble Average ---
    st.subheader("Ensemble Average Prediction")
    final_pred = (pred_lstm + pred_rf + pred_xgb) / 3
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(actual, color='black', label='Actual')
    ax.plot(final_pred, color='purple', linestyle='--', label='Ensemble Prediction')
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (INR)")
    ax.legend()
    st.pyplot(fig)
    # --- Evaluation Metrics ---
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    st.subheader("Model Evaluation Metrics")
    metrics = {
        "Model": ["LSTM", "Random Forest", "XGBoost"],
        "RMSE": [rmse(actual, pred_lstm), rmse(actual, pred_rf), rmse(actual, pred_xgb)],
        "MAPE (%)": [mape(actual, pred_lstm), mape(actual, pred_rf), mape(actual, pred_xgb)]
    }
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)
    data_load_state.text("Data loaded and models trained!")

    if st.checkbox("Show raw data"):
            st.dataframe(data)
else:
    st.warning("Please enter a valid ticker symbol for prediction.")
#----routes----
@app.route("/")
def home():
    return render_template("login.html")
#----handle login----
@app.route("/login", methods=["POST"])
def login():    
    username = request.form.get("username")
    password = request.form.get("password")
    print(f"Attempted login with username: {username} and password: {password}")
   
    if username in users and users[username] == password:
        return redirect("http://localhost:8501/")
    else:
        return "Login Failed. Invalid username or password.<a href='/'>Try Again</a>"

    #----run app----
if __name__ == "__main__":   
    app.run(debug=True, use_reloader=False)