# Stock Market Prediction Using Machine Learning

## Project Overview

This project focuses on predicting stock prices of selected Nifty50 companies using Machine Learning and Deep Learning techniques. Historical stock market data is collected and processed to identify patterns and generate stock price predictions.

The project compares different machine learning models to analyze their performance and provides an interactive interface for stock prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- XGBoost
- yFinance
- Flask
- Streamlit

## Machine Learning Models

The project uses and compares:

- Random Forest Regressor
- XGBoost Regressor
- LSTM (Long Short-Term Memory)

## Features

- Fetches historical stock market data
- Data preprocessing and normalization
- Stock price prediction
- Comparison of different ML models
- Interactive stock prediction interface
- Visualization of stock price trends
- Model performance evaluation

## Dataset

Historical stock market data is obtained using the yFinance library.

The project focuses on selected companies from the Nifty50 index.

## Project Workflow

1. Collect historical stock data
2. Clean and preprocess the data
3. Normalize the stock price data
4. Split the data into training and testing sets
5. Train machine learning and deep learning models
6. Generate stock price predictions
7. Evaluate model performance
8. Visualize the prediction results

## Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Project Structure

```text
Stock-Market-Prediction
│
├── backend
│   └── main.py
│
├── frontend
│   └── ...
│
├── stock_prediction_app
│   └── ...
│
├── datasets
│   └── ...
│
├── requirements.txt
└── README.md
