# StockPriceTomorrow2.o

Stock Price Prediction using ensemble of BiLSTM & RNN

## Overview
This Python script utilizes deep learning models to predict the stock price of a selected Nifty50 stock for the following day. The script uses historical stock data from Yahoo Finance and employs a combination of Bidirectional Long Short-Term Memory (BiLSTM) and Simple Recurrent Neural Network (RNN) model for prediction. The predictions are visualized through Streamlit, a web application framework.

## Prerequisites
Python 3.x
Libraries: streamlit, datetime, yfinance, numpy, tensorflow, matplotlib, sklearn, and PIL

## Usage
Access the web application in your browser by navigating to the provided address.

## Functionality
The user can select a Nifty50 stock from a sidebar dropdown menu.
The script retrieves historical stock data from Yahoo Finance, preprocesses it, and trains an ensemble of BiLSTM and RNN models.
The trained models make predictions on both training and test data, and the results are visualized using matplotlib.
Predictions for the selected stock's next day closing price are displayed, along with key metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for both training and test data.

## Files
StockPriceTomorrow2o.py: The main Python script containing the stock price prediction logic.

training_plot.png: Image file containing the plot of actual vs predicted values for the training data.

test_plot.png: Image file containing the plot of actual vs predicted values for the test data.

## Streamlit Web Application
The web application is created using Streamlit and provides an interactive interface for users to select a stock and view predictions.
The application displays the predicted stock price for the next day, along with training and test data visualizations.

## Acknowledgments
The script uses the Yahoo Finance API through the yfinance library for fetching historical stock data.
Machine learning models (LSTM and RNN) are built using TensorFlow.

## Notes
This script is intended for educational and informational purposes and should not be considered financial advice.
The accuracy of predictions may vary, and users should perform due diligence before making any financial decisions based on the predictions.

## Author
Sai Kumar Diguvapatnam
