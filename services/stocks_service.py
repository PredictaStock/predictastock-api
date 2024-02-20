from flask import Flask, request, jsonify
import yfinance as yf
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from inference.inference import Inference
from constants import correlations

app = Flask(__name__)

@app.route('/stock/<stock_name>')
def get_stock_prices_route(stock_name):
    n_days = request.args.get('days', default=15, type=int)
    stock_data = yf.download(stock_name, period=f'{n_days}d')
    prices = stock_data['Close'].tolist()
    return jsonify(prices)

@app.route('/stock/<stock_name>/prediction')
def get_stock_price_prediction(stock_name):
    # Get stock data for the given stock and correlated stocks
    stock_data = yf.download([stock_name] + correlations[stock_name]['top_correlated'] + correlations[stock_name]['bottom_correlated'], period='10d')
    stock_close_prices = stock_data['Close']
    stock_volumes = stock_data['Volume']

    # Extract the features for prediction
    features_values = []

    # Extract features for the target stock (stock itself)
    for i in range(1, 10):
        features_values.append(stock_close_prices[stock_name].shift(i).iloc[-1])  # Close price
    
    for i in range(1, 10):
        features_values.append(stock_volumes[stock_name].shift(i).iloc[-1])  # Volume

    # Extract features for the top correlated stocks
    for top_correlated_stock in correlations[stock_name]['top_correlated']:
        for i in range(1, 10):
            features_values.append(stock_close_prices[top_correlated_stock].shift(i).iloc[-1])  # Close price
        for i in range(1, 10):
            features_values.append(stock_volumes[top_correlated_stock].shift(i).iloc[-1])  # Volume

    # Extract features for the bottom correlated stocks
    for bottom_correlated_stock in correlations[stock_name]['bottom_correlated']:
        for i in range(1, 10):
            features_values.append(stock_close_prices[bottom_correlated_stock].shift(i).iloc[-1])  # Close price
        for i in range(1, 10):
            features_values.append(stock_volumes[bottom_correlated_stock].shift(i).iloc[-1])  # Volume

    print(features_values)

    # Instantiate the Inference class and get prediction
    predictor = Inference(stock_name, features_values)
    prediction = predictor.predictor()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
    