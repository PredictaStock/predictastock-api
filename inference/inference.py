import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

class Inference:
    def __init__(self, stock_name, features_values):
        self.stock_name = stock_name
        self.feat_val = features_values
        self.model = self.load_model()

    def load_model(self):
        # Load the trained model
        model_filename = f'models/stock_models/model_{self.stock_name}_svr.pkl'
        loaded_model = joblib.load(model_filename)
        return loaded_model

    def predictor(self):
        fit_file = pd.read_csv(f'models/fit_files/series_{self.stock_name}.csv')

        target_column_name = fit_file.columns.values[0]

        fit_sr = fit_file.loc[:, fit_file.columns != target_column_name]

        sc = StandardScaler()

        sc.fit(fit_sr)

        X_input = np.array([self.feat_val])

        X_input = sc.transform(X_input)

        prediction = self.model.predict(X_input)

        return prediction

# Example usage:
stock_name = "UNH"
feat_val = [516.8499755859375, 517.6400146484375, 518.219970703125, 520.0900268554688, 519.3900146484375, 3436000.0, 3194000.0, 2715400.0, 3343000.0, 4109100.0, 156.47000122070312, 157.85000610351562, 156.75999450683594, 156.39999389648438, 157.97999572753906, 8620300.0, 6701500.0, 6529200.0, 9755000.0, 8690300.0, 275.80999755859375, 275.07000732421875, 276.42999267578125, 275.7799987792969, 278.8699951171875, 5493500.0, 3676200.0, 4618300.0, 5833400.0, 4785900.0, 174.25999450683594, 175.7899932861328, 175.00999450683594, 174.8000030517578, 175.42999267578125, 8395000.0, 8539300.0, 6296700.0, 6060300.0, 7225500.0, 184.02000427246094, 188.1300048828125, 193.57000732421875, 189.55999755859375, 187.5800018310547, 86759500.0, 95498600.0, 84476300.0, 83034000.0, 111535200.0, 460.1199951171875, 468.8999938964844, 468.1099853515625, 470.0, 469.5899963378906, 20916600.0, 19382000.0, 18413100.0, 18815100.0, 23066000.0, 168.63999938964844, 172.33999633789062, 174.4499969482422, 169.83999633789062, 170.52999877929688, 56300500.0, 51050400.0, 56986000.0, 42316500.0, 47174100.0]
predictor = Inference(stock_name, feat_val)
prediction = predictor.predictor()
print(f"Prediction for {stock_name}: {prediction}")
