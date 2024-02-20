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
        model_filename = f'models/stock_models/model_{self.stock_name}.pkl'
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

        return prediction[0]
