# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:56:16 2024

@author: aleniak
"""

import pandas as pd
import joblib
import sys
import os
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def predict_and_evaluate(csv_file, model_file):
    try:
        # Wczytanie danych z pliku CSV
        data = pd.read_csv(csv_file)
        X = data.iloc[:, 1:]
        y_actual = data.iloc[:, 0]

        # Wczytanie wytrenowanego modelu
        model = joblib.load(model_file)

        # Przeprowadzenie predykcji
        y_pred = model.predict(X)

        # Lista do przechowywania wyników dla każdego wiersza
        results = []

        # Metryki dla globalnych wyników
        rmse_values = []
        mae_values = []

        for actual, predicted in zip(y_actual, y_pred):
            # Obliczenie metryk dla pojedynczego wiersza
            mse = mean_squared_error([actual], [predicted])
            rmse = math.sqrt(mse)
            mae = mean_absolute_error([actual], [predicted])

            # Zaokrąglenie wartości
            predicted_rounded = round(predicted, 2)
            rmse_rounded = round(rmse, 3)
            mae_rounded = round(mae, 3)

            # Zapisanie wyników dla wiersza
            results.append({
                'Actual Value': round(actual, 3),
                'Predicted Value': predicted_rounded,
                'RMSE': rmse_rounded,
                'MAE': mae_rounded
            })

            # Zbieranie metryk dla globalnych obliczeń
            rmse_values.append(rmse)
            mae_values.append(mae)

        # Obliczenie globalnych metryk
        averaged_rmse = round(np.mean(rmse_values), 3)
        std_dev_rmse = round(np.std(rmse_values), 3)
        averaged_mae = round(np.mean(mae_values), 3)
        std_dev_mae = round(np.std(mae_values), 3)
        r2 = round(r2_score(y_actual, y_pred), 4)

        # Dodanie globalnych metryk na końcu wyników
        results.append({
            'Actual Value': 'Global Metrics',
            'Predicted Value': '',
            'RMSE': '',
            'MAE': ''
        })
        results.append({
            'Actual Value': '',
            'Predicted Value': '',
            'RMSE': f'Average RMSE: {averaged_rmse}',
            'MAE': f'Average MAE: {averaged_mae}'
        })
        results.append({
            'Actual Value': '',
            'Predicted Value': '',
            'RMSE': f'Std Dev RMSE: {std_dev_rmse}',
            'MAE': f'Std Dev MAE: {std_dev_mae}'
        })
        results.append({
            'Actual Value': '',
            'Predicted Value': '',
            'RMSE': f'R2: {r2}',
            'MAE': ''
        })

        # Zapisanie wyników do pliku CSV
        result_df = pd.DataFrame(results)
        output_file = os.path.splitext(csv_file)[0] + "_predictions.csv"
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} <csv_file> <model_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    model_file = sys.argv[2]

    predict_and_evaluate(csv_file, model_file)
