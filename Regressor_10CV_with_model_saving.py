# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:49:02 2024

@author: aleniak
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import concurrent.futures
import joblib

def ML_regression(model_name):
    # Global list to store results from each processed file
    global_results = []

    def log_details(test_y, predicted_y, mse, rmse, mae, r2, fold_index, file_name, debug_logs):
        """
        Logs detailed metrics for each fold of cross-validation.
        """
        squared_errors = (test_y - predicted_y) ** 2
        absolute_errors = abs(test_y - predicted_y)
        for actual, predicted, sq_error, abs_error in zip(test_y, predicted_y, squared_errors, absolute_errors):
            debug_logs.append({
                'Fold': fold_index,
                'Actual': round(actual, 3),
                'Predicted': round(predicted, 3),
                'Squared Error': round(sq_error, 3),
                'Absolute Error': round(abs_error, 3),
            })
        debug_logs.append({
            'Fold': fold_index,
            'MSE': round(mse, 3),
            'RMSE': round(rmse, 3),
            'MAE': round(mae, 3),
            'R2': round(r2, 3),
        })

    def process_csv(csv_file, pbar):
        """
        Processes a single CSV file to perform regression, log details, and compute statistics.
        """
        debug_logs = []
        try:
            base_file_name = os.path.basename(csv_file).split('.')[0]
            debug_dir = f'./{input_last_dir_name}/{model_name}/DEBUG'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            debug_file_name = f"{debug_dir}/{script_name}_{base_file_name}_DEBUG.csv"

            rmse_values = []
            mae_values = []
            r2_values = []

            start_time = time.time()

            data = pd.read_csv(csv_file)
            X = data.iloc[:, 1:]
            y = data.iloc[:, 0]

            if len(data) <= 1:
                raise ValueError("Insufficient data points in file: {}".format(csv_file))
            
            # Add support for SVR, AdaBoost, and GradientBoosting models

            if model_name == 'SVR':  # Code for SVR regression
                svr_regressor = SVR()
                kf = KFold(n_splits=10, shuffle=True, random_state=42)

                for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    svr_regressor.fit(X_train, y_train)
                    y_pred = svr_regressor.predict(X_test)
                    
                    # Zapisanie modelu po wytrenowaniu
                    model_filename = f"{debug_dir}/{script_name}_SVR_model.pkl"
                    joblib.dump(svr_regressor, model_filename)

                    mse = mean_squared_error(y_test, y_pred)
                    rmse = math.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    log_details(y_test, y_pred, mse, rmse, mae, r2, fold_index, csv_file, debug_logs)

                    rmse_values.append(rmse)
                    mae_values.append(mae)
                    r2_values.append(r2)

            elif model_name == 'AdaBoost':  # Code for AdaBoost regression
                ada_regressor = AdaBoostRegressor(random_state=42)
                kf = KFold(n_splits=10, shuffle=True, random_state=42)

                for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    ada_regressor.fit(X_train, y_train) 
                    y_pred = ada_regressor.predict(X_test)
                    
                    # Zapisanie modelu po wytrenowaniu
                    model_filename = f"{debug_dir}/{script_name}_AdaBoost_model.pkl"
                    joblib.dump(ada_regressor, model_filename)

                    mse = mean_squared_error(y_test, y_pred)
                    rmse = math.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    log_details(y_test, y_pred, mse, rmse, mae, r2, fold_index, csv_file, debug_logs)

                    rmse_values.append(rmse)
                    mae_values.append(mae)
                    r2_values.append(r2)

            elif model_name =='GradientBoosting':   # Code for GradientBoosting regression
                gb_regressor = GradientBoostingRegressor()
                kf = KFold(n_splits=10, shuffle=True, random_state=42)

                for fold_index, (train_index, test_index) in enumerate(kf.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    gb_regressor.fit(X_train, y_train)
                    y_pred = gb_regressor.predict(X_test)

                    model_filename = f"{debug_dir}/{script_name}_GradientBoosting_model.pkl"
                    joblib.dump(gb_regressor, model_filename)

                    mse = mean_squared_error(y_test, y_pred)
                    rmse = math.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    log_details(y_test, y_pred, mse, rmse, mae, r2, fold_index, csv_file, debug_logs)

                    rmse_values.append(rmse)
                    mae_values.append(mae)
                    r2_values.append(r2)

            # Calculate and store averaged metrics
            averaged_rmse = round(np.mean(rmse_values), 3)
            std_dev_rmse = round(np.std(rmse_values), 3)
            averaged_mae = round(np.mean(mae_values), 3)
            averaged_r2 = round(np.mean(r2_values), 3)

            end_time = time.time()
            execution_time = round(end_time - start_time, 2)

            global_results.append({
                'File': base_file_name,
                'Averaged RMSE': averaged_rmse,
                'Standard Deviation of RMSE': std_dev_rmse,
                'MAE values': averaged_mae,
                'R2 values': averaged_r2,
                'Execution Time (seconds)': execution_time
            })

            debug_df = pd.DataFrame(debug_logs)
            debug_df.insert(0, 'File Name', base_file_name)
            debug_df.to_csv(debug_file_name, index=False)
            pbar.update(1) 
        except Exception as e:
            print("Error processing CSV file: {}. {}".format(csv_file, str(e)))
        
    if __name__ == "__main__":
        script_name = os.path.splitext(os.path.basename(__file__))[0]

        if len(sys.argv) < 2:
            print(f"Usage: python {script_name} <directory_path>")
            sys.exit(1)

        directory_path = sys.argv[1]
        input_last_dir_name = os.path.basename(os.path.normpath(directory_path))
        list_csv = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]

        if not list_csv:
            print("No CSV files found in the provided directory.")
            sys.exit(1)

        batch_size = 4  # Adjust the batch size as needed
        file_batches = [list_csv[i:i+batch_size] for i in range(0, len(list_csv), batch_size)]
        total_files = sum(len(batch) for batch in file_batches)

        total_execution_start_time = time.time()

        with tqdm(total=total_files, desc=f"--{model_name}--Processing files", unit="file") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                futures = [executor.submit(process_csv, csv_file, pbar) for batch in file_batches for csv_file in batch]
                concurrent.futures.wait(futures)

        total_execution_end_time = time.time()
        total_execution_time = total_execution_end_time - total_execution_start_time
        print(f"--{model_name}--Total execution time for all CSV files: {total_execution_time:.2f} seconds\n")

        global_output_df = pd.DataFrame(global_results)
        global_dir = f'./{input_last_dir_name}/{model_name}'
        if not os.path.exists(global_dir):
            os.makedirs(global_dir, exist_ok=True)
        global_output_file = f"{global_dir}/{script_name}_output.csv"
        global_output_df.to_csv(global_output_file, index=False)

model = ['SVR', 'AdaBoost', 'GradientBoosting']

for model_name in model:
    ML_regression(model_name)

print("\nCalculations finished\n")
