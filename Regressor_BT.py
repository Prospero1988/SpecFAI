import pandas as pd
from sklearn.utils import resample
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

def ML_regression(model_name):
    # Global list to store results from each processed file
    global_results = []

    def log_details(test_y, predicted_y, mse, rmse, mae, r2, bootstrap_sample_index, file_name, debug_logs):
        """
        Logs detailed metrics for each bootstrap sample.
        """
        squared_errors = (test_y - predicted_y) ** 2
        absolute_errors = abs(test_y - predicted_y)
        for actual, predicted, sq_error, abs_error in zip(test_y, predicted_y, squared_errors, absolute_errors):
            debug_logs.append({
                'Sample': bootstrap_sample_index,
                'Actual': round(actual, 2),
                'Predicted': round(predicted, 2),
                'Squared Error': round(sq_error, 2),
                'Absolute Error': round(abs_error, 2),
            })
        debug_logs.append({
            'Sample': bootstrap_sample_index,
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'R2': round(r2, 2),
        })

    def process_csv(csv_file, pbar):
        """
        Processes a single CSV file to perform regression, log details, and compute statistics using bootstrapping.
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

            # Instantiate the model based on model_name
            if model_name == 'SVR':
                model = SVR()
            elif model_name == 'AdaBoost':
                model = AdaBoostRegressor(random_state=42)
            elif model_name == 'GradientBoosting':
                model = GradientBoostingRegressor()

            # Perform bootstrapping
            n_iterations = 10
            for i in range(n_iterations):
                X_sample, y_sample = resample(X, y, random_state=i)
                X_train, y_train = X_sample, y_sample
                X_test, y_test = X[~X.index.isin(X_sample.index)], y[~y.index.isin(y_sample.index)]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = math.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                log_details(y_test, y_pred, mse, rmse, mae, r2, i, csv_file, debug_logs)

                rmse_values.append(rmse)
                mae_values.append(mae)
                r2_values.append(r2)

            # Calculate and store averaged metrics
            averaged_rmse = round(np.mean(rmse_values), 2)
            std_dev_rmse = round(np.std(rmse_values), 2)
            averaged_mae = round(np.mean(mae_values), 2)
            std_dev_mae = round(np.std(mae_values), 2)
            averaged_r2 = round(np.mean(r2_values), 2)

            end_time = time.time()
            execution_time = round(end_time - start_time, 2)

            global_results.append({
                'File': base_file_name,
                'Averaged RMSE': averaged_rmse,
                'Standard Deviation of RMSE': std_dev_rmse,
                'MAE values': averaged_mae,
                'Standard Deviation of MAE': std_dev_mae,
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
