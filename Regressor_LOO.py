import pandas as pd
from sklearn.model_selection import LeaveOneOut
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
    """
    Perform Machine Learning regression using Leave-One-Out Cross-Validation.

    :param model_name: Name of the regression model to be used.
    """
    global_results = []

    def process_csv(csv_file, pbar):
        """
        Process a single CSV file to perform regression, log details, and compute statistics.

        :param csv_file: Path to the CSV file to be processed.
        :param pbar: Progress bar object from tqdm to update progress.
        """
        debug_logs = []
        predicted_values = []
        actual_values = []
        rmse_values = []  
        mae_values = []

        try:
            base_file_name = os.path.basename(csv_file).split('.')[0]
            debug_dir = f'./{input_last_dir_name}/{model_name}/DEBUG'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            debug_file_name = f"{debug_dir}/{script_name}_{base_file_name}_DEBUG.csv"

            start_time = time.time()

            data = pd.read_csv(csv_file)
            X = data.iloc[:, 1:]
            y = data.iloc[:, 0]

            if len(data) <= 1:
                raise ValueError("Insufficient data points in file: {}".format(csv_file))

            loo = LeaveOneOut()
            model = SVR() if model_name == 'SVR' else AdaBoostRegressor(random_state=42) if model_name == 'AdaBoost' else GradientBoostingRegressor()

            for fold_index, (train_index, test_index) in enumerate(loo.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                predicted_values.extend(y_pred)
                actual_values.extend(y_test.tolist())

                mse = mean_squared_error(y_test, y_pred)
                rmse = math.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                rmse_values.append(rmse)  
                mae_values.append(mae)

                # Log details for the fold, rounding the values to two decimal places
                debug_logs.append({
                    'Fold': fold_index,
                    'Actual': round(y_test.iloc[0], 2),  
                    'Predicted': round(y_pred[0], 2),    
                    'MSE': round(mse, 2),
                    'RMSE': round(rmse, 2),
                    'MAE': round(mae, 2)
                })

            # Calculate global RMSE and MAE
            global_rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
            global_mae = mean_absolute_error(actual_values, predicted_values)
            global_r2 = r2_score(actual_values, predicted_values) if len(actual_values) > 1 else np.nan
            std_dev_global_rmse = np.std(rmse_values)  # Oblicz odchylenie standardowe dla RMSE
            std_dev_global_mae = np.std(mae_values)

            end_time = time.time()
            execution_time = round(end_time - start_time, 2)

            # Append global results
            global_results.append({
                'File': base_file_name,
                'RMSE': round(global_rmse, 2),
                'Standard Deviation of RMSE': round(std_dev_global_rmse, 2),  # Dodane odchylenie standardowe RMSE
                'MAE': round(global_mae, 2),
                'Standard Deviation of MAE': round(std_dev_global_mae, 2),  # Dodane odchylenie standardowe MAE
                'R2': round(global_r2, 2) if not np.isnan(global_r2) else global_r2,
                'Execution Time (seconds)': execution_time
            })

            # Save debug file
            debug_df = pd.DataFrame(debug_logs)
            debug_df.insert(0, 'File Name', base_file_name)
            debug_df.to_csv(debug_file_name, index=False)
            pbar.update(1)
        except Exception as e:
            print(f"Error processing CSV file: {csv_file}. {e}")

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

        # Save global results
        global_output_df = pd.DataFrame(global_results)
        global_dir = f'./{input_last_dir_name}/{model_name}'
        if not os.path.exists(global_dir):
            os.makedirs(global_dir, exist_ok=True)
        global_output_file = f"{global_dir}/{script_name}_output.csv"
        global_output_df.to_csv(global_output_file, index=False)

# Run the models
model = ['SVR', 'AdaBoost', 'GradientBoosting']
for model_name in model:
    ML_regression(model_name)

print("\nCalculations finished\n")
