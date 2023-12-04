import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import math
import os
import time
from sklearn.model_selection import LeaveOneOut  # Change import
import numpy as np
import multiprocessing

# Funkcja wykonująca operacje na pojedynczym pliku
def process_file(file_name):
    try:
        # Read the current file
        data = pd.read_csv(file_name)

        # Extract the features and target variable
        X = data.iloc[:, 1:]  # Features (assuming they start from the second column)
        y = data.iloc[:, 0]  # Target variable (first column)

        # Check if there are enough data points for training
        if len(data) <= 1:
            raise ValueError("Insufficient data points in file: {}".format(file_name))

        # Create an AdaBoost Regressor model
        gb_regressor = GradientBoostingRegressor()

        # Perform leave-one-out splitting
        loo = LeaveOneOut()
        rmse_values = []

        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            gb_regressor.fit(X_train, y_train)
            y_pred = gb_regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse_values.append(math.sqrt(mse))

        # Calculate the root mean squared error (RMSE) and standard deviation
        rmse = np.mean(rmse_values)
        std_dev_rmse = np.std(rmse_values)

        return file_name, rmse, std_dev_rmse

    except Exception as e:
        print("Error processing file: {}. {}".format(file_name, str(e)))
        return None

# Clear the terminal screen
# os.system('cls' if os.name == 'nt' else 'clear')

# Read the CSV file containing the list of files
list_csv_input = input("Enter the names of CSV files with lists of files (comma-separated): ")
list_csv_files = [file.strip() for file in list_csv_input.split(',')]

# Get the script name without the extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Create a directory for output files if it doesn't exist
output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)

# Create a list to store execution times
execution_times = []

# Iterate over the CSV files with lists of files
for list_csv in list_csv_files:
    # Remove whitespaces from the filename and add '_output.csv' as the output filename
    output_filename = os.path.join(output_dir, f"{script_name}_{list_csv}_output.csv")

    # Create empty lists to store RMSE and standard deviation values for each file
    rmse_values = []
    std_dev_values = []

    # Start measuring the execution time for the current list
    start_time = time.time()

    try:
        # Read the CSV file containing the list of files
        file_list = pd.read_csv(list_csv)

        # Parallel processing using multiprocessing
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(process_file, file_list['File Path'])

        # Stop measuring the execution time for the current list
        end_time = time.time()

        # Calculate the total execution time for the current list
        execution_time = end_time - start_time

        # Append execution time to the list
        execution_times.append({'list_csv': list_csv, 'execution_time': execution_time})

        # Print the file names with corresponding RMSE values and standard deviations for the current list
        print()
        print(f"RMSE values with Standard Deviations for datasets in {list_csv}:")
        print()
        for file_name, rmse, std_dev in results:
            if file_name is not None:
                # Extract the filename from the file path
                filename = os.path.basename(file_name)
                print("RMSE: {:.5f}  ----->  Standard Deviation of RMSE: {:.5f} ----->  File: {}".format(rmse, std_dev, filename))
                rmse_values.append(rmse)
                std_dev_values.append(std_dev)
            else:
                print("Error processing one or more files in the list.")

        # Save the results to a CSV file
        result_df = pd.DataFrame({
            'File': file_list['File Path'],
            'RMSE': rmse_values,
            'Standard Deviation of RMSE': std_dev_values
        })
        result_df.to_csv(output_filename, index=False)

        print("Total execution time for {}: {:.2f} seconds".format(list_csv, execution_time))
        print()

    except Exception as e:
        print("Error processing CSV file list: {}. {}".format(list_csv, str(e)))

# Create a DataFrame from the list of execution times
execution_times_df = pd.DataFrame(execution_times)

# Save the execution times DataFrame to a CSV file
execution_times_df.to_csv(os.path.join(output_dir, f"{script_name}_execution_times.csv"), index=False)

# End of the script
