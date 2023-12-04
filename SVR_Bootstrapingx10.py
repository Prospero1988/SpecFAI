import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import math
import os
import time
from sklearn.model_selection import train_test_split
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

        # Create an SVR model
        svr_regressor = SVR()

        # Perform fivefold bootstrapping
        n_bootstraps = 10
        rmse_values = []

        for _ in range(n_bootstraps):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(1, 1000))
            svr_regressor.fit(X_train, y_train)
            y_pred = svr_regressor.predict(X_test)
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
print()
list_csv_input = input("Enter the name of the CSV file with a list of files: ")
list_csv = list_csv_input.split(',')

# Get the script name without the extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Create a directory for output files if it doesn't exist
output_dir = "output_files"
os.makedirs(output_dir, exist_ok=True)

# Create a list to store execution times
execution_times = []

# Iterate over the list of CSV files
for current_list_csv in list_csv:
    # Remove whitespaces from the filename and add '_output.csv' as the output filename
    output_filename = os.path.join(output_dir, f"{script_name}_{current_list_csv.strip()}_output.csv")

    # Create empty lists to store RMSE and standard deviation values for each file
    rmse_values = []
    std_dev_values = []

    # Start measuring the execution time for the current list
    start_time = time.time()

    try:
        # Read the CSV file containing the list of files
        file_list = pd.read_csv(current_list_csv)

        # Parallel processing using multiprocessing
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(process_file, file_list['File Path'])

        # Stop measuring the execution time for the current list
        end_time = time.time()

        # Calculate the total execution time for the current list
        execution_time = end_time - start_time

        # Append execution time to the list
        execution_times.append({'List': current_list_csv, 'Execution Time (seconds)': execution_time})

        # Print the file names with corresponding RMSE values and standard deviations for the current list
        print()
        print(f"RMSE values with Standard Deviations for datasets in {current_list_csv}:")
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

        print("Total execution time for {}: {:.2f} seconds".format(current_list_csv, execution_time))
        print()

    except Exception as e:
        print("Error processing CSV file list: {}. {}".format(current_list_csv, str(e)))

# Create a DataFrame from the list of execution times
execution_times_df = pd.DataFrame(execution_times)

# Save the execution times DataFrame to a CSV file
execution_times_df.to_csv(os.path.join(output_dir, f"{script_name}_execution_times.csv"), index=False)

# End of the script
