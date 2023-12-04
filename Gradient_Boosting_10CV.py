import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import math
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

# Clear the terminal screen
# os.system('cls' if os.name == 'nt' else 'clear')

# Pobierz nazwę skryptu bez rozszerzenia
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Read the CSV file containing the list of files
print()
list_csv_input = input("Enter the names of CSV files with lists of files (comma-separated): ")
list_csv = list_csv_input.split(',')

# Start measuring the execution time
total_execution_start_time = time.time()

# Function to process a single CSV file
def process_csv(csv_file):
    try:
        # Read the current CSV file
        file_list = pd.read_csv(csv_file.strip())
        output_file_name = f"{script_name}_{os.path.splitext(csv_file.strip())[0].replace('.', '_')}_output.csv"  # Change the output file name

        # Create an empty list to store RMSE values and standard deviations for each fold
        rmse_values = []
        std_dev_values = []

        # Start measuring the execution time for the current CSV file
        start_time = time.time()

        # Iterate over the files
        for file_name in file_list['File Path']:
            try:
                # Read the current file
                data = pd.read_csv(file_name)

                # Extract the features and target variable
                X = data.iloc[:, 1:]  # Features (assuming they start from the second column)
                y = data.iloc[:, 0]  # Target variable (first column)

                # Check if there are enough data points for training
                if len(data) <= 1:
                    raise ValueError("Insufficient data points in file: {}".format(file_name))

                # Create a Gradient Boosting Regressor model
                gb_regressor = GradientBoostingRegressor()

                # Perform 10-fold cross-validation
                kf = KFold(n_splits=10, shuffle=True, random_state=42)
                fold_rmse_values = []

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Train the model
                    gb_regressor.fit(X_train, y_train)

                    # Make predictions on the test set
                    y_pred = gb_regressor.predict(X_test)

                    # Calculate the mean squared error of the model
                    mse = mean_squared_error(y_test, y_pred)

                    # Calculate the root mean squared error (RMSE)
                    rmse = math.sqrt(mse)

                    # Store RMSE value for this fold
                    fold_rmse_values.append(rmse)

                # Calculate Averaged RMSE and Standard Deviation for this file
                averaged_rmse = np.mean(fold_rmse_values)
                std_dev_rmse = np.std(fold_rmse_values)

                # Store the averaged RMSE and Standard Deviation values in the lists
                rmse_values.append(averaged_rmse)
                std_dev_values.append(std_dev_rmse)

            except Exception as e:
                print("Error processing file: {}. {}".format(file_name, str(e)))

        # Stop measuring the execution time for the current CSV file
        end_time = time.time()

        # Calculate the total execution time for the current CSV file
        execution_time = end_time - start_time

        # Print the file names with corresponding Averaged RMSE values and standard deviations
        print()
        print(" Averaged RMSE values with Standard Deviations for datasets in {}: ".format(csv_file.strip()))
        print()
        for file_name, rmse, std_dev in zip(file_list['File Path'], rmse_values, std_dev_values):
            # Extract the filename from the file path
            filename = os.path.basename(file_name)
            print("Averaged RMSE: {:.5f}  ----->  Standard Deviation of RMSE: {:.5f} ----->  File: {}".format(rmse, std_dev, filename))
        print("Execution time for {}: {:.2f} seconds".format(csv_file.strip(), execution_time))

        # Save the output to a CSV file
        output_df = pd.DataFrame({'File': file_list['File Path'], 'Averaged RMSE': rmse_values, 'Standard Deviation of RMSE': std_dev_values})
        output_df.to_csv(output_file_name, index=False)  # Save results to a separate output file

        # Save execution time to a separate file
        execution_times_df = pd.DataFrame({'list_csv': [csv_file.strip()], 'execution_time': [execution_time]})
        execution_times_df.to_csv(f"{script_name}_{os.path.splitext(csv_file.strip())[0].replace('.', '_')}_execution_times.csv", index=False)

    except Exception as e:
        print("Error processing CSV file: {}. {}".format(csv_file.strip(), str(e)))

# Use multiprocessing to process CSV files concurrently
with Pool(processes=4) as pool:
    pool.map(process_csv, list_csv)

# Stop measuring the total execution time
total_execution_end_time = time.time()

# Calculate and print the total execution time for all CSV files
total_execution_time = total_execution_end_time - total_execution_start_time
print()
print("Total execution time for all CSV files: {:.2f} seconds".format(total_execution_time))
print()
