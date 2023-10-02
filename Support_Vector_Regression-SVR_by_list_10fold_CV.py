import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import math
import os
import time

# Clear the terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Read the CSV file containing the list of files
print()
list_csv = input("Enter the name of the CSV file with a list of files: ")
file_list = pd.read_csv(list_csv)
print()

# Create an empty dictionary to store file names and RMSE values
rmse_results = {}

# Start measuring the execution time
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

        # Create an SVR model
        svr = SVR()

        # Perform tenfold cross-validation
        scores = cross_val_score(svr, X, y, cv=10, scoring='neg_mean_squared_error')

        # Calculate the root mean squared error (RMSE)
        rmse = math.sqrt(-scores.mean())

        # Store the file name and RMSE value in the dictionary
        rmse_results[file_name] = rmse

    except Exception as e:
        print("Error processing file: {}. {}".format(file_name, str(e)))

# Stop measuring the execution time
end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time

# Print the file names with corresponding RMSE values
print()
print("RMSE values for datasets:")
print()
for file_name, rmse in rmse_results.items():
    # Extract the filename from the file path
    filename = os.path.basename(file_name)
    print("RMSE: {:.10f}  ----->  File: {}".format(rmse, filename))
print()
print("Total execution time: {:.2f} seconds".format(execution_time))
print()
