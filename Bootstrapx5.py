import sys
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import math
import os
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import time

# Clear the terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Define the output file name
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# Read the CSV file containing the list of files
print()
list_csv = input(" Enter the name of csv file with list of files: ")
file_list = pd.read_csv(list_csv)
print()
num_bootstrap_samples = int(input(" Enter the number of bootstrap samples: "))
percent_train = int(input(" Enter percentage of data used to test the model (0-100): ")) / 100
print()

# Define the output file name
csv_file_name = os.path.splitext(os.path.basename(list_csv))[0]
output_file_name = f"{script_name}_{csv_file_name}.txt"

# Redirect output to the file
original_stdout = sys.stdout
sys.stdout = open(output_file_name, 'w')

# Create an empty dictionary to store file names and MSE values
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

        # Calculate the number of data points for bootstrapping
        num_data_points = len(X)
        
        # Initialize a list to store the bootstrap RMSE values
        bootstrap_rmse = []

        # Perform bootstrapping
        for _ in range(num_bootstrap_samples):
            # Create a bootstrap sample
            X_bootstrap, y_bootstrap = resample(X, y, n_samples=num_data_points, replace=True, random_state=42)
            
            # Split the bootstrap sample into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_bootstrap, y_bootstrap, test_size=percent_train, random_state=42)
            
            # Create an SVR model
            svr = SVR()

            # Train the model
            svr.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = svr.predict(X_test)

            # Calculate the mean squared error of the model
            mse = mean_squared_error(y_test, y_pred)
            
            # Calculate the root mean squared error (RMSE)
            rmse = math.sqrt(mse)
            
            # Append the RMSE to the list
            bootstrap_rmse.append(rmse)

        # Calculate the average RMSE of the bootstrap samples
        avg_rmse = sum(bootstrap_rmse) / num_bootstrap_samples
        
        # Store the file name and average RMSE value in the dictionary
        rmse_results[file_name] = avg_rmse

    except Exception as e:
        print("Error processing file: {}. {}".format(file_name, str(e)))

# Stop measuring the execution time
end_time = time.time()

# Calculatethe total execution time
execution_time = end_time - start_time

# Print the file names with corresponding average RMSE values
print()
print(" Average RMSE values for datasets using Bootstrapping:")
print()
print()
for file_name, avg_rmse in rmse_results.items():

    # Extract the filename from the file path
    filename = os.path.basename(file_name)
    print("Avg RMSE: {:.5f}  ----->  File: {}".format(avg_rmse, filename))
print()
print("Total execution time: {:.2f} seconds".format(execution_time))
print()

# Restore the original stdout
sys.stdout = original_stdout

print("Output saved to file:", output_file_name)
