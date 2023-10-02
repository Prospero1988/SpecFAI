# Import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import os

# Clear the terminal screen
os.system('cls' if os.name == 'nt' else 'clear')

# Read the CSV file containing the list of files
print()
list_csv = input(" Enter the name of csv file with list of files: ")
file_list = pd.read_csv(list_csv)
print()
percent_train = int(input(" Enter percentage of data used to test the model (0-100): ")) / 100
print()

# Create an empty dictionary to store file names and MSE values
rmse_results = {}

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

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_train, random_state=42)
        
        # Check if there are enough data points for testing
        if len(y_test) == 0:
            raise ValueError("Insufficient data points for testing in file: {}".format(file_name))

        # Create a Random Forest model
        rf = RandomForestRegressor()

        # Train the model
        rf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf.predict(X_test)

        # Calculate the mean squared error of the model
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate the root mean squared error (RMSE)
        rmse = "{:.4f}".format(math.sqrt(mse))
        
        # Store the file name and RMSE value in the dictionary
        rmse_results[file_name] = rmse

    except Exception as e:
        print("Error processing file: {}. {}".format(file_name, str(e)))

# Print the file names with corresponding RMSE values
print()
print(" RMSE values for datasets:")
print()
print()
for file_name, rmse in rmse_results.items():

    # Extract the filename from the file path
    filename = os.path.basename(file_name)
    print(" RMSE:", rmse, "  ----->  ", "File:", filename)
print()
print()
