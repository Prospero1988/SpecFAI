import os
import csv
import sys
import pandas as pd

def print_help():
    print()
    print("This script is designed for manipulating CSV files containing NMR data sets.")
    print("Usage: python script.py [options]")
    print()
    print("Options:")
    print("  -h, --help            show this help message and exit")
    print("  -d DIRECTORY_PATH, --directory DIRECTORY_PATH")
    print("                        specify the directory path containing the CSV files")
    print("  -c CUSTOM_HEADERS, --headers CUSTOM_HEADERS")
    print("                        specify the column headers name")
    print("  -s SECOND_FILE_PATH, --second-file SECOND_FILE_PATH")
    print("                        specify the path for the second CSV file")
    print("  -n COLUMN_NUMBER, --column-number COLUMN_NUMBER")
    print("                        specify the column number to copy")
    print()
    print("Description:")
    print("  This script performs the following operations on CSV files in the specified directory:")
    print()
    print("  1. Asks the user if they want to add headers to the data (as C1, C2, C3, etc.).")
    print("  2. Asks the user if they want to copy a defined column from another CSV file as the first column to the current one.")
    print()
    print("Example usage:")
    print("python header_and_column_insert_dir.py -d /path/to/directory -c 'C' -s /path/to/second_file.csv -n 3")
    print()

if len(sys.argv) > 1 and sys.argv[1] == "--help":
    print_help()
    sys.exit(0)

def add_new_line(file_name):
    file_name = os.path.join(directory_path, file_name)
    with open(file_name, 'r', newline='') as new_line_input:
        reader = csv.reader(new_line_input)
        dane = list(reader)

    df = pd.read_csv(file_name)
    num_columns_df = len(df.columns)
    num_rows_df = len(df)
    num_rows_df = num_rows_df + 1

    print()
    print(f"----> File contains {num_columns_df} columns and {num_rows_df} rows.")

    number_of_columns = len(dane[0])

    zero_line = [custom_headers + str(i + 1) for i in range(number_of_columns)]
    dane.insert(0, zero_line)
    
    with open(file_name, 'w', newline='') as new_line_output:
        writer = csv.writer(new_line_output)
        writer.writerows(dane)
    
    print()
    print(f"----> {custom_headers}1, {custom_headers}2, {custom_headers}3, etc. column headers has been")
    print(f"      added to the {file_name} CSV file.")

    df = pd.read_csv(file_name)
    num_columns_df = len(df.columns)
    num_rows_df = None
    num_rows_df = len(df)
    num_rows_df = num_rows_df + 1

    print()
    print(f"----> Now the file contains {num_columns_df} columns and {num_rows_df} rows.")

# Generate message
print()
print()
print(" |--------------------------------------|")
print(" |--------CSV Manipulation SCRIPT-------|")
print(" |----for NMR data sets in CSV files----|")
print(" |-------------------------------------------------------------------------------|")
print(" | 1.  Script asks if user wants to add headers to the data (as C1, C2, C3, etc.)|")
print(" | 2.  Script asks if user wants to copy defined column from another CSV file    |")
print(" |     as the first column to the current one                                    |")
print(" |-------------------------------------------------------------------------------|")
print()

# Get the directory path
directory_path = input("Enter the directory path containing the CSV files: ")
print()
# Get the list of CSV files in the directory
csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

custom_headers = input("Specify headers name: ")
# Ask the user to input the path for the second CSV file
print()
second_file_path = input("Enter the path for the second CSV file: ")
second_file_path = second_file_path
print()
column_number = int(input("Enter the column number to copy: ")) - 1

# Process each CSV file in the directory
for file_name in csv_files:
    print()
    print("-------------------------------------------------------------------------------------")
    print(f"Processing file: {file_name}")
    print("-------------------------------------------------------------------------------------")

    #Add new line at the begining of file

    add_new_line(file_name)

    # Add headers to the final CSV file
    final_file_path = os.path.join(directory_path, file_name)

    # Appending first column from other CSV files

    # Read the first CSV file into a DataFrame
    df1 = pd.read_csv(final_file_path)


    # Read the second CSV file into a DataFrame
    df2 = pd.read_csv(second_file_path)

    # Get the number of columns in the DataFrame
    num_columns = len(df2.columns)

    # Print the number of columns
    print()

    while True:
        # Validate the column number input
        column_number2 = column_number
        if column_number2 < 0 or column_number >= len(df2.columns):
            print()
            print("Invalid column number!")
        else:
            break

    # Extract the column you want to copy from the second file
    column_name = df2.columns[column_number2]
    column_to_copy = df2.iloc[:, column_number2]

    # Insert the copied column as the first column in the first file
    df1.insert(0, column_name, column_to_copy)

    # Save the updated DataFrame to the first CSV file
    df1.to_csv(final_file_path, index=False)
    column_number2 = column_number2 + 1
    print(f"----> Column {column_number2} from {second_file_path} was copied successfuly")
    print(f"      to the {file_name} and saved as a first column.")
    print()

    with open(final_file_path, 'r') as input_file:
        reader = csv.reader(input_file)
        data = []
        for row in reader:
            converted_row = []
            for value in row:
                try:
                    converted_row.append(float(value))
                except ValueError:
                    converted_row.append(value)
            data.append(converted_row)

        df = pd.read_csv(final_file_path)
        num_columns_df = len(df.columns)
        num_rows_df = None
        num_rows_df = len(df)
        num_rows_df += 1
    column_number2 = None
    print(f"----> File contains {num_columns_df} columns and {num_rows_df} rows.")
    print()
    print()

print("------------------------------------------------------------")
KONIEC = input("Press ENTER to end script: ")
sys.exit(1)