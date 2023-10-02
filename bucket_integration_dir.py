import csv
import os
import sys

def print_help():
    print()
    print("This script performs bucket integration on NMR (Nuclear Magnetic Resonance) data sets stored in CSV files. It divides the data into a specified number of ranges and calculates the sum of values within each range. The processed data is then saved to new CSV files.")
    print()
    print("The script provides a command-line interface that allows you to specify the directory containing the CSV files and the number of ranges for bucket integration. It processes all the CSV files in the specified directory.")
    print()
    print("Here's how the script works:")
    print()
    print("1. You are prompted to enter the name of the directory containing the CSV files.")
    print("2. You are prompted to enter the number of ranges for bucket integration.")
    print("3. The script loads and processes each CSV file in the directory.")
    print("4. For each file, it divides the number of columns into ranges and calculates the sum of values within each range.")
    print("5. The processed data is saved to a new CSV file with the original file name appended with '_bucketed'.")
    print("6. The script displays the name of the created file.")
    print("7. Once all files in the directory have been processed, the script displays a completion message.")
    print()
    print("Usage: python bucket_integration_dir.py [OPTIONS]")
    print()
    print("Options:")
    print("  --help\t\tDisplay this help message and exit.")
    print("  --dir DIRECTORY\tSpecify the directory containing the CSV files.")
    print("  --ranges N\t\tSpecify the number of ranges for bucket integration.")
    print()

if len(sys.argv) > 1 and sys.argv[1] == "--help":
    print_help()
    sys.exit(0)

def load_data_from_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    return data


def create_bucketed_data(data, n):
    num_columns = len(data[0])
    if num_columns % n != 0:
        print("The number of columns is not divisible by the number of ranges.")
        exit()

    columns_per_range = num_columns // n

    new_data = []
    for row in data:
        new_row = []
        for i in range(n):
            range_values = [float(value) for value in row[i * columns_per_range:(i + 1) * columns_per_range]]
            sum_points = sum(value for value in range_values if value != 0)
            new_row.append(sum_points)
        new_data.append(new_row)

    return new_data


def save_data_to_csv(data, filename):
    
    file_name, file_extension = os.path.splitext(filename)
    new_filename = os.path.join(file_name + "_bucketed" + file_extension)
    with open(new_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    return new_filename


# General message
print()
print(" |--------------------------------------|")
print(" |----------BUCKET INTEGRATION----------|")
print(" |----for NMR data sets in CSV files----|")
print(" |--------------------------------------|")
print(" |-by Arkadiusz Leniak -----------------|")
print(" |-arek.kein@gmail.com -----------------|")
print(" |--------------------------------------|")
print()

# Step 1: Provide the name of the directory
directory = input("Enter the name of the directory containing the CSV files: ")
print()
n = int(input("Enter the number of ranges: "))
print()

# Step 2: Load and process data from CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        print("Processing file:", filename)
        data = load_data_from_csv(filepath)

        # Divide the number of columns into ranges and create bucketed data
        bucketed_data = create_bucketed_data(data, n)

        # Step 4: Create a new file with range values
        new_filename = save_data_to_csv(bucketed_data, filename)

        print("Created a new file:", new_filename)
        print()

print("Processing completed for all files in the directory.")

