# Skrypt kopiuje pierwsza kolumne z jednego pliku csv i 
# zapisuje ja jako nowy plik .csv

import csv

# Input the path to the original CSV file
input_file = input("Enter the path to the input CSV file: ")

# Input the path to the new CSV file
output_file = input("Enter the path to the output CSV file: ")

# Open the original CSV file for reading
with open(input_file, 'r') as file:
    # Create a CSV reader
    reader = csv.reader(file)
    
    # Read the first column from each row
    first_column = [row[0] for row in reader]

# Open the new CSV file for writing
with open(output_file, 'w', newline='') as file:
    # Create a CSV writer
    writer = csv.writer(file)
    
    # Write only the first column to the new CSV file
    for value in first_column:
        writer.writerow([value])

print("Copied the first column to the new CSV file.")
