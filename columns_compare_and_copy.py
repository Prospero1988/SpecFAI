# -*- coding: cp1250 -*-

# Skrypt ktory ktory kopiuje wiersze z pierwszego pliku csv do nowego pliku.
# Kopiowane sa tylko te wiersze, w których pierwsz kolumna zawiera wartości
# z drugiego pliku wsadowego csv. Wartosci w drugim pliku muszą byc w jednej 
# kolumnie, w odrebnych wierszach.

import csv

def compare_and_remove(input_file, reference_file, output_file):
    molecules = set()

    # Reading MOLECULE column from the second file and storing them in a set
    with open(reference_file, 'r') as ref_csv:
        ref_reader = csv.reader(ref_csv)
        next(ref_reader)  # Skip the header
        for row in ref_reader:
            molecule = row[0]
            molecules.add(molecule)

    # Comparing MOLECULE column from the first file and writing the appropriate rows to a new file
    with open(input_file, 'r') as input_csv, open(output_file, 'w', newline='') as output_csv:
        input_reader = csv.reader(input_csv)
        output_writer = csv.writer(output_csv)
        header = next(input_reader)
        output_writer.writerow(header)

        for row in input_reader:
            molecule = row[0]
            if molecule in molecules:
                output_writer.writerow(row)

    print("Removed the appropriate rows and saved the file:", output_file)

# Allowing the user to enter the file names
input_file = input("Enter the name of the input file: ")
reference_file = input("Enter the name of the reference file: ")
output_file = input("Enter the name of the output file: ")

# Example usage
compare_and_remove(input_file, reference_file, output_file)
