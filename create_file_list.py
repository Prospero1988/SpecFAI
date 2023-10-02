import os
import csv

def create_file_list_csv(directory, output_file):
    file_list = []
    for entry in os.scandir(directory):
        if entry.is_file():
            file_path = entry.path
            file_list.append(file_path)
        elif entry.is_dir():
            subdirectory = entry.path
            file_list.extend(get_files_in_directory(subdirectory))

    return file_list

def get_files_in_directory(directory):
    file_list = []
    for entry in os.scandir(directory):
        if entry.is_file():
            file_path = entry.path
            file_list.append(file_path)
        elif entry.is_dir():
            subdirectory = entry.path
            file_list.extend(get_files_in_directory(subdirectory))

    return file_list

def write_file_list_to_csv(file_list, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File Path'])
        writer.writerows([[file_path] for file_path in file_list])

    print(f"CSV file '{output_file}' created successfully with the list of files.")

directory = input("Enter the directory path: ")
output_file = input("Enter the output file name: ")

file_list = create_file_list_csv(directory, output_file)
write_file_list_to_csv(file_list, output_file)
