
# Scientific Python Scripts Collection

This repository contains several Python scripts developed to facilitate various tasks related to scientific research, including regression analysis, molecular data processing, NMR spectrum integration, and dimensionality reduction.

## Table of Contents
1. [General Description](#general-description)
2. [Script Descriptions](#script-descriptions)
   - [columns_compare_and_copy.py](#columns_compare_and_copypy)
   - [create_file_list.py](#create_file_listpy)
   - [delete_empty_rows.py](#delete_empty_rowspy)
   - [empty_records.py](#empty_recordspy)
   - [extract_first_column_from_csv.py](#extract_first_column_from_csvpy)
   - [header_and_column_insert_dir.py](#header_and_column_insert_dirpy)
   - [MOL_REP_generation.py](#mol_rep_generationpy)
   - [preparation_total.py](#preparation_totalpy)
   - [Regressor_10CV.py](#regressor_10cvpy)
   - [Regressor_BT.py](#regressor_btpy)
   - [Regressor_LOO.py](#regressor_loop_py)
   - [smiles_to_mol.py](#smiles_to_molpy)
   - [total_reduction_from_list.py](#total_reduction_from_listpy)
   - [bucket_integration_dir.py](#bucket_integration_dirpy)
3. [Installation and Usage](#installation-and-usage)
4. [Contact Information](#contact-information)

## General Description
This repository includes a collection of scripts that serve various purposes in scientific data analysis, particularly in handling CSV files, performing regression analysis, molecular structure processing, and more. These scripts are specifically designed to be run on scientific datasets such as NMR data or molecular SMILES codes.

## Script Descriptions

### columns_compare_and_copy.py
**Description**: This script compares the rows in two CSV files and copies rows from the first file where the values in the first column are present in the second file. The reference values must be stored in a single column in the second file.
- **Usage**: The script will prompt you for the file names of the input, reference, and output files.
- **Arguments**: Takes input, reference, and output file paths.
  
### create_file_list.py
**Description**: This script recursively scans a directory for files and writes a list of the file paths to a CSV file.
- **Usage**: The script will prompt you to provide a directory and an output CSV file name.
  
### delete_empty_rows.py
**Description**: This script removes rows from a CSV file that contain empty cells.
- **Usage**: The script prompts for the CSV file name, processes it, and overwrites it by removing empty rows.
  
### empty_records.py
**Description**: This script scans a CSV file for rows containing empty records and logs which rows contain them.
- **Usage**: The script will ask for the CSV file name and return a summary of the rows containing empty records.

### extract_first_column_from_csv.py
**Description**: Extracts the first column from a CSV file and saves it as a new CSV file.
- **Usage**: It prompts the user for the input and output file names.

### header_and_column_insert_dir.py
**Description**: This script inserts headers into CSV files and allows the user to add a column from one file to another.
- **Usage**: Use command-line arguments for options like directory, headers, second file, and column number.
  
### MOL_REP_generation.py
**Description**: Generates molecular representations in the form of fingerprints and descriptors from SMILES codes.
- **Usage**: It processes a CSV file with SMILES codes and saves molecular descriptors and fingerprints.

### preparation_total.py
**Description**: This script processes NMR datasets, performs interpolation, normalization, and allows column removal based on user-defined ranges. It also visualizes the data at different stages.
- **Usage**: You need to specify a directory containing CSV files for processing.

### Regressor_10CV.py
**Description**: This script performs regression analysis using 10-fold cross-validation on CSV datasets with models like SVR, AdaBoost, and Gradient Boosting.
- **Usage**: The script takes a directory path containing CSV files as input.
  
### Regressor_BT.py
**Description**: This script uses bootstrapping for regression analysis, performing multiple random resampling iterations to improve accuracy.
- **Usage**: The script will ask for a directory of CSV files.
  
### Regressor_LOO.py
**Description**: Implements leave-one-out cross-validation for regression tasks. It logs detailed metrics for each fold and computes global statistics.
- **Usage**: You can pass the directory of CSV files as an argument.
  
### smiles_to_mol.py
**Description**: This script converts SMILES strings to 3D molecular representations and saves them in `.mol` format.
- **Usage**: Provide the path to a CSV containing SMILES strings.

### total_reduction_from_list.py
**Description**: A script to perform dimensionality reduction using methods such as PCA, ICA, NMF, and UMAP. The user provides a list of files and the target dimensionality.
- **Usage**: You will be prompted for dimensionality, decimal rounding, and UMAP parameters.
  
### bucket_integration_dir.py
**Description**: This script processes NMR datasets by dividing the data into buckets (ranges) and summing values within each range.
- **Usage**: You will be asked for the number of ranges and the directory containing the CSV files.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run each script by following the specific usage instructions mentioned above.

## Contact Information
For further assistance or inquiries, please contact:
- **Author**: Arkadiusz Leniak
- **Email**: arek.kein@gmail.com
