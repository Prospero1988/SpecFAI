
# Scientific Python Scripts Collection

This repository contains a series of Python scripts that were instrumental in the data analysis and processing pipeline of the study conducted by Arkadiusz Leniak et al., titled "NMR Spectral Data Analysis and Machine Learning for Molecular Property Prediction". These scripts enabled various computational tasks, such as regression analysis, molecular data processing, integration of nuclear magnetic resonance (NMR) spectra, and dimensionality reduction, which contributed to the accurate prediction and simulation of molecular properties based on NMR data.

## Summary of the Publication
In this study, we aimed to improve the prediction accuracy of molecular properties by integrating experimental NMR data with machine learning models. The repository contains key Python scripts that automated data pre-processing, regression model training, evaluation through cross-validation techniques, and the post-processing of results. These scripts are vital for handling large volumes of CSV-based spectral data and generating molecular representations, which are then used to train predictive models.

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
This repository includes a comprehensive collection of scripts designed for scientific data analysis, particularly focusing on NMR data sets and molecular property prediction through machine learning techniques. Each script serves a specific role, from preprocessing CSV files to generating molecular fingerprints, applying machine learning models, and performing dimensionality reduction on large datasets.

## Script Descriptions

### columns_compare_and_copy.py
**Purpose**: This script cross-references two CSV files and copies rows from the first file where values in the first column match those in the second CSV file. It's useful for filtering data based on specific criteria.
- **Input**: Two CSV files, where the first column of the second file is used as a reference.
- **Output**: A new CSV file with the filtered rows from the first file.

### create_file_list.py
**Purpose**: Recursively scans a directory and compiles a list of all files into a CSV. This script is especially useful when dealing with large datasets that are split into multiple files.
- **Input**: Directory path.
- **Output**: A CSV containing the file paths of all detected files.

### delete_empty_rows.py
**Purpose**: This script checks each row in a CSV file and removes any rows that contain empty cells. This is essential for data cleaning, especially before feeding data into machine learning algorithms.
- **Input**: A CSV file.
- **Output**: The same CSV file, cleaned of rows with empty cells.

### empty_records.py
**Purpose**: Similar to `delete_empty_rows.py`, but this script specifically logs the number of empty records for further analysis rather than deleting them immediately.
- **Input**: A CSV file.
- **Output**: A report that indicates which rows contain empty records.

### extract_first_column_from_csv.py
**Purpose**: Extracts the first column from a CSV file, which is useful in isolating labels or key identifiers from large datasets.
- **Input**: A CSV file.
- **Output**: A new CSV file with only the first column of the original file.

### header_and_column_insert_dir.py
**Purpose**: This script offers multiple functionalities, including adding custom headers to CSV files and copying a specific column from one file into others. It's particularly helpful for organizing and merging datasets.
- **Input**: Directory path, column number, and file paths for header customization and column copying.
- **Output**: Modified CSV files with added headers or new columns.

### MOL_REP_generation.py
**Purpose**: Converts SMILES strings to molecular fingerprints and descriptors, which are then used in machine learning models for molecular property prediction. The script supports multiple types of fingerprints (RDKit, ECFP4, MACCS, etc.).
- **Input**: A CSV file with SMILES codes.
- **Output**: Fingerprint and descriptor CSV files.

### preparation_total.py
**Purpose**: This comprehensive script processes NMR datasets by interpolating, normalizing, and removing specific columns. It also includes visualization capabilities, which allow users to plot the dataset at various stages of processing.
- **Input**: Directory of CSV files.
- **Output**: Cleaned and processed CSV files, with interactive plots of the data.

### Regressor_10CV.py
**Purpose**: Performs 10-fold cross-validation regression using models such as SVR, AdaBoost, and Gradient Boosting. This script is a key part of evaluating machine learning models for molecular property prediction.
- **Input**: Directory of CSV files with features and labels.
- **Output**: CSV files containing model performance metrics for each fold.

### Regressor_BT.py
**Purpose**: Employs a bootstrapping technique to resample the data and perform regression analysis. This method enhances the robustness of model evaluation by creating multiple resampled datasets.
- **Input**: Directory of CSV files.
- **Output**: Detailed CSV files logging results for each bootstrapped sample.

### Regressor_LOO.py
**Purpose**: Implements leave-one-out cross-validation for regression tasks. This method ensures that every single data point is used as a test point, while the rest are used for training.
- **Input**: Directory of CSV files.
- **Output**: Detailed CSV files containing metrics for each leave-one-out iteration.

### smiles_to_mol.py
**Purpose**: Converts SMILES strings into 3D molecular representations and saves them in `.mol` format. This is critical for downstream molecular modeling and simulation tasks.
- **Input**: CSV file with SMILES strings and molecule names.
- **Output**: `.mol` files representing 3D molecular structures.

### total_reduction_from_list.py
**Purpose**: A powerful dimensionality reduction tool that supports methods such as PCA, ICA, NMF, and UMAP. It is essential for reducing high-dimensional datasets to more manageable forms before analysis.
- **Input**: A list of CSV files and dimensionality reduction method.
- **Output**: Reduced datasets in CSV format.

### bucket_integration_dir.py
**Purpose**: Processes NMR datasets by dividing them into buckets (ranges) and calculating the sum of values in each bucket. This is essential for reducing high-resolution NMR data into more interpretable features.
- **Input**: Directory of CSV files and the number of desired ranges (buckets).
- **Output**: New CSV files with bucketed data.

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
