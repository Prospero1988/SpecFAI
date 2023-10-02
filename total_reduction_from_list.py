import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection
#from MulticoreTSNE import MulticoreTSNE
import umap
import sys
import subprocess
import time

def print_help():
    print()
    print("Script Usage:")
    print("-------------")
    print("Run the script using the command: python script_name.py")
    print()
    print("If you need additional information about the script, type 'help' when prompted.")
    print("This will display the script's description and author information.")
    print()
    print("Follow the prompts to provide the necessary inputs and parameters for the script to execute each step.")
    print("The script will guide you through the process and display relevant information and outputs.")
    print()
    print("At the end of the script, you will be asked if you want to delete any temporary files.")
    print()
    print("Note: Make sure to have the required libraries installed before running the script")
    print("(matplotlib, numpy, pandas, and scipy).")
    print()
    print("Description:")
    print("-------------")
    print("This script performs dimensionality reduction on a list of input CSV files.")
    print("It supports several reduction methods, including PCA, ICA, NMF, Random Projection,")
    print("MulticoreTSNE, and UMAP. The script allows you to specify the target dimensionality")
    print("and number of decimals for value rounding.")
    print()
    print("The script reads each input file, applies the specified reduction method, and saves")
    print("the reduced data to separate output files. The output filenames include the reduction")
    print("method and the target dimensionality.")
    print()
    print("To run the script, provide the following inputs:")
    print("- Target dimensionality: The desired dimensionality for the reduced data.")
    print("- Path to the input file list: A CSV file containing a list of input file paths.")
    print("- Target number of decimals: The number of decimals to round the values in the data.")
    print()
    print("If you choose UMAP reduction, the script will prompt you for additional parameters:")
    print("- n_neighbors: The number of neighbors to consider during UMAP reduction. (default: 50)")
    print("- min_dist: The minimum distance between points in the UMAP embedding. (default: 0.5)")
    print()
    print("During the execution, the script will display progress messages and save the reduced")
    print("data as separate CSV files. At the end, you will be asked if you want to delete any")
    print("temporary files generated during the process.")
    print()

if len(sys.argv) > 1 and sys.argv[1] == "--help":
    print_help()
    sys.exit(0)

# Start measuring the execution time
start_time = time.time()

def reduce_csv_pca(filepath, n_components, output_filepath):
    data = pd.read_csv(filepath, header=None)
    rounded_data = data.round(decimals)

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(rounded_data)

    # Save reduced data to .csv file
    pd.DataFrame(reduced_data).to_csv(output_filepath, header=None, index=None)

    print()
    print(f">>> PCA reduction completed. Data saved as: {output_filepath}")
    print()

def reduce_csv_ica(filepath, n_components, output_filepath):
    data = pd.read_csv(filepath, header=None)
    rounded_data = data.round(decimals)

    # Create the ICA model with n_components
    ica = FastICA(n_components, whiten='arbitrary-variance', max_iter=10000, tol=0.001)  # Set whiten to True
    reduced_data = ica.fit_transform(rounded_data)

    # Save reduced data to .csv file
    pd.DataFrame(reduced_data).to_csv(output_filepath, header=None, index=None)

    print()
    print(f">>> ICA reduction completed. Data saved as: {output_filepath}")
    print()

def reduce_csv_nmf(filepath, n_components, output_filepath):
    data = pd.read_csv(filepath, header=None)
    rounded_data = data.round(decimals)

    # Przekształć wartości ujemne na 0 i zastosuj logarytmowanie

    #Dane nieznormalizowane
    #rounded_data[rounded_data < 0] = 0
    #rounded_data = np.log1p(rounded_data + 1e-7)

    #nmf = NMF(n_components=n_components, max_iter=2000, l1_ratio=0.1, solver='mu', beta_loss='itakura-saito')

    #dane znormalizowane
    nmf = NMF(n_components=n_components, max_iter=2000)

    reduced_data = nmf.fit_transform(rounded_data)

    # Save reduced data to .csv file
    pd.DataFrame(reduced_data).to_csv(output_filepath, header=None, index=None)

    print()
    print(f">>> NMF reduction completed. Data saved as: {output_filepath}")
    print()


def reduce_csv_random_projection(filepath, n_components, output_filepath):
    data = pd.read_csv(filepath, header=None)
    rounded_data = data.round(decimals)

    rp = GaussianRandomProjection(n_components=n_components)
    reduced_data = rp.fit_transform(rounded_data)

    # Save reduced data to .csv file
    pd.DataFrame(reduced_data).to_csv(output_filepath, header=None, index=None)

    print()
    print(f">>> Random Projections reduction completed. Data saved as: {output_filepath}")
    print()

#def reduce_csv_mctsne(filepath, n_components, output_filepath):
    #data = pd.read_csv(filepath, header=None)
    #rounded_data = data.round(decimals)

    #mctsne = MulticoreTSNE(n_components=n_components, n_jobs=-1)
    #reduced_data = mctsne.fit_transform(rounded_data)

    # Save reduced data to .csv file
    #pd.DataFrame(reduced_data).to_csv(output_filepath, header=None, index=None)

    #print()
    #print(f">>> MulticoreTSNE reduction completed. Data saved as: {output_filepath}")
    #print()

def reduce_csv_umap(filepath, n_components, output_filepath):
    data = pd.read_csv(filepath, header=None)
    rounded_data = data.round(decimals)

    umap_reducer = umap.UMAP(n_neighbors=n_neighbors_umap, min_dist=min_dist, n_components=n_components)
    reduced_data = umap_reducer.fit_transform(rounded_data)

    # Save reduced data to .csv file
    pd.DataFrame(reduced_data).to_csv(output_filepath, header=None, index=None)

    print()
    print(f">>> UMAP reduction completed. Data saved as: {output_filepath}")
    print()

def process_file_list(filepath_list, n_components, decimals):
    reduction_methods = ["PCA", "ICA", "NMF", "RandomProjection", "MulticoreTSNE", "UMAP"]

    for filepath in filepath_list:
        file_basename = os.path.basename(filepath)
        file_name, _ = os.path.splitext(file_basename)

        for reduction_method in reduction_methods:
            output_filepath = f"{file_name}_{reduction_method}_dim{n_components}.csv"

            if reduction_method == "PCA":
                reduce_csv_pca(filepath, n_components, output_filepath)
            elif reduction_method == "ICA":
                reduce_csv_ica(filepath, n_components, output_filepath)
            elif reduction_method == "NMF":
                reduce_csv_nmf(filepath, n_components, output_filepath)
            elif reduction_method == "RandomProjection":
                reduce_csv_random_projection(filepath, n_components, output_filepath)
            #elif reduction_method == "MulticoreTSNE":
                #reduce_csv_mctsne(filepath, n_components, output_filepath)
            elif reduction_method == "UMAP":
                reduce_csv_umap(filepath, n_components, output_filepath)

# Get input parameters
n_components = int(input("Enter target dimensionality: "))
filepath_list = input("Enter path to the input file list in CSV format: ")
decimals = int(input("Enter target number of decimals for value rounding: "))
print()
print(">>> You have to specify additional parameters for UMAP reduction: ")
print()
need_help = input("Do you need addtional info about parameters n_neighbors and min_dist? (yes/no): ")
print()
if need_help == 'yes':
    subprocess.call(['python', 'mctsne_halp.py'])

n_neighbors_umap = int(input("Enter value of n_neighbors, default (50): "))
print()
min_dist = float(input("Enter value of min_dist, default (0.5): "))

filepath_list = pd.read_csv(filepath_list, header=None)
filepath_list = filepath_list[0].tolist()

process_file_list(filepath_list, n_components, decimals)

# Stop measuring the execution time
end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time

print()
print("Total execution time: {:.2f} seconds".format(execution_time))
print()