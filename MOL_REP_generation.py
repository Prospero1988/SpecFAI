import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMolDescriptors
import numpy as np

# Funkcja do generowania konkretnego typu fingerprintu
def generate_fingerprint(smiles, fp_type):
    mol = Chem.MolFromSmiles(smiles)
    if fp_type == "RDKit":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 0, nBits=2048)
    elif fp_type == "ECFP4":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    elif fp_type == "MACCS":
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == "Klekota-Roth":
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    return list(fp)

# Funkcja do generowania i zapisywania fingerprintów
def save_fingerprints(df, fp_type):
    fp_data = df['SMILES'].apply(lambda x: generate_fingerprint(x, fp_type))
    fp_df = pd.DataFrame(fp_data.tolist(), index=df.index)
    output_df = pd.concat([df.iloc[:, 0], fp_df], axis=1)
    output_file = f"{fp_type}_fingerprints.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Zapisano {fp_type} fingerprinty do pliku {output_file}")

# Funkcja do generowania deskryptorów molekularnych
def generate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    descriptors = [desc(mol) for _, desc in Descriptors._descList]
    return descriptors

# Funkcja do zapisywania deskryptorów molekularnych
def save_descriptors(df):
    descriptors_data = df['SMILES'].apply(generate_descriptors)
    descriptors_df = pd.DataFrame(descriptors_data.tolist(), index=df.index)
    # Normalizacja
    descriptors_df = (descriptors_df - descriptors_df.min()) / (descriptors_df.max() - descriptors_df.min())
    output_df = pd.concat([df.iloc[:, 0], descriptors_df], axis=1)
    output_file = "Molecular_Descriptors.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Zapisano deskryptory molekularne do pliku {output_file}")

# Główna część skryptu
if __name__ == "__main__":
    input_file = sys.argv[1]
    df = pd.read_csv(input_file)

    # Generowanie i zapisywanie fingerprintów
    for fp_type in ["RDKit", "ECFP4", "MACCS", "Klekota-Roth"]:
        save_fingerprints(df, fp_type)
    
    # Generowanie i zapisywanie deskryptorów
    save_descriptors(df)
