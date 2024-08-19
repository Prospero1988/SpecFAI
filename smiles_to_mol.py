import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_mol_files(csv_path):
    try:
        # Load the CSV file
        data = pd.read_csv(csv_path)
        
        # Check if the necessary columns exist
        if 'MOLECULE_NAME' not in data.columns or 'SMILES' not in data.columns:
            raise ValueError("CSV file must contain 'MOLECULE_NAME' and 'SMILES' columns.")
        
        file_count = 0
        
        for index, row in data.iterrows():
            try:
                name = row['MOLECULE_NAME']
                smiles = row['SMILES']
                
                # Generate the molecule from SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError(f"Invalid SMILES string: {smiles}")
                
                # Add hydrogens
                mol = Chem.AddHs(mol)
                
                # Generate 3D coordinates
                AllChem.EmbedMolecule(mol)
                
                # Optimize the molecule in 2D
                AllChem.Compute2DCoords(mol)
                
                # Save the molecule to a .mol file
                mol_file = f"{name}.mol"
                with open(mol_file, 'w') as f:
                    f.write(Chem.MolToMolBlock(mol))
                
                file_count += 1
                
            except Exception as e:
                print(f"Error processing row {index}: {e}")
        
        print(f"Generated {file_count} .mol files.")
    
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except pd.errors.EmptyDataError:
        print(f"File is empty: {csv_path}")
    except pd.errors.ParserError:
        print(f"Error parsing CSV file: {csv_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
    else:
        csv_path = sys.argv[1]
        generate_mol_files(csv_path)
