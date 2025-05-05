import os
from Bio import SeqIO
import pandas as pd

script_dir = os.path.dirname(__file__)

# Function to extract scientific name from the description
def extract_scientific_name(description):
    # Split the description by '|'
    parts = description.split('|')
    # The scientific name is the last part of the description
    if len(parts) > 5:
        scientific_name = parts[6].strip()
        # Replace spaces with underscores
        scientific_name = scientific_name.replace(' ', '_')
        return scientific_name
    return None

# Replace 'your_folder_path' with the actual folder path containing the FASTA files
folder_path = r'C:\Users\Gebruiker\Desktop\test python\wetransfer_bachelorproef_amp_2024-11-07_1055\Bachelorproef_AMP\bachelorproef\mitofish_data'

# Lists to store the extracted information
animal_names = []
gene_sequences = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.fa') or filename.endswith('.fasta'):  # Check if the file is a FASTA file
        file_path = os.path.join(folder_path, filename)
        
        # Read the FASTA file
        for record in SeqIO.parse(file_path, "fasta"):
            animal_name = extract_scientific_name(record.description)
            sequence = str(record.seq)
            
            animal_names.append(animal_name)
            gene_sequences.append(sequence)

# Create a DataFrame
df = pd.DataFrame({
    'ID': animal_names,
    'sequentie': gene_sequences
})

# Read the second DataFrame containing more scientific names
df_more_names = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list.csv'))

# Merge the two DataFrames on the 'Scientific_Name' column
merged_df = pd.merge(df, df_more_names, on='ID', how='inner')

merged_df.to_csv('mitofish_sequences.csv', index=False)
