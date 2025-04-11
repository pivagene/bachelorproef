import pandas as pd
from Bio.Seq import Seq
import os

script_dir = os.path.dirname(__file__)

# Define the Folmer region primers
folmer_forward_primer = "GGTCAACAAATCATAAAGATATTGG"
folmer_reverse_primer = "TAAACTTCAGGGTGACCAAAAAATCA"

# Function to search for a primer with allowed mismatches
def search_with_mismatches(sequence, primer, max_mismatches=2):
    sequence = str(sequence)
    primer_len = len(primer)
    for i in range(len(sequence) - primer_len + 1):
        mismatches = sum(1 for a, b in zip(sequence[i:i + primer_len], primer) if a != b)
        if mismatches <= max_mismatches:
            return i
    return -1

# Function to extract the Folmer region from a sequence
def extract_folmer_region(sequence):
    sequence = str(sequence)  # Ensure sequence is a string

    # Search for the forward primer
    forward_start = search_with_mismatches(sequence, folmer_forward_primer, max_mismatches=5)
    if forward_start == -1:
        print(f"Forward primer not found in sequence: {sequence[:50]}...")
        return None

    # Search for the reverse primer on the reverse complement
    reverse_complement = str(Seq(sequence).reverse_complement())
    reverse_start = search_with_mismatches(reverse_complement, folmer_reverse_primer, max_mismatches=5)

    if reverse_start == -1:
        print(f"Reverse primer not found in sequence: {sequence[:50]}...")
        return None

    # Convert reverse_start to match the original sequence coordinates
    reverse_start = len(sequence) - (reverse_start + len(folmer_reverse_primer))

    # Ensure forward_start is before reverse_start
    if forward_start >= reverse_start:
        print(f"Invalid primer positions in sequence: {sequence[:50]}...")
        return None

    # Extract the Folmer region
    folmer_region = sequence[forward_start:reverse_start + len(folmer_reverse_primer)]
    return folmer_region

# Read the CSV file
csv_path = os.path.join(script_dir, '../../csv_files/gene_seq_COX1_final.csv')
if not os.path.exists(csv_path):
    print(f"Error: File not found at {csv_path}")
else:
    df = pd.read_csv(csv_path)

    # Lists to store the extracted information
    gene_ids = []
    folmer_regions = []
    not_found_count = 0

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        gene_id = row['Gene_ID']
        sequence = row['sequentie']

        folmer_region = extract_folmer_region(sequence)

        if folmer_region:
            gene_ids.append(gene_id)
            folmer_regions.append(str(folmer_region))
        else:
            not_found_count += 1  # Count missing regions

    # Create a new DataFrame with the extracted information
    folmer_df = pd.DataFrame({
        'Gene_ID': gene_ids,
        'sequentie': folmer_regions
    })

    # Print the DataFrame
    print(folmer_df)

    # Print the count of Folmer regions not found
    print(f"Number of Folmer regions not found: {not_found_count}")

    # Save the DataFrame to a CSV file
    folmer_df.to_csv(os.path.join(script_dir, '../../csv_files/extracted_folmer_regions.csv'), index=False)
