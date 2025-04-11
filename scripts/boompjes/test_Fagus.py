from Bio import SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
from Bio import Phylo
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
script_dir = os.path.dirname(__file__)
# Define the full path to the Clustal Omega executable
clustalomega_exe = r"C:\Program Files\Clustal omega\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\clustalo.exe"  # Replace with the actual path to clustalo.exe


# Combine all FASTA sequences from the CSV file into one big FASTA file
# gene_fasta_csv = os.path.join(script_dir, '../../csv_files/gene_FASTA_mitofish_final.csv')
# output_fasta_file = "mitofish_sequences.fasta"

# with open(output_fasta_file, 'w') as output_fasta:
    # gene_fasta_df = pd.read_csv(gene_fasta_csv)
    # for fasta_sequence in gene_fasta_df['FASTA']:
        # output_fasta.write(f"{fasta_sequence}\n")


# Combine all single FASTA files in the mitofish_data folder into one big FASTA file
mitofish_data_dir = os.path.join(script_dir, '..\..\mitofish_data')  # Adjust path if needed
combined_fasta_file = "mitofish_sequences.fasta"

# Debugging: Check if the directory exists
if not os.path.exists(mitofish_data_dir):
    raise ValueError(f"The directory {mitofish_data_dir} does not exist. Please check the path.")

# Debugging: Log all files in the directory
all_files = os.listdir(mitofish_data_dir)
print(f"All files in {mitofish_data_dir}: {all_files}")

# Filter files containing '.fasta' or '.fa' in their names
fasta_files = [f for f in all_files if '.fasta' in f or '.fa' in f]
print(f"Found {len(fasta_files)} FASTA files in {mitofish_data_dir}.")
if not fasta_files:
    raise ValueError(f"No FASTA files found in the directory {mitofish_data_dir}. Please check the input data.")

# Combine the contents of all FASTA files
with open(combined_fasta_file, 'w') as combined_fasta:
    for filename in fasta_files:
        file_path = os.path.join(mitofish_data_dir, filename)
        try:
            # Debugging: Log the file being processed
            print(f"Processing file: {filename}")
            
            # Parse the FASTA file to ensure it contains valid sequences
            records = list(SeqIO.parse(file_path, "fasta"))
            if not records:
                print(f"Warning: The file {filename} contains no valid sequences and will be skipped.")
                continue
            
            # Write each sequence individually to the combined file
            for record in records:
                SeqIO.write(record, combined_fasta, "fasta")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

# Debugging: Log the number of sequences in the combined FASTA file
combined_sequences = list(SeqIO.parse(combined_fasta_file, "fasta"))
print(f"Number of sequences in {combined_fasta_file}: {len(combined_sequences)}")
if len(combined_sequences) == 0:
    raise ValueError(f"The file {combined_fasta_file} is empty after combining. Please check the input FASTA files.")

# Ensure unique sequence IDs
input_fasta = "mitofish_sequences.fasta"
unique_fasta = "unique_mitofish_sequences.fasta"
names_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list_mitofish.csv'))

# Create a new column for unique IDs
names_df['unique_id'] = [f"seq{i+1}" for i in range(len(names_df))]

# Create the id_mapping_df DataFrame
id_mapping_df = names_df[['unique_id', 'ScientificName']].rename(columns={'ScientificName': 'scientific_name'})

# Create a dictionary for easy lookup
id_mapping = pd.Series(id_mapping_df.scientific_name.values, index=id_mapping_df.unique_id).to_dict()

# Ensure unique scientific names
seen_names = set()
for unique_id, scientific_name in id_mapping.items():
    if scientific_name in seen_names:
        count = 1
        new_name = f"{scientific_name}_{count}"
        while new_name in seen_names:
            count += 1
            new_name = f"{scientific_name}_{count}"
        id_mapping[unique_id] = new_name
    seen_names.add(id_mapping[unique_id])

# Update the FASTA file with the new unique IDs
with open(unique_fasta, 'w') as output_handle:
    for record, unique_id in zip(SeqIO.parse(input_fasta, "fasta"), names_df['unique_id']):
        record.id = unique_id  # Ensure the ID is a string without quotes
        record.name = unique_id  # Ensure the name is a string without quotes
        record.description = ""
        SeqIO.write(record, output_handle, "fasta")

# Align sequences using Clustal Omega
clustalomega_cline = ClustalOmegaCommandline(cmd=clustalomega_exe, infile=unique_fasta, outfile="mitofish_aligned_sequences.fasta", verbose=True, auto=True, force=True)
stdout, stderr = clustalomega_cline()

# Print the standard output and error (if any)
print(stdout)
print(stderr)

# Read the aligned sequences
alignment = AlignIO.read("mitofish_aligned_sequences.fasta", "fasta")

# Save the aligned sequences to a file in PHYLIP format for FastTree
phylip_file = "mitofish_aligned_sequences.phy"
AlignIO.write(alignment, phylip_file, "phylip")

# Read the aligned sequences
alignment = AlignIO.read(phylip_file, "phylip")

# Run FastTree to construct the phylogenetic tree
fasttree_exe = r"C:\Program Files\fastTree\FastTree.exe"  # Replace with the actual path to FastTree.exe
tree_file = "mitofish_tree.newick"
subprocess.run([fasttree_exe, "-nt", phylip_file], stdout=open(tree_file, 'w'))

# Read the tree
tree = Phylo.read(tree_file, "newick")

# Replace unique IDs with scientific names
for clade in tree.find_clades():
    if clade.name in id_mapping:
        clade.name = id_mapping[clade.name].replace("'", "")

# Adjust the figure size
plt.figure(figsize=(20, 20))  # Adjust the size as needed

# Visualize the tree
Phylo.draw(tree)
plt.show()

# Save the tree to a file in Newick format
Phylo.write(tree, 'mitofish_tree_with_names.newick', 'newick')

#creating the annotation file for the tree
# Load your DataFrame with IDs and qualitative metrics
df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list_mitofish.csv'))

# Define a function to map qualitative metrics to colors
def metric_to_color(metric):
    color_mapping = {
        'std': '#FF0000',  # Red
        'abj': '#00FF00',  # Green
        'stx': '#0000FF',  # Blue
        'stf': '#FFFF00',  # Yellow
        'abp': '#FF00FF',  # Magenta
        'hex': '#00FFFF',  # Cyan
        'ssj': '#800000',  # Maroon
        'hep': '#808000',  # Olive
        'hax': '#008000',  # Dark Green
        'asj': '#800080',  # Purple
        'sbp': '#000080'   # Navy
    }
    return color_mapping.get(metric, '#000000')

# Create the annotation file
with open('mitofish_annotations.txt', 'w') as f:
    f.write('DATASET_COLORSTRIP\n')
    f.write('SEPARATOR TAB\n')
    f.write('DATASET_LABEL\tmodel Colors\n')
    f.write('COLOR\t#ff0000\n')
    f.write('LEGEND_TITLE\tmodel\n')
    f.write('LEGEND_SHAPES\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\n')
    f.write('LEGEND_COLORS\t#FF0000\t#00FF00\t#0000FF\t#FFFF00\t#FF00FF\t#00FFFF\t#800000\t#808000\t#008000\t#800080\t#000080\n')
    f.write('LEGEND_LABELS\tstd\tabj\tstx\tstf\tabp\thex\tssj\thep\thax\tasj\tsbp\n')
    f.write('BORDER_WIDTH\t1\n')
    f.write('BORDER_COLOR\t#000000\n')
    f.write('DATA\n')
    for index, row in df.iterrows():
        color = metric_to_color(row['Mod'])
        id_with_quotes = f"'{row['ID'].replace('_', ' ')}'"  # Add single quotes and replace underscores with spaces
        f.write(f"{id_with_quotes}\t{color}\n")

# Aspidoscelis_sexlineatus stf -> std maar std ook goeie