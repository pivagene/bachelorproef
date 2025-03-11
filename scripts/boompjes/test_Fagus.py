from Bio import SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
from Bio import Phylo
import matplotlib.pyplot as plt
import pandas as pd
import subprocess

# Define the full path to the Clustal Omega executable
clustalomega_exe = r"C:\Program Files\Clustal omega\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\clustalo.exe"  # Replace with the actual path to clustalo.exe

# Ensure unique sequence IDs
input_fasta = "COX1_sequences.fasta"
unique_fasta = "unique_COX1_sequences.fasta"
names_df = pd.read_csv("AMP_species_list_COX1.csv")

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
clustalomega_cline = ClustalOmegaCommandline(cmd=clustalomega_exe, infile=unique_fasta, outfile="COX1_aligned_sequences.fasta", verbose=True, auto=True, force=True)
stdout, stderr = clustalomega_cline()

# Print the standard output and error (if any)
print(stdout)
print(stderr)

# Read the aligned sequences
alignment = AlignIO.read("COX1_aligned_sequences.fasta", "fasta")

# Save the aligned sequences to a file in PHYLIP format for FastTree
phylip_file = "COX1_aligned_sequences.phy"
AlignIO.write(alignment, phylip_file, "phylip")

# Read the aligned sequences
alignment = AlignIO.read(phylip_file, "phylip")

# Run FastTree to construct the phylogenetic tree
fasttree_exe = r"C:\Program Files\fastTree\FastTree.exe"  # Replace with the actual path to FastTree.exe
tree_file = "COX1_tree.newick"
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
Phylo.write(tree, 'COX1_tree_with_names_C.newick', 'newick')

#creating the annotation file for the tree
# Load your DataFrame with IDs and qualitative metrics
df = pd.read_csv('AMP_species_list_COX1.csv')  # Replace with the actual path to your CSV file

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
with open('COX1_annotations.txt', 'w') as f:
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