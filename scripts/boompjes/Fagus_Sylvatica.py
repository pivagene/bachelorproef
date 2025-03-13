from Bio import SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio import Phylo
import matplotlib.pyplot as plt
import subprocess
import os
import pandas as pd
# packages voor alignment raxml en fasttree
# Define the full path to the Clustal Omega executable
clustalomega_exe = r"C:\Program Files\Clustal omega\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\clustalo.exe"  # Replace with the actual path to clustalo.exe

# Ensure unique sequence IDs
input_fasta = "COX1_sequences.fasta"
unique_fasta = "unique_COX1_sequences.fasta"

unique_ids = set()
with open(unique_fasta, 'w') as output_handle:
    for record in SeqIO.parse(input_fasta, "fasta"):
        original_id = record.id
        count = 1
        while record.id in unique_ids:
            record.id = f"{original_id}_{count}"
            count += 1
        unique_ids.add(record.id)
        SeqIO.write(record, output_handle, "fasta")

# Align sequences using Clustal Omega
clustalomega_cline = ClustalOmegaCommandline(cmd=clustalomega_exe, infile=unique_fasta, outfile="COX1_aligned_sequences.fasta", verbose=True, auto=True, force=True)
stdout, stderr = clustalomega_cline()

# Print the standard output and error (if any)
print(stdout)
print(stderr)

# Read the aligned sequences
alignment = AlignIO.read("COX1_aligned_sequences.fasta", "fasta")

# Ensure unique sequence IDs in the alignment
for i, record in enumerate(alignment):
    description_parts = record.description.split()
    if len(description_parts) >= 3:
        record.id = f"{description_parts[1][:5]}_{description_parts[2][:4]}_{i+1}"  # Use the first 5 characters of the second word and the first 4 characters of the third word, plus a unique index
    elif len(description_parts) >= 2:
        record.id = f"{description_parts[1][:9]}_{i+1}"  # Fallback to the first 9 characters of the second word, plus a unique index
    else:
        record.id = f"{record.id[:9]}_{i+1}"  # Fallback to the first 9 characters of the original ID, plus a unique index
    record.name = record.id
    record.description = ""

# Save the aligned sequences to a file in FASTA format for FastTree
phylip_file = "COX1_aligned_sequences.phy"
AlignIO.write(alignment, phylip_file, "phylip")

# Run FastTree to construct the phylogenetic tree
fasttree_exe = r"C:\Program Files\fastTree\FastTree.exe"  # Replace with the actual path to FastTree.exe
tree_file = "COX1_tree.newick"
subprocess.run([fasttree_exe, "-nt", phylip_file], stdout=open(tree_file, 'w'))

# Read the tree
tree = Phylo.read(tree_file, "newick")

# Adjust the figure size
plt.figure(figsize=(20, 20))  # Adjust the size as needed

# Visualize the tree
Phylo.draw(tree)
plt.show()

# Save the tree to a file in Newick format
Phylo.write(tree, 'COX1_tree_with_names.newick', 'newick')







# Construct the phylogenetic tree
calculator = DistanceCalculator('identity')
dm = calculator.get_distance(alignment)
constructor = DistanceTreeConstructor()
tree = constructor.nj(dm)



for clade in tree.find_clades():
    clade.name = None

# Custom function to collapse clades with fewer than a certain number of terminal nodes
def collapse_small_clades(tree, threshold):
    for clade in tree.find_clades(order='postorder'):
        if len(clade.get_terminals()) < threshold:
            clade.collapse_all()

# Collapse clades with fewer than 5 terminal nodes
collapse_small_clades(tree, 0.1)

# Adjust the figure size
plt.figure(figsize=(20, 20))  # Adjust the size as needed

# Visualize the tree
Phylo.draw(tree)
plt.show()


from Bio import Phylo

# Save the tree to a file in Newick format
Phylo.write(tree, 'COX1_tree.newick', 'newick')


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