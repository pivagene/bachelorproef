from Bio import SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
import subprocess

# Define the full path to the Clustal Omega executable
clustalomega_exe = r"C:\Program Files\Clustal omega\clustal-omega-1.2.2-win64\clustal-omega-1.2.2-win64\clustalo.exe"  # Replace with the actual path to clustalo.exe

# Ensure unique sequence IDs
input_fasta = "COX1_sequences.fasta"
unique_fasta = "unique_COX1_sequences.fasta"
id_mapping_file = "id_mapping.txt"

id_mapping = {}
with open(unique_fasta, 'w') as output_handle, open(id_mapping_file, 'w') as mapping_handle:
    for i, record in enumerate(SeqIO.parse(input_fasta, "fasta")):
        description_parts = record.description.split()
        if len(description_parts) >= 3:
            original_id = f"{description_parts[1]}_{description_parts[2]}"
        elif len(description_parts) >= 2:
            original_id = description_parts[1]
        else:
            original_id = record.id
        unique_id = f"seq{i+1}"
        id_mapping[unique_id] = original_id
        mapping_handle.write(f"{unique_id}\t{original_id}\n")
        record.id = unique_id
        record.name = unique_id
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

from Bio import AlignIO
from Bio import Phylo
import matplotlib.pyplot as plt
import subprocess

# Load the ID mapping
id_mapping_file = "id_mapping.txt"
id_mapping = {}
with open(id_mapping_file, 'r') as mapping_handle:
    for line in mapping_handle:
        unique_id, original_id = line.strip().split('\t')
        id_mapping[unique_id] = original_id

# Read the aligned sequences
phylip_file = "COX1_aligned_sequences.phy"
alignment = AlignIO.read(phylip_file, "phylip")

# Run FastTree to construct the phylogenetic tree
fasttree_exe = r"C:\Program Files\fastTree\FastTree.exe"  # Replace with the actual path to FastTree.exe
tree_file = "COX1_tree.newick"
subprocess.run([fasttree_exe, "-nt", phylip_file], stdout=open(tree_file, 'w'))

# Read the tree
tree = Phylo.read(tree_file, "newick")

# Replace unique IDs with original IDs
for clade in tree.find_clades():
    if clade.name in id_mapping:
        clade.name = id_mapping[clade.name]

# Adjust the figure size
plt.figure(figsize=(20, 20))  # Adjust the size as needed

# Visualize the tree
Phylo.draw(tree)
plt.show()

# Save the tree to a file in Newick format
Phylo.write(tree, 'COX1_tree_with_names.newick', 'newick')
