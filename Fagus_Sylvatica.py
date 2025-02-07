from Bio import SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio import Phylo

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

# Construct the phylogenetic tree
calculator = DistanceCalculator('identity')
dm = calculator.get_distance(alignment)
constructor = DistanceTreeConstructor()
tree = constructor.nj(dm)

# Visualize the phylogenetic tree
Phylo.draw(tree)