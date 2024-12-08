import pandas as pd

import matplotlib.pyplot as plt

species = pd.read_csv('AMP_species_list.csv')

from Bio import Entrez
from Bio import SeqIO


def extract_nucleotide_sequence(fasta_file):
    """
    Extracts and returns only the nucleotide sequence from a FASTA file.
    """
    sequences = []
    # Parse the FASTA file
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append(str(record.seq))  # Extract the nucleotide sequence
    return sequences




def get_gene_summary(gene_id):
    """
    Fetch the nucleotide sequence of a gene using its Gene ID.
    """
    try:
 # Use efetch to retrieve the gene sequence
        with Entrez.esummary(db="gene", id=gene_id) as handle:
            gene_summary = Entrez.read(handle)
        return gene_summary
    except Exception as e:
        return int(0)
gene_id = '39411721'
Entrez.email = "robbvos.devos@ugent.be"
temp = get_gene_summary(gene_id)

chrom_info = temp['DocumentSummarySet']['DocumentSummary'][0]['GenomicInfo']

if chrom_info:
    # For simplicity, we take the first genomic context
    chrom_accession = chrom_info[0]['ChrAccVer']
    start = chrom_info[0]['ChrStart']
    end = chrom_info[0]['ChrStop']

    # Step 2: Fetch the nucleotide sequence using Entrez.efetch
    with Entrez.efetch(db="nucleotide", id=chrom_accession, rettype="fasta", strand=1, seq_start=start, seq_stop=end) as handle:
        record = SeqIO.read(handle, "fasta")



Entrez.email = "robbvos.devos@ugent.be"
gene_IDs = pd.read_csv('gene_IDS_COX1.csv')
sequentie = []
ID = []
range = []
test = gene_IDs[["ID"]].loc[0:10]

for i in test['ID']:
    gene_id = i
    summary = get_gene_summary(gene_id)
    chrom_info = summary['DocumentSummarySet']['DocumentSummary'][0]['GenomicInfo']
    ID += [i]
    if chrom_info:
    # For simplicity, we take the first genomic context
        chrom_accession = chrom_info[0]['ChrAccVer']
        start = int(chrom_info[0]['ChrStart'])+1
        end = int(chrom_info[0]['ChrStop'])+1
        range += [abs(int(end)-int(start))+1]

    # Step 2: Fetch the nucleotide sequence using Entrez.efetch
    with Entrez.efetch(db="nucleotide", id=chrom_accession, rettype="fasta", strand=1, seq_start=start, seq_stop=end) as handle:
        record = SeqIO.read(handle, "fasta")
    sequentie += [record.seq]




#effectief uitvoeren

Entrez.email = "robbvos.devos@ugent.be"
gene_IDs = pd.read_csv('gene_IDS_COX1.csv')
sequentie = []
ID = []
range = []
search = gene_IDs[["ID"]].loc[0:len(gene_IDs)-1]

for i in search['ID']:
    gene_id = i
    summary = get_gene_summary(gene_id)
    chrom_info = summary['DocumentSummarySet']['DocumentSummary'][0]['GenomicInfo']
    ID += [i]
    if chrom_info:
    # For simplicity, we take the first genomic context
        chrom_accession = chrom_info[0]['ChrAccVer']
        start = int(chrom_info[0]['ChrStart'])+1
        end = int(chrom_info[0]['ChrStop'])+1
        range += [abs(int(end)-int(start))+1]

    # Step 2: Fetch the nucleotide sequence using Entrez.efetch
    with Entrez.efetch(db="nucleotide", id=chrom_accession, rettype="fasta", strand=1, seq_start=start, seq_stop=end) as handle:
        record = SeqIO.read(handle, "fasta")
    sequentie += [record.seq]