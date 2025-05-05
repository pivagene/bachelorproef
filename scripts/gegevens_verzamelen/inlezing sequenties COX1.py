import pandas as pd
from Bio import Entrez, SeqIO
import os

script_dir = os.path.dirname(__file__)

Entrez.email = "your@email.com"
Entrez.api_key = "your_API"  

def get_gene_summary(gene_id):
    """Fetch the gene summary from NCBI using Entrez.esummary."""
    try:
        with Entrez.esummary(db="gene", id=gene_id) as handle:
            gene_summary = Entrez.read(handle)
        return gene_summary
    except Exception:
        return None

def get_best_genomic_info(summary):
    """Select the best genomic accession based on quality priority."""
    chrom_info = summary['DocumentSummarySet']['DocumentSummary'][0]['GenomicInfo']
    
    if not chrom_info:
        return None, None, None
    
    best_accession = None
    best_quality = float('inf')
    best_start, best_end = None, None

    priority_order = ["NC_", "NG_", "NT_", "NW_"]  # Reference > Curated > Contig > Scaffold

    for context in chrom_info:
        accession = context['ChrAccVer']
        start, end = int(context['ChrStart']) + 1, int(context['ChrStop']) + 1

        for i, prefix in enumerate(priority_order):
            if accession.startswith(prefix) and i < best_quality:
                best_quality = i
                best_accession = accession
                best_start, best_end = start, end

    return best_accession, best_start, best_end

# Read gene IDs from CSV
gene_IDs = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_12SrRNA.csv'))
gene_IDs = gene_IDs[['ID']]  

# Initialize lists
ID = []
range_values = []
sequentie = []
fasta_files = []
for gene_id in gene_IDs['ID']:  # Ensure 'ID' column exists in the CSV
    summary = get_gene_summary(gene_id)
    
    if summary:
        chrom_accession, start, end = get_best_genomic_info(summary)
        
        if chrom_accession:
            ID.append(gene_id)
            if int(start) > int(end):
                start, end = end, start
            range_values.append(abs(end - start) + 1)

            # Fetch the best sequence
            with Entrez.efetch(db="nucleotide", id=chrom_accession, rettype="fasta", strand=1, seq_start=start, seq_stop=end) as handle:
                record = SeqIO.read(handle, "fasta")
            
            sequentie.append(str(record.seq))  # Store sequence as string
            fasta_files.append(record.format("fasta"))  # Store FASTA format
            
# Convert results into a DataFrame and save 
df_fasta = pd.DataFrame({'Gene_ID': ID, 'Range': range_values, 'FASTA': fasta_files}) # sequentie kan nog toegevoegd worden maar zit al in fasta_files
df_seq = pd.DataFrame({'Gene_ID': ID, 'Range': range_values, 'sequentie': sequentie})
df_fasta.to_csv(os.path.join(script_dir, '../../csv_files/gene_FASTA_12SrRNA.csv'), index=False)
df_seq.to_csv(os.path.join(script_dir, '../../csv_files/gene_seq_12SrRNA.csv'), index=False)