#packages inladen
import pandas as pd
from Bio import Entrez
from Bio import SeqIO
import numpy as np
#altijd e-mail meegeven aan ncbi 
Entrez.email = "uw.eigen@emailadres.be"
#functie voor het opvragen van gen samenvatting definiëren (enkel in weekends en 's nachts)
def get_gene_summary(gene_id):
    """
    Fetch the nucleotide sequence of a gene using its Gene ID.
    """
    try:
 #efetch gebruiken om samenvatting van het gen op te vragen
        with Entrez.esummary(db="gene", id=gene_id) as handle:
            gene_summary = Entrez.read(handle)
        return gene_summary
    except Exception as e:
        return int(0)
#altijd e-mail meegeven aan ncbi
Entrez.email = "uw.eigen@emailadres.be"
#inlezen van de gen ids uit vorige script
gene_IDs = pd.read_csv('gene_IDS_hetgen.csv')
#lege lijsten klaarzetten voor straks onze waarden in te steken
sequentie = []
ID = []
range = []
#definiëren voor welke IDs je wilt opzoeken 
search = gene_IDs[["ID"]].loc[0:len(gene_IDs)-1]
#for loop initiëren om alle data op te halen
for i in search['ID']:
    gene_id = i
    summary = get_gene_summary(gene_id)
    #info over het gen opvragen
    chrom_info = summary['DocumentSummarySet']['DocumentSummary'][0]['GenomicInfo']
    ID += [i]
    if chrom_info:
        chrom_accession = chrom_info[0]['ChrAccVer']
        #start van gen sequentie opvragen
        start = int(chrom_info[0]['ChrStart'])+1
        #einde van gen sequentie opvragen
        end = int(chrom_info[0]['ChrStop'])+1
        #checken of de start en end plaats logisch zijn indien niet wisselen
        if int(start)>int(end):
            start1=start
            end1=end
            end=start1
            start=end1
        #lengte van het gen bepalen
        range += [abs(int(end)-int(start))+1]

    #met efetch de nucleotiden sequentie ophalen
    with Entrez.efetch(db="nucleotide", id=chrom_accession, rettype="fasta", strand=1, seq_start=start, seq_stop=end) as handle:
        record = SeqIO.read(handle, "fasta")
    sequentie += [record.seq]
#data in een data frame zetten
ID = np.array(ID)
range = np.array(range)
df = pd.DataFrame({"ID": ID, "length": range,"sequence":sequentie})
# df.to_csv('gene_seq_hetgen.csv', index =False)