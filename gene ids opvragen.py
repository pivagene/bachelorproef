from Bio import Entrez
import pandas as pd
# altijd eigen email meegeven
Entrez.email = "lars.prvte@gmail.com"
Entrez.api_key = "7d69d81fcd4a1d8a414cf6dce34c16fc2a09"
# functie om gene id op te halen
def get_gene_id(id, gene_name):
    # zoek het gen in NCBI aan de hand van deze query
    search_term = f"{gene_name}[title] AND {id}[Organism] NOT discontinued[properties]"
    handle = Entrez.esearch(db="gene", term=search_term, retmax=1)
    record = Entrez.read(handle)
    handle.close()
        
    # haal het gen uit de handle
    if record["IdList"]:
        gene_id = record["IdList"][0]
        return gene_id
    else:
        gene_id = int(0)
# lees de soorten uit de database in 
species = pd.read_csv('AMP_species_list.csv')
search = species[['ID']].loc[0:4941]
# geef het gen waar je naar op zoek bent mee 
gene_name = "12S ribosomal RNA"
# initieer 2 lege dataframes om de waarden straks in te steken pre-allocatie heeft geen nut aangezien er een maximum snelheid op het opvragen van IDs
gene_ids = []
species_with_COI = []
# vraag de genen op en steek ze in de lijsten
for id in search['ID'] : 
    gene_id = get_gene_id(id, gene_name)
    if type(gene_id) == int:
        continue
    gene_ids += [gene_id]
    species_with_COI += [id]
# maak een dataframe van deze 2 lijsten
df = pd.DataFrame({"ID": species_with_COI, "Gene_ID": gene_ids})
# verwijder de species die geen ID hebben opgeleverd (en vervang de vorige dataframe)
df.dropna(subset=['Gene_ID'],inplace=True)
#verander de float64 in integer
df['Gene_ID']=df['Gene_ID'].astype(int)
# schrijf deze dataframe weg
df.to_csv('gene_IDS_12SrRNA.csv', index =False)