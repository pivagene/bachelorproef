from Bio import Entrez
import pandas as pd
# altijd eigen email meegeven
Entrez.email = "lars.vandenbossche@ugent.be"
# functie om gene id op te halen
def get_gene_id(id, gene_name):
    # zoek het gen in NCBI aan de hand van deze query
    search_term = f"{gene_name}[Gene] AND {id}[Organism]"
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
# geef het gen waar je naar op zoek bent mee (in dit geval COX1)
gene_name = "COX1"
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
df = pd.DataFrame({"Species": species_with_COI, "ID": gene_ids})
# verwijder de species die geen ID hebben opgeleverd (en vervang de vorige dataframe)
df.dropna(subset=['ID'],inplace=True)
# schrijf deze dataframe weg (vervang COX1 door het gewenste gen)
df.to_csv('gene_IDS_COX1.csv', index =False)