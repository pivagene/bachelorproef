import pandas as pd
seq1 = pd.read_csv('AMP_species_list_COX1.csv')
seq2 = pd.read_csv('AMP_species_list_12SrRNA.csv')
seq12 = pd.merge(seq1, seq2, on='ScientificName' )
seq12.to_csv('merged_species_list.csv', index=False)
