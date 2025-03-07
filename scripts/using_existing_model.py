import joblib
from collections import Counter
import pandas as pd
import os

script_dir = os.path.dirname(__file__)

# Load the model
model = joblib.load(os.path.join(script_dir, '../../other_files/random_forest_model_kmer_COX1.pkl'))

# Function to extract k-mers from a sequence
def get_kmers(sequence, k=4):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Example sequence
sequence = 'TTAGTTTTTTGTGCTGGATCTAATTGGGACAAAAACAAATGGTAGTTCTTCATATGTATGATGGCTGGGTGGGGAATTGTGTACTCATTCTAATGATGGTTGGTATCCATTATTGTTTGTTCATCCTACAAATTTTATTTCTCTGTAAAAAGCGTCGTATATTATATATATAAATAATATAGTAGCAACAATTGATATTGCAGAACCTATTGAGCTTATCAAATTTCATCCGGCAAATCCATCAGGAAAATCTGAGTAACGTCTTGGCATTCCTGCTAATCCTAAAAAATGTTGAGGGAAAAATGTTGTATTTACTCCTATAAACATTAATCAGAAATGGATTTTGGCATAAGTTTCGTTATAAGAATACCCTGTTATTTTGCCAAACCAATAATAGAATCCGGCAAATATTGTAAATACGGCTCCTAAGGACAATACATAATGAAAGTGTGCTACTACATAATATGTGTCATGAAATGCAATATCTAATGATCCGTTAGCTAATAATACTCCTGTTAATCCTCC'
#feature columns moeten mee opgeslaan worden wanneer een model wordt gemaakt. 
# Om het model te gebruiken na het opnieuw in te laden heb je deze exacte feature columns namelijk nodig om geen errors te krijgen.