import pandas as pd
import subprocess
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Blast import NCBIXML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score,median_absolute_error
from collections import Counter
import io
import joblib
import threading
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the CSV file and write to a FASTA file and choose how to make the db
csv_file = 'AMP_species_list_12SrRNA.csv'
df = pd.read_csv(csv_file)
# method 1: alle fasta bestanden uit csv halen om db te maken 
with open('12SrRNA_db.fasta', 'w') as fasta_file:   # alle fasta bestanden uit csv halen om db te maken
    for index, row in df.iterrows():
        fasta_file.write(f">{row['FASTA']}\n")
# method 2: 1 fasta bestand uit csv halen om db te maken
# Extract the value from the 'fasta' column at row 604
fasta_value = df.at[604, 'FASTA']

# Save the extracted value as a FASTA file
with open('12SrRNA_db.fasta', 'w') as fasta_file:
    fasta_file.write(f'>sequence_604\n{fasta_value}\n')
# voor deze stap moet je blast+ installeren en in path zetten bij de omgevingsvariabelen (https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
# Step 2: Create the BLAST database (hier verschillende groottes gebruiken kan ook via ncbi gdn worden expected time is 5 uur tho)
subprocess.run(['makeblastdb', '-in', '12SrRNA_db.fasta', '-dbtype', 'nucl', '-out', '12SrRNA_blastdb'])

# Function to extract features from a sequence
def extract_features(sequence):
    features = {
        'length': len(sequence),
        'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence)
    }
    return features

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to perform local BLAST alignment and extract alignment score and bit score
def get_blast_scores(sequence, db='12SrRNA_blastdb'):
    # Use a unique filename for each thread
    temp_fasta_filename = f'temp_query_{threading.get_ident()}.fasta'
    
    # Write the sequence to a temporary FASTA file
    with open(temp_fasta_filename, 'w') as temp_fasta:
        temp_fasta.write(f'>query\n{sequence}\n')
    
    # Run the local BLAST search
    result = subprocess.run(
        ['blastn', '-query', temp_fasta_filename, '-db', db, '-outfmt', '5'],
        capture_output=True, text=True
    )
    
    # Remove the temporary FASTA file
    os.remove(temp_fasta_filename)
    
    # Check if the BLAST search produced valid output
    if result.stdout.strip():
        # Parse the BLAST XML output
        blast_records = NCBIXML.parse(io.StringIO(result.stdout))
        for blast_record in blast_records:
            if blast_record.alignments:
                hsp = blast_record.alignments[0].hsps[0]
                return hsp.score, hsp.bits
    return 0, 0

# Load the CSV files
sequences_df = pd.read_csv('gene_seq_12SrRNA_final.csv')  # Contains sequence and ID
animals_df = pd.read_csv('gene_IDS_12SrRNA_final.csv')  # Contains ID and animal information
characteristics_df = pd.read_csv('AMP_species_list.csv')  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'
merged_df.to_csv('AMP_species_list_12SrRNA.csv', index=False)

# Merge dataframes with parameters dataframes
df_AMP_collection = pd.read_csv('AMP_collection.csv')
df_parameter = df_AMP_collection[df_AMP_collection['Data']=='tg']
df_parameter_selection = df_parameter[['Data','Observed','Predicted','Species']]
df_parameter_selection = df_parameter_selection.rename(columns={"Species":"ID"})
merged_df_par = pd.merge(df_parameter_selection,merged_df,on='ID',how='inner')


# Extract features from the FASTA sequences
features = []
kmer_features = []
#blast_scores = []
#bit_scores = []

# Process each sequence one by one
for seq in merged_df_par['sequentie']:
    sequence = str(seq)
    feature = extract_features(sequence)
    kmer_feature = get_kmers(sequence)
    # blast_score, bit_score = get_blast_scores(sequence) dit moet met fasta bestanden gedaan worden
    
    features.append(feature)
    kmer_features.append(kmer_feature)
    #blast_scores.append(blast_score)
    #bit_scores.append(bit_score)

# Convert features to a DataFrame
features_df = pd.DataFrame(features)
# features_df = pd.read_csv("blast_scores.csv") als je de blast al hebt gedaan
# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Add BLAST scores and bit scores to the features DataFrame
#features_df['blast_score'] = blast_scores
#features_df['bit_score'] = bit_scores
#features_df = pd.read_csv('blast_scores_12SrRNA_fulldb.csv')
# Combine features with the merged DataFrame
df = pd.concat([merged_df_par, features_df, kmer_df], axis=1)

# Prepare the data
# Assuming 'Characteristic' is the column you want to predict
# Specify the columns you want to use for X
feature_columns = [] + list(kmer_df.columns)  # Add more features as needed length and gc doen ni veel kmer wint zelf zonder blast score
X = df[feature_columns]
y = df['Observed']  # Replace with your target column

# Ensure there are no missing values
X = X.fillna(0)
y = y.fillna(0)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the Random Forest model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3,random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
med = median_absolute_error(y_test,y_pred)
print(f'Mean Squared Error:{mse}')
print(f'R^2 Score:{r2}')
print(f'Median Absolute Error:{med}')

x = np.linspace(0,500,1000)
y = x
plt.scatter(y_pred,y_test)
plt.plot(x,y)

# Optionally, save the model for future use
joblib.dump(model, 'random_forest_model_kmer_12SrRNA.pkl')

# Save the blast scores and bit scores for future use
features_df.to_csv('blast_scores_12SrRNA_1db.csv', index=False)