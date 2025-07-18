import pandas as pd
import subprocess
import os
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Blast import NCBIXML
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import io
import joblib
import threading
from Bio import pairwise2
from xgboost import XGBRegressor
script_dir = os.path.dirname(__file__)

# Step 1: Read the CSV file and write to a FASTA file and choose how to make the db
fasta_df = pd.read_csv('gene_FASTA_12SrRNA_final.csv')  # Contains sequence and ID
animals_df = pd.read_csv('gene_IDS_12SrRNA_final.csv')  # Contains ID and animal information
characteristics_df = pd.read_csv('AMP_species_list.csv')  # Contains animal and characteristic

# Merge the DataFrames
df = pd.merge(fasta_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
df = pd.merge(df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'

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

# Function to perform local BLAST alignment and extract alignment score, bit score, and global alignment score
def get_blast_and_global_scores(sequence, db='12SrRNA_blastdb'):
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
                # Perform global alignment
                global_alignments = pairwise2.align.globalxx(sequence, hsp.sbjct)
                global_score = global_alignments[0][2] if global_alignments else 0
                return hsp.score, hsp.bits, global_score
    return 0, 0, 0

# Load the CSV files
sequences_df = pd.read_csv('gene_seq_12SrRNA_final.csv')  # Contains sequence and ID
animals_df = pd.read_csv('gene_IDS_12SrRNA_final.csv')  # Contains ID and animal information
characteristics_df = pd.read_csv('AMP_species_list.csv')  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'
merged_df.to_csv('AMP_species_list_12SrRNA.csv', index=False)
merged_df = pd.read_csv('mitofish_sequences.csv')
# Merge dataframes with parameters dataframes
df_AMP_collection = pd.read_csv('AMP_collection.csv')
df_parameter = df_AMP_collection[df_AMP_collection['Data']=='Ri']
df_parameter['Observed_log'] = np.log(df_AMP_collection['Observed'])
df_parameter_selection = df_parameter[['Data','Observed_log','Predicted','Species','Unit']]
# df_parameter_selection = df_parameter[['Data','Observed','Predicted','Species','Unit']]
df_parameter_selection = df_parameter_selection.rename(columns={"Species":"ID"})
merged_df_par = pd.merge(df_parameter_selection,merged_df,on='ID',how='inner')


# Extract features from the FASTA sequences
features = []
kmer_features = []
blast_scores = []
bit_scores = []
global_scores = []

# Process each sequence one by one
for seq in merged_df['sequentie']:
    sequence = str(seq)
    feature = extract_features(sequence)
    kmer_feature = get_kmers(sequence)
    blast_score, bit_score, global_score = get_blast_and_global_scores(sequence)#  dit moet met fasta bestanden gedaan worden
    
    features.append(feature)
    kmer_features.append(kmer_feature)
    blast_scores.append(blast_score)
    bit_scores.append(bit_score)
    global_scores.append(global_score)

# Convert features to a DataFrame
features_df = pd.DataFrame(features)
# features_df = pd.read_csv("blast_scores.csv") als je de blast al hebt gedaan
# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Add BLAST scores, bit scores, and global alignment scores to the features DataFrame
features_df['blast_score'] = blast_scores
features_df['bit_score'] = bit_scores
features_df['global_score'] = global_scores

# Combine features with the merged DataFrame
df = pd.concat([merged_df, features_df, kmer_df], axis=1)

# Prepare the data
# Assuming 'Characteristic' is the column you want to predict
# Specify the columns you want to use for X
feature_columns = [] + list(kmer_df.columns)  # Add more features as needed length and gc doen ni veel kmer wint zelf zonder blast score
X = df[feature_columns]
y = df['Mod']  # Replace with your target column

# Ensure there are no missing values
X = X.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

# Train the Random Forest model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2, random_state=42)
# xgboost package gebruiken
# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

# Convert negative MSE to positive
cv_scores = -cv_scores

print(f'Cross-Validation Mean Squared Error: {cv_scores.mean()}')
print(f'Cross-Validation Standard Deviation: {cv_scores.std()}')

# Fit the model on the entire training set
# Train the SVM model
# model = SVC(kernel='linear', random_state=37) # svc tot nu toe het beste voor 12SrRNA
model = RandomForestClassifier(n_estimators=100, random_state=37)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Nauwkeurigheid: {accuracy}')
print('Classificatie Rapport:')
print(classification_report(y_test, y_pred))

# Optionally, save the model for future use
joblib.dump(model, 'SVM_model_kmer_12SrRNA.pkl')

# Save the blast scores, bit scores, and global alignment scores for future use
features_df.to_csv('blast_scores_12SrRNA_1db.csv', index=False)

value_counts = merged_df_par['Data'].value_counts()
print(value_counts)
