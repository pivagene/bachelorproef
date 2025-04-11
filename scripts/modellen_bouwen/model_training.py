import pandas as pd
import subprocess
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Blast import NCBIXML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import io
import joblib
import threading
script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)


# Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_FASTA_COX1_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_IDS_COX1_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir,'../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'
merged_df.to_csv(os.path.join(script_dir,'../csv_files/AMP_species_list_COX1.csv'), index=False)

# Extract features from the FASTA sequences
features = []
kmer_features = []
blast_scores = []
bit_scores = []

# Process each sequence one by one
for fasta in merged_df['FASTA']:
    sequence = str(Seq(fasta))
    kmer_feature = get_kmers(sequence)
    
    kmer_features.append(kmer_feature)

# Convert features to a DataFrame
features_df = pd.DataFrame(features)
# features_df = pd.read_csv("blast_scores.csv")
# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Add BLAST scores and bit scores to the features DataFrame
features_df['blast_score'] = blast_scores
features_df['bit_score'] = bit_scores

# Combine features with the merged DataFrame
df = pd.concat([merged_df, features_df, kmer_df], axis=1)

# Prepare the data
# Assuming 'Characteristic' is the column you want to predict
# Specify the columns you want to use for X
feature_columns = ['blast_score', 'bit_score'] + list(kmer_df.columns)  # Add more features as needed length and gc doen ni veel kmer wint zelf zonder blast score
X = df[feature_columns]
y = df['Mod']  # Replace with your target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=300, random_state=37)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Optionally, save the model for future use
joblib.dump(model, os.path.join('../other_files/random_forest_model.pkl'))

# Save the blast scores and bit scores for future use
features_df.to_csv(os.path.join(script_dir,'../csv_files/blast_scores_COX1_1db.csv'), index=False)