import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from collections import Counter
import numpy as np


script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_seq_COX1_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_COX1_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_tref_parameters.csv'))  # Contains animal and characteristic
characteristics_df['Species'] = characteristics_df['ScientificName'].str.replace(' ', '_', regex=False)

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'Gene_ID'
merged_df = pd.merge(merged_df, characteristics_df, left_on='ID', right_on='Species')  # Merge on 'ID'

# Ensure the target column 'v' exists and drop rows with missing values in 'v'
merged_df = merged_df.dropna(subset=['v'])  # Drop rows where 'kap_R' is NaN

# Extract k-mer features from the DNA sequences
kmer_features = []
for seq in merged_df['sequentie']:
    sequence = str(seq)
    kmer_feature = get_kmers(sequence)
    kmer_features.append(kmer_feature)

# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Reset indices before concatenation
merged_df = merged_df.reset_index(drop=True)
kmer_df = kmer_df.reset_index(drop=True)

# Combine features with the merged DataFrame
df = pd.concat([merged_df, kmer_df], axis=1)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Filter out columns with invalid k-mers
valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
filtered_kmer_df = kmer_df[valid_kmer_columns]

# Prepare the data
feature_columns = list(filtered_kmer_df.columns)  # Use filtered k-mer features
X = df[feature_columns]
y = df['v']

# Train the XGBoost model
model = XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=4,random_state=42)

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_r2_scores}")
print(f"Mean Cross-Validation R² Score: {cv_r2_scores.mean()}")