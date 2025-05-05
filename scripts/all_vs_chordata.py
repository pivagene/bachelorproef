import pandas as pd
import os
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_seq_12SrRNA_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_IDS_12SrRNA_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'

# Precompute k-mer features for all sequences
kmer_features = []
for seq in merged_df['sequentie']:
    sequence = str(seq)
    kmer_feature = get_kmers(sequence)
    kmer_features.append(kmer_feature)

# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

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
feature_columns = list(filtered_kmer_df.columns)  # Add more features as needed
X = df[feature_columns].to_numpy()  # Convert to NumPy array for faster operations
y = df['Mod'].to_numpy()  # Convert to NumPy array for faster operations

# Ensure there are no missing values
X = np.nan_to_num(X)

# Perform cross-validation 30 times and store the mean scores
cv_scores_means = []
rkf = RepeatedKFold(n_splits=7, n_repeats=3)  # Define cross-validation strategy
model = RandomForestClassifier(n_estimators=100)  # Define the model


def compute_cv_mean():
    cv_scores = cross_val_score(model, X, y, cv=rkf, scoring='accuracy', n_jobs=-1)
    return cv_scores.mean()

# Run cross-validation 30 times in parallel
cv_scores_means = Parallel(n_jobs=-1)(delayed(compute_cv_mean)() for _ in range(30))
chordata_df = merged_df[merged_df['Phylum'] == 'Chordata'].copy()   

# Precompute k-mer features for all sequences
kmer_features_chordata = []
for seq in chordata_df['sequentie']:
    sequence = str(seq)
    kmer_feature = get_kmers(sequence)
    kmer_features_chordata.append(kmer_feature)

# Convert k-mer features to a DataFrame
kmer_chordata = pd.DataFrame(kmer_features_chordata).fillna(0)

# Combine features with the merged DataFrame
df_chordata = pd.concat([chordata_df, kmer_chordata], axis=1)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Filter out columns with invalid k-mers
valid_kmer_columns = [column for column in kmer_chordata.columns if is_valid_kmer(column)]
filtered_kmer_chordata = kmer_df[valid_kmer_columns]

# Prepare the data
feature_columns_chordata = list(filtered_kmer_chordata.columns)  # Add more features as needed
X_chordata = df_chordata[feature_columns_chordata].to_numpy()  # Convert to NumPy array for faster operations
y_chordata = df_chordata['Mod'].to_numpy()  # Convert to NumPy array for faster operations

# Ensure there are no missing values
X_chordata = np.nan_to_num(X_chordata)

# Ensure there are no missing values in X_chordata and y_chordata
valid_indices = ~np.isnan(X_chordata).any(axis=1) & ~pd.isnull(y_chordata)
X_chordata = X_chordata[valid_indices]
y_chordata = y_chordata[valid_indices]

# Perform cross-validation 30 times and store the mean scores
cv_scores_means_chordata = []
rkf = RepeatedKFold(n_splits=7, n_repeats=3)  # Define cross-validation strategy
model = RandomForestClassifier(n_estimators=100)  # Define the model

def compute_cv_mean_chordata():
    cv_scores = cross_val_score(model, X_chordata, y_chordata, cv=rkf, scoring='accuracy', n_jobs=-1)
    return cv_scores.mean()

# Run cross-validation 30 times in parallel
cv_scores_means_chordata = Parallel(n_jobs=-1)(delayed(compute_cv_mean_chordata)() for _ in range(30))

# Create a combined boxplot of the 30 cv_scores_mean values for both datasets
plt.figure(figsize=(10, 6))
plt.boxplot([cv_scores_means, cv_scores_means_chordata], labels=['All Data', 'Chordata'])
plt.title('Boxplot of Repeated Cross-Validation Mean Scores')
plt.ylabel('Mean Accuracy')
plt.show()