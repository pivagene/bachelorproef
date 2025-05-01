import pandas as pd
import os
import numpy as np
from collections import Counter
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Load the cleaned CSV file
script_dir = os.path.dirname(__file__)
cleaned_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_collection_cleaned.csv'))
cox1_df = pd.read_csv('c:\\Users\\devos\\OneDrive - UGent\\3de bach\\bachelorproef\\Bachelorproef_AMP\\python\\bachelorproef github\\csv_files\\AMP_species_list_COX1.csv')
collection_df = pd.read_csv('c:\\Users\\devos\\OneDrive - UGent\\3de bach\\bachelorproef\\Bachelorproef_AMP\\python\\bachelorproef github\\csv_files\\AMP_collection_cleaned.csv')

# Replace underscores with spaces in the 'Species' column of the collection DataFrame
collection_df['Species'] = collection_df['Species'].str.replace('_', ' ')

# Merge the two DataFrames on 'ScientificName' and 'Species'
df_merged = pd.merge(cox1_df, collection_df, left_on='ScientificName', right_on='Species', how='inner')

# Filter the DataFrame for the "maximum reproduction rate" parameter
parameter = "maximum reproduction rate"
df_parameter = df_merged[df_merged['Description'] == parameter].copy()

# Apply log transformation if needed
df_parameter['Target'] = np.log(df_parameter['Observed'])

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Extract k-mer features in parallel
kmer_features = Parallel(n_jobs=-1)(delayed(get_kmers)(str(seq)) for seq in df_parameter['sequentie'])
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Filter valid k-mer columns
valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
filtered_kmer_df = kmer_df[valid_kmer_columns]

# Combine k-mer features with the target
df_parameter = df_parameter.reset_index(drop=True)
filtered_kmer_df = filtered_kmer_df.reset_index(drop=True)
df = pd.concat([df_parameter, filtered_kmer_df], axis=1)

# Prepare the data
X = df[filtered_kmer_df.columns]  # Use k-mer features as X
y = df['Target']

# Ensure there are no missing values
X = X.fillna(0)
y = y.fillna(0)

# Function to perform cross-validation for a given data size
def evaluate_data_size(size, repeats=10):
    scores = []
    for _ in range(repeats):
        # Randomly sample the data
        sampled_df = df.sample(n=size, random_state=None)  # Random sampling without fixed seed
        X_sample = sampled_df[filtered_kmer_df.columns]
        y_sample = sampled_df['Target']

        # Train the GradientBoostingRegressor model
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2)

        # Perform cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=None)
        cv_score = cross_val_score(model, X_sample, y_sample, cv=kf, scoring='r2', n_jobs=-1).mean()
        scores.append(cv_score)
    return scores

# Evaluate CV scores for each data size in parallel
data_sizes = range(100, 801, 100)  # Data sizes from 100 to 800
results = Parallel(n_jobs=-1)(
    delayed(evaluate_data_size)(size, repeats=5) for size in data_sizes
)

# Create a boxplot for the results
plt.figure(figsize=(12, 6))
plt.boxplot(results, labels=data_sizes, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.xlabel('Amount of Data Used', fontsize=12)
plt.ylabel('Cross-Validation Score (R²)', fontsize=12)
plt.title(f'CV Score Evolution for "{parameter}"', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()