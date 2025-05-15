import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


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
merged_df = merged_df.dropna(subset=['kap'])  # Drop rows where 'kap' is NaN
#merged_df['kap'] = merged_df['kap'].astype('category').cat.codes

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
y = df['kap']  # Target variable
y_log = np.log(y) 

# Train the XGBoost model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.01, max_depth=2)
odel = RandomForestClassifier(n_estimators=200, max_depth=5, n_jobs=-1)

# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True)
cv_scores = cross_val_score(model, X, y, cv=kf)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {cv_scores.mean()}")

results = []

results.append({
    'Parameter': 'kap',
    'score': cv_scores.mean(),
    'info': 'geen log gebruikt voor regressie',
})
results_df = pd.DataFrame(results)

results_df.to_csv(os.path.join(script_dir, '../../csv_files/kernparameter_modellen.csv'), index=False)

results_df = results_df.drop(14)
#parameters = [,,,,,,,,,,,,,,]

# log gebruikt voor regressie
# behandeld als een classificatie probleem

df = pd.read_csv(os.path.join(script_dir, '../../csv_files/kernparameter_modellen.csv'))




# Define colors for each category in the 'info' column
color_mapping = {
    'log gebruikt voor regressie': 'blue',
    'geen log gebruikt voor regressie': 'skyblue',
    'behandeld als een classificatie probleem': 'darkblue',
}
df['color'] = df['info'].map(color_mapping)

# Create the bar graph
plt.figure(figsize=(12, 6))
bars = plt.bar(df['Parameter'], df['score'], color=df['color'], edgecolor='black')

# Add the score as text above each bar
for bar, score in zip(bars, df['score']):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Center of the bar
        bar.get_height() + 0.01,           # Slightly above the bar
        f'{score:.2f}',                    # Format the score to 2 decimal places
        ha='center',                       # Horizontal alignment
        va='bottom',                       # Vertical alignment
        fontsize=10                        # Font size
    )

# Add labels, title, and legend
plt.xlabel('Parameter', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Scores for Kernparameters', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color, edgecolor='black') for color in color_mapping.values()],
           labels=color_mapping.keys(),
           title='Info',
           fontsize=10)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


#kap met grote korrel zout nemen (bestaat uit 37 verschillende waarden waarvan er 36 1 keer lijken voor te komen en de andere voor alle andere species)
#idem voor kap



# Get unique values and their counts in the F_m column
unique_values = merged_df['kap'].value_counts()

# Print the result
print(unique_values)
