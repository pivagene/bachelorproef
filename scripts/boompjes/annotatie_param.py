import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os
from collections import Counter
script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)
# Load the data
sequences_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_seq_COX1_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_COX1_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')

# Load the AMP collection and parameter data
df_AMP_collection = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_collection.csv'))
df_parameter = df_AMP_collection[df_AMP_collection['Description'] == 'ultimate wet weight'].copy()
df_parameter['Observed_log'] = np.log(df_parameter['Observed'])
df_parameter_selection = df_parameter[['Data', 'Observed_log', 'Predicted', 'Species', 'Unit']]
df_parameter_selection = df_parameter_selection.rename(columns={"Species": "ID"})
merged_df_par = pd.merge(df_parameter_selection, merged_df, on='ID', how='inner')

# Extract features from the FASTA sequences
kmer_features = []
# Process each sequence one by one
for seq in merged_df_par['sequentie']:
    sequence = str(seq)
    kmer_feature = get_kmers(sequence)
    kmer_features.append(kmer_feature)

# Convert kmer_features (list of Counter objects) into a DataFrame
kmer_df = pd.DataFrame(kmer_features)

def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Filter out columns with invalid k-mers
valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
filtered_kmer_df = kmer_df[valid_kmer_columns]

# Prepare features and target
df = pd.concat([merged_df_par, filtered_kmer_df], axis=1)

# Specify feature columns and target
feature_columns = list(filtered_kmer_df.columns)  # Add k-mer or other features here
X = df[feature_columns]
y = df['Observed_log']

# Split the data
X = X.fillna(0)
y = y.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

# train the model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2, random_state=42)
model.fit(X_train, y_train)
#joblib.dump(model, model_path)

# Predict on the entire dataset
df['Predicted'] = model.predict(X)

# Normalize the prediction error for gradient mapping
df['prediction_error'] = abs(df['Observed_log'] - df['Predicted'])
df['normalized_error'] = (df['prediction_error'] - df['prediction_error'].min()) / (df['prediction_error'].max() - df['prediction_error'].min())

# Define a function to map normalized values to a harsher gradient of hex colors
def normalized_to_color(value):
    value = value ** 2   # Apply harsher scaling by squaring the normalized value
    red = int(255 * value)
    green = int(255 * (1 - value))
    blue = 0
    return f"#{red:02X}{green:02X}{blue:02X}"

# Create the annotation file for iTOL
annotation_file = os.path.join(script_dir, 'COX1_gradient_annotations_ultimate_wet_weight.txt')
all_ids = set(merged_df['ID'])  # Get all IDs from the merged dataset
predicted_ids = set(df['ID'])  # Get IDs with predictions

with open(annotation_file, 'w', encoding='utf-8') as f:
    f.write('DATASET_COLORSTRIP\n')
    f.write('SEPARATOR TAB\n')
    f.write('DATASET_LABEL\tPrediction Error Gradient\n')
    f.write('COLOR\t#FF0000\n')
    f.write('LEGEND_SHAPES\t1\t1\n')
    f.write('LEGEND_TITLE\tPrediction Error\n')
    f.write('LEGEND_COLORS\t#FF0000\t#00FF00\n')  
    f.write('LEGEND_LABELS\tHigh Error\tLow Error\n')
    f.write('DATA\n')

    for animal_id in all_ids:
        if animal_id in predicted_ids:
            row = df[df['ID'] == animal_id].iloc[0]
            color = normalized_to_color(row['normalized_error'])
        else:
            color = "#808080"  # Default gray color for missing predictions
        id_with_quotes = f"'{animal_id.replace('_', ' ')}'"  # Add single quotes and replace underscores with spaces
        f.write(f"{id_with_quotes}\t{color}\n")
