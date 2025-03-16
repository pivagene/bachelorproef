
import pandas as pd
import subprocess
import os
from Bio.Blast import NCBIXML
from collections import Counter
import io
import threading
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import numpy as np
import joblib
import xgboost as xgb
from xgboost import XGBRegressor

script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_seq_COX1_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_COX1_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'
merged_df.to_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list_COX1.csv'), index=False)

# Merge dataframes with parameters dataframes
df_AMP_collection = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_collection.csv'))
df_parameter = df_AMP_collection[df_AMP_collection['Data'] == 'tR'].copy()
df_parameter.loc[:,'Observed_log'] = np.log(df_parameter['Observed'])
df_parameter_selection = df_parameter[['Data', 'Observed_log', 'Predicted', 'Species', 'Unit']]
# df_parameter_selection = df_parameter[['Data','Observed','Predicted','Species','Unit']]
df_parameter_selection = df_parameter_selection.rename(columns={"Species": "ID"})
merged_df_par = pd.merge(df_parameter_selection, merged_df, on='ID', how='inner')

# Extract features from the FASTA sequences
features = []
kmer_features = []
# blast_scores = []
# bit_scores = []

# Process each sequence one by one
for seq in merged_df_par['sequentie']:
    sequence = str(seq)
    kmer_feature = get_kmers(sequence)
    kmer_features.append(kmer_feature)

# Convert features to a DataFrame
features_df = pd.DataFrame(features)
# features_df = pd.read_csv("blast_scores.csv") als je de blast al hebt gedaan
# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Add BLAST scores and bit scores to the features DataFrame
# features_df['blast_score'] = blast_scores
# features_df['bit_score'] = bit_scores
# features_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/blast_scores_COX1_1db.csv'))
# Combine features with the merged DataFrame
df = pd.concat([merged_df_par, features_df, kmer_df], axis=1)
df = pd.get_dummies(df, columns=['Mod'])

# Prepare the data
# Assuming 'Characteristic' is the column you want to predict
# Specify the columns you want to use for X
feature_columns = [] + list(kmer_df.columns)  # Add more features as needed length and gc doen ni veel kmer wint zelf zonder blast score
X = df[feature_columns]
y = df['Observed_log']  # Replace with your target column

# Ensure there are no missing values
X = X.fillna(0)
y = y.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

# Train the Random Forest model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2, random_state=42)
# xgboost package gebruiken
# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
# Convert negative MSE to positive
cv_mse_scores = -cv_mse_scores

print(f'Cross-Validation Mean Squared Error: {cv_mse_scores.mean()}')
print(f'Cross-Validation Standard Deviation: {cv_mse_scores.std()}')

print(f'Cross-Validation R^2 Score: {cv_r2_scores.mean()}')
print(f'Cross-Validation Standard Deviation (R^2): {cv_r2_scores.std()}')


# Fit the model on the entire training set
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
med = median_absolute_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Median Absolute Error: {med}')

x = np.linspace(-15, 15, 1000)
y = x
plt.scatter(y_pred, y_test)
plt.plot(x, y)

# Optionally, save the model for future use
joblib.dump(model, os.path.join(script_dir, '../../other_files/random_forest_model_kmer_COX1.pkl'))

# Save the blast scores and bit scores for future use
features_df.to_csv(os.path.join(script_dir, '../../csv_files/blast_scores_12SrRNA_1db.csv'), index=False)

value_counts = df_AMP_collection['Data'].value_counts()
features_df.to_csv(os.path.join(script_dir, '../../csv_files/blast_scores_COX1_1db.csv'), index=False)