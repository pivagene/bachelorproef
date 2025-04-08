import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import numpy as np
from xgboost import XGBRegressor
from joblib import Parallel, delayed

## samenvatten van parameters in AMP_collection---------------------------------------------------------------

script_dir = os.path.dirname(__file__)

# Load the CSV file
df_AMP_collection = pd.read_csv(os.path.join(script_dir, '../csv_files/AMP_collection.csv'))
df_AMP_collection['Description'] = df_AMP_collection['Description'].str.lower()
df_AMP_collection['Description'] = df_AMP_collection['Description'].str.replace('reprod', 'reproduction', regex=False)

# Get the unique values in the first column and count their occurrences
parameter_counts = df_AMP_collection['Description'].value_counts()

# Filter out parameters with less than 50 occurrences
filtered_parameters = parameter_counts[parameter_counts >= 1].index

# Filter the DataFrame to include only the filtered parameters
filtered_df = df_AMP_collection[df_AMP_collection['Data'].isin(filtered_parameters)]

# Calculate the maximum and minimum values for each filtered parameter
parameter_stats = filtered_df.groupby('Data')['Observed'].agg(['min', 'max'])

# Combine the counts and stats into a single DataFrame
summary_df = parameter_counts[parameter_counts >= 50].to_frame().join(parameter_stats)

# Rename the columns for clarity
summary_df.columns = ['Count', 'Min', 'Max']

# Reset the index to make the parameter names a column
summary_df = summary_df.reset_index()

# Print the results
print("Filtered Parameter Summary (Parameter, Count, Min, Max):")
print(summary_df)

## define functions and read csv's------------------------------------------------------------------------------------

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_seq_COX1_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_IDS_COX1_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'

# Initialize a list to store the results
results = []

# Function to process each parameter
def process_parameter(i):
    parameter_name = summary_df['Data'][i]

    # Merge dataframes with parameters dataframes
    df_parameter = df_AMP_collection[df_AMP_collection['Data'] == parameter_name].copy()
    df_parameter['Observed_log'] = np.log(df_parameter['Observed'])
    df_parameter_selection = df_parameter[['Data', 'Observed_log', 'Predicted', 'Species', 'Unit']]
    df_parameter_selection = df_parameter_selection.rename(columns={"Species": "ID"})
    merged_df_par = pd.merge(df_parameter_selection, merged_df, on='ID', how='inner')

    # Extract features from the FASTA sequences
    kmer_features = Parallel(n_jobs=-1)(delayed(get_kmers)(str(seq)) for seq in merged_df_par['sequentie'])

    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)

    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]

    # Combine features with the merged DataFrame
    df = pd.concat([merged_df_par, filtered_kmer_df], axis=1)

    # Prepare the data
    # Specify the columns you want to use for X
    feature_columns = list(filtered_kmer_df.columns)
    X = df[feature_columns]
    y = df['Observed_log'] 

    datacount = len(y)
    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Train the model and perform cross-validation
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    cv_mse_scores = -cv_mse_scores  # Convert negative MSE to positive

    # Append the results to the list
    return {
        'Parameter': parameter_name,
        'MSE': cv_mse_scores.mean(),
        'R^2': cv_r2_scores.mean(),
        'amount of data': datacount
    }

# Process each parameter in parallel
results = Parallel(n_jobs=-1)(delayed(process_parameter)(i) for i in range(len(summary_df)))

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Print the results DataFrame
print("Results DataFrame:")
print(results_df)

# Plot the R^2 scores as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(results_df['Parameter'], results_df['R^2'], color='skyblue')
plt.xlabel('Parameter')
plt.ylabel('R^2 Score')
plt.title('R^2 Scores for Each Parameter')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Merge the results_df with summary_df to add the Count column
results_df = results_df.merge(summary_df[['Data','Count']], left_on='Parameter', right_on='Data', how='left')
results_df = results_df.drop(columns='Data')

# Filter out parameters with less than 300 values
filtered_results_df = results_df[results_df['Count'] >= 300]

# Sort the DataFrame by R^2 score
sorted_results_df = filtered_results_df.sort_values(by='R^2')


# Plot the R^2 scores and datacount as a dual-axis bar graph
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the R^2 scores on the primary y-axis
ax1.bar(results_df['Parameter'], results_df['R^2'], color='skyblue', label='R^2 Score', alpha=0.7)
ax1.set_xlabel('Parameter')
ax1.set_ylabel('R^2 Score', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(range(len(results_df['Parameter'])))
ax1.set_xticklabels(results_df['Parameter'], rotation=90)

# Create a secondary y-axis for datacount
ax2 = ax1.twinx()
ax2.bar(results_df['Parameter'], results_df['amount of data'], color='orange', label='amount of data', alpha=0.5)
ax2.set_ylabel('amount of data', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add a title and layout adjustments
fig.suptitle('R^2 Scores and amount of data for Each Parameter')
fig.tight_layout()

# Add a legend
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

# Show the plot
plt.show()