import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import numpy as np
from xgboost import XGBRegressor
from joblib import Parallel, delayed

## samenvatten van parameters in merged_data---------------------------------------------------------------

script_dir = os.path.dirname(__file__)

# Load the merged CSV file
df_merged = pd.read_csv(os.path.join(script_dir, '../../csv_files/merged_data.csv'))
df_merged['Description'] = df_merged['Description'].str.lower()
df_merged['Description'] = df_merged['Description'].str.replace('reprod', 'reproduction', regex=False)


# Get the unique values in the 'Description' column and count their occurrences
parameter_counts = df_merged['Description'].value_counts()

# Filter the DataFrame to include only rows where Phylum is 'Chordata'
# df_merged = df_merged[df_merged['Phylum'] == 'Chordata']

# Filter out parameters with less than 50 occurrences
filtered_parameters = parameter_counts[parameter_counts >= 100].index

# Filter the DataFrame to include only the filtered parameters
filtered_df = df_merged[df_merged['Description'].isin(filtered_parameters)]

# Calculate the maximum and minimum values for each filtered parameter
parameter_stats = filtered_df.groupby('Description')['Observed'].agg(['min', 'max'])

# Combine the counts and stats into a single DataFrame
summary_df = parameter_counts[parameter_counts >= 100].to_frame().join(parameter_stats)

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

# Initialize a list to store the results
results = []

# Function to process each parameter
def process_parameter(i):
    parameter_description = summary_df['Description'][i]

    # Filter the merged DataFrame for the current parameter
    df_parameter = df_merged[df_merged['Description'] == parameter_description].copy()
    
    # Check if max/min ratio is less than 10,000
    max_value = summary_df.loc[i, 'Max']
    min_value = summary_df.loc[i, 'Min']
    use_log = (max_value / min_value) >= 10000

    # Use log-transformed or normal values based on the condition
    if use_log:
        df_parameter['Target'] = np.log(df_parameter['Observed'])
    else:
        df_parameter['Target'] = df_parameter['Observed']
    
    df_parameter_selection = df_parameter[['Description', 'Target', 'Predicted', 'Species', 'Unit', 'sequentie']]
    df_parameter_selection = df_parameter_selection.rename(columns={"Species": "ID"})

    # Get k-mer features
    kmer_features = Parallel(n_jobs=-1)(delayed(get_kmers)(str(seq)) for seq in df_parameter_selection['sequentie'])

    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)

    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]

    # Combine features with the parameter DataFrame
    df = pd.concat([df_parameter_selection, filtered_kmer_df], axis=1)

    # Prepare the data
    feature_columns = list(filtered_kmer_df.columns)
    X = df[feature_columns]
    y = df['Target']

    # Calculate the actual number of unique data points
    datacount = len(df_parameter_selection)

    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Train the model and perform cross-validation
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2)
    kf = KFold(n_splits=5, shuffle=True)
    cv_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    cv_mse_scores = -cv_mse_scores  # Convert negative MSE to positive

    # Append the results to the list
    return {
        'Parameter': parameter_description,
        'MSE': cv_mse_scores.mean(),
        'R^2': cv_r2_scores.mean(),
        'amount of data': datacount,
        'use_log': use_log  
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
colors = ['skyblue' if use_log else 'orange' for use_log in results_df['use_log']]
plt.bar(results_df['Parameter'], results_df['R^2'], color=colors)
plt.xlabel('Parameter')
plt.ylabel('R^2 Score')
plt.title('R^2 Scores for Each Parameter')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Sort the DataFrame by R^2 score
sorted_results_df = results_df.sort_values(by='R^2')

# Plot the R^2 scores and datacount as a dual-axis bar graph
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the R^2 scores on the primary y-axis
colors_sorted = ['skyblue' if use_log else 'orange' for use_log in sorted_results_df['use_log']]
ax1.bar(sorted_results_df['Parameter'], sorted_results_df['R^2'], color=colors_sorted, label='R^2 Score')
ax1.set_xlabel('Parameter')
ax1.set_ylabel('R^2 Score', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(range(len(sorted_results_df['Parameter'])))
ax1.set_xticklabels(sorted_results_df['Parameter'], rotation=90)

# Create a secondary y-axis for datacount
ax2 = ax1.twinx()
ax2.bar(sorted_results_df['Parameter'], sorted_results_df['amount of data'], color='gray', label='amount of data', alpha=0.5)
ax2.set_ylabel('amount of data', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# Add a title and layout adjustments
fig.suptitle('R^2 Scores and amount of data for Each Parameter')
fig.tight_layout()

# Add a legend
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

# Show the plot
plt.show()