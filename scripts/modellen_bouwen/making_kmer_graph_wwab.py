import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import numpy as np

# Load the cleaned CSV file
script_dir = os.path.dirname(__file__)
df_merged = pd.read_csv(os.path.join(script_dir, '../../csv_files/merged_data_cleaned.csv'))

# Filter the DataFrame for the selected description
filtered_df = df_merged[df_merged['Description'] == "weight at birth"]

# Ensure the Unit is consistent for each description
def filter_by_most_abundant_unit(df):
    most_common_units = df.groupby('Description')['Unit'].agg(lambda x: x.value_counts().idxmax())
    return df[df['Unit'] == most_common_units[df['Description'].iloc[0]]]

filtered_df = filter_by_most_abundant_unit(filtered_df)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k):
    return Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Precompute valid k-mers for filtering
def filter_valid_kmers(kmer_df):
    valid_kmer_columns = [col for col in kmer_df.columns if is_valid_kmer(col)]
    valid_kmer_columns.append('Species')  # Ensure 'Species' is preserved
    return kmer_df[valid_kmer_columns]

# Initialize a list to store results for each k-mer length
results = []


kmer_lengths = [2, 3, 4, 5, 6, 7]
# Loop through k-mer lengths from 2 to 6
for k in kmer_lengths:
    print(f"Processing k-mer length: {k}")
    
    # Add the Target column
    filtered_df['Target'] = np.log(filtered_df['Observed'])
    
    # Select relevant columns
    df_parameter_selection = filtered_df[['Description', 'Target', 'Species', 'Unit', 'sequentie']]
    
    # Group by 'Species' and calculate the mean of 'Target'
    df_parameter_selection = (
        df_parameter_selection.groupby('Species', as_index=False)
        .agg({
            'Description': 'first',  # Keep the first value (all should be identical)
            'Target': 'mean',        # Calculate the mean of the 'Target' column
            'Unit': 'first',         # Keep the first value (all should be identical)
            'sequentie': 'first'     # Keep the first value (all should be identical)
        })
    )
    
    # Optimize k-mer extraction using a single function
    def process_sequence(seq, species):
        kmer_counts = get_kmers(str(seq), k)
        kmer_counts['Species'] = species
        return kmer_counts

    # Use Parallel processing for k-mer extraction
    kmer_features = Parallel(n_jobs=-1)(
        delayed(process_sequence)(seq, species)
        for seq, species in zip(df_parameter_selection['sequentie'], df_parameter_selection['Species'])
    )
    
    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)
    
    # Filter valid k-mers
    filtered_kmer_df = filter_valid_kmers(kmer_df)
    
    # Combine features with the parameter DataFrame
    df = pd.merge(df_parameter_selection, filtered_kmer_df, on='Species', how='inner')
    
    # Prepare the data
    X = df.drop(columns=['Species', 'Target', 'Description', 'Unit', 'sequentie'])
    y = df['Target']
    
    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Train the model and perform cross-validation
    model = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=2)
    kf = KFold(n_splits=2, shuffle=True)
    
    # Cross-validation for R^2
    cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    # Store the results
    results.append({
        'k-mer Length': k,
        'R^2': cv_r2_scores.mean(),
        'Data Count': len(df_parameter_selection)
    })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Print the results
print(results_df)

# Save results to a CSV file
results_df.to_csv(os.path.join(script_dir, '../../csv_files/R2_weight_at_birth.csv'), index=False)


# Initialize lists to store results
results = []
mean_training_scores = []  # To store training R^2 scores
mean_validation_scores = []  # To store validation R^2 scores

kmer_lengths = [2, 3, 4, 5, 6, 7]
# Loop through k-mer lengths from 2 to 6
for k in kmer_lengths:
    print(f"Processing k-mer length: {k}")
    
    # Add the Target column
    filtered_df['Target'] = np.log(filtered_df['Observed'])
    
    # Select relevant columns
    df_parameter_selection = filtered_df[['Description', 'Target', 'Species', 'Unit', 'sequentie']]
    
    # Group by 'Species' and calculate the mean of 'Target'
    df_parameter_selection = (
        df_parameter_selection.groupby('Species', as_index=False)
        .agg({
            'Description': 'first',  # Keep the first value (all should be identical)
            'Target': 'mean',        # Calculate the mean of the 'Target' column
            'Unit': 'first',         # Keep the first value (all should be identical)
            'sequentie': 'first'     # Keep the first value (all should be identical)
        })
    )
    
    # Optimize k-mer extraction using a single function
    def process_sequence(seq, species):
        kmer_counts = get_kmers(str(seq), k)
        kmer_counts['Species'] = species
        return kmer_counts

    # Use Parallel processing for k-mer extraction
    kmer_features = Parallel(n_jobs=-1)(
        delayed(process_sequence)(seq, species)
        for seq, species in zip(df_parameter_selection['sequentie'], df_parameter_selection['Species'])
    )
    
    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)
    
    # Filter valid k-mers
    filtered_kmer_df = filter_valid_kmers(kmer_df)
    
    # Combine features with the parameter DataFrame
    df = pd.merge(df_parameter_selection, filtered_kmer_df, on='Species', how='inner')
    
    # Prepare the data
    X = df.drop(columns=['Species', 'Target', 'Description', 'Unit', 'sequentie'])
    y = df['Target']
    
    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Perform cross-validation
    kf = KFold(n_splits=2, shuffle=True)
    training_scores = []  # To store training R^2 for each fold
    validation_scores = []  # To store validation R^2 for each fold

    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2)
        model.fit(X_train, y_train)

        # Calculate training R^2
        training_r2 = model.score(X_train, y_train)
        training_scores.append(training_r2)

        # Calculate validation R^2
        validation_r2 = model.score(X_test, y_test)
        validation_scores.append(validation_r2)

    # Store the mean training and validation R^2 for this k-mer length
    mean_training_scores.append(np.mean(training_scores))
    mean_validation_scores.append(np.mean(validation_scores))

    # Store the results
    results.append({
        'k-mer Length': k,
        'Training R^2': np.mean(training_scores),
        'Validation R^2': np.mean(validation_scores),
        'Data Count': len(df_parameter_selection)
    })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Print the results
print(results_df)

# Save results to a CSV file
results_df.to_csv(os.path.join(script_dir, '../../csv_files/R2_training_vs_validation.csv'), index=False)

# Plot training vs validation R^2
plt.figure(figsize=(10, 6))
plt.plot(results_df['k-mer Length'], results_df['Training R^2'], label='Training R^2', marker='o', color='blue')
plt.plot(results_df['k-mer Length'], results_df['Validation R^2'], label='Validation R^2', marker='o', color='orange')
plt.title('Training vs Validation R^2 for Different k-mer Lengths')
plt.xlabel('k-mer Length')
plt.ylabel('R^2 Score')
plt.legend()
plt.grid()
plt.show()