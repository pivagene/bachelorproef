import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import numpy as np
from xgboost import XGBRegressor
from joblib import Parallel, delayed

# Load the cleaned CSV file
script_dir = os.path.dirname(__file__)
#df_merged = pd.read_csv(os.path.join(script_dir, '../../csv_files/merged_data_cleaned.csv'))
cleaned_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_collection_cleaned.csv'))
cox1_df = pd.read_csv('c:\\Users\\devos\\OneDrive - UGent\\3de bach\\bachelorproef\\Bachelorproef_AMP\\python\\bachelorproef github\\csv_files\\AMP_species_list_COX1.csv')
collection_df = pd.read_csv('c:\\Users\\devos\\OneDrive - UGent\\3de bach\\bachelorproef\\Bachelorproef_AMP\\python\\bachelorproef github\\csv_files\\AMP_collection_cleaned.csv')

# Replace underscores with spaces in the 'Species' column of the collection DataFrame
collection_df['Species'] = collection_df['Species'].str.replace('_', ' ')

# Merge the two DataFrames on 'ScientificName' and 'Species'
df_merged = pd.merge(cox1_df, collection_df, left_on='ScientificName', right_on='Species', how='inner')


# List of descriptions to include
selected_descriptions = [
    "lifespan", "weight at birth", "maximum reproduction rate", "ultimate weight",
    "age at birth", "ultimate total length", "time since birth at puberty",
    "total length at puberty", "weight at puberty", "time since birth at weaning",
    "gestation time", "ultimate weight female", "ultimate weight male",
    "total length at birth", "time since birth at fledging",
    "time since birth at 1st brood", "time since birth at puberty female"
]

# Filter the DataFrame for the selected descriptions
filtered_df = df_merged[df_merged['Description'].isin(selected_descriptions)]

# Ensure the Unit is consistent for each description
def filter_by_most_abundant_unit(df):
    most_common_units = df.groupby('Description')['Unit'].agg(lambda x: x.value_counts().idxmax())
    return df[df.apply(lambda row: row['Unit'] == most_common_units[row['Description']], axis=1)]

filtered_df = filter_by_most_abundant_unit(filtered_df)

# Get unique descriptions and their stats
parameter_counts = filtered_df['Description'].value_counts()
parameter_stats = filtered_df.groupby('Description')['Observed'].agg(['min', 'max'])

# Combine the counts and stats into a summary DataFrame
summary_df = parameter_counts.to_frame().join(parameter_stats)
summary_df.columns = ['Count', 'Min', 'Max']
summary_df = summary_df.reset_index().rename(columns={'index': 'Description'})

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Function to process each parameter
def process_parameter(i):
    parameter_description = summary_df['Description'][i]

    # Filter the DataFrame for the current parameter
    df_parameter = filtered_df[filtered_df['Description'] == parameter_description].copy()
    
    df_parameter['Target'] = np.log(df_parameter['Observed'])
    
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

    # Cross-validation for R^2, MSE, and MAE
    cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    cv_mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')  # Convert negative MSE to positive
    cv_mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')  # Convert negative MAE to positive

    # Calculate RMSE from MSE
    cv_rmse_scores = np.sqrt(cv_mse_scores)

    # Append the results to the list
    return {
        'Parameter': parameter_description,
        'MSE': cv_mse_scores.mean(),
        'RMSE': cv_rmse_scores.mean(),
        'MAE': cv_mae_scores.mean(),
        'R^2': cv_r2_scores.mean(),
        'amount of data': datacount,
        #'use_log': use_log  
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
#colors = ['skyblue' if use_log else 'orange' for use_log in results_df['use_log']]
plt.bar(results_df['Parameter'], results_df['R^2'], color= 'skyblue')
plt.xlabel('Parameter')
plt.ylabel('R^2 Score')
plt.title('R^2 Scores for Each Parameter')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



cleaned_df = df_merged[df_merged['Description'] == 'lifespan']
lifespan_data = cleaned_df[['Species', 'Observed','sequentie']]
# Prepare to store mean CV scores
mean_cv_scores = []

# Define k-mer lengths to test
kmer_lengths = [5]

for k in kmer_lengths:
    kmer_features = []
    
    # Extract k-mer features for each sequence
    for seq in lifespan_data['sequentie']:
        sequence = str(seq)
        kmer_feature = get_kmers(sequence, k)
        kmer_features.append(kmer_feature)
    
    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)
    
    # Filter out columns with invalid k-mers
    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]
    
    # Combine features with the merged DataFrame
    df = pd.concat([lifespan_data, filtered_kmer_df], axis=1)

    # Prepare the data
    feature_columns = list(filtered_kmer_df.columns)  # Use k-mer features as predictors
    X = df[feature_columns]
    y = df['Observed']  # Use 'Observed' as the target variable

    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Train the XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
    
    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')  # Use R² as the scoring metric
    mean_cv_scores.append(cv_scores.mean())