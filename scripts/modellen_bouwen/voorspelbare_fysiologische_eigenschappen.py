import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    
    # Check if max/min > 1000 for the current parameter
    max_value = summary_df.loc[i, 'Max']
    min_value = summary_df.loc[i, 'Min']
    if max_value / min_value > 1000:
        # Use log transformation if the condition is met
        df_parameter['Target'] = np.log(df_parameter['Observed'])
        use_log = True
    else:
        # Use the raw values otherwise
        df_parameter['Target'] = df_parameter['Observed']
        use_log = False
    
    df_parameter_selection = df_parameter[['Description', 'Target', 'Predicted', 'Species', 'Unit', 'sequentie']]
    df_parameter_selection = df_parameter_selection.rename(columns={"Species": "ID"})

    # Get k-mer features
    kmer_features = Parallel(n_jobs=-1)(delayed(get_kmers)(str(seq)) for seq in df_parameter_selection['sequentie'])

    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)

    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]

    # Combine features with the parameter DataFrame
    df_parameter_selection = df_parameter_selection.reset_index(drop=True)
    filtered_kmer_df = filtered_kmer_df.reset_index(drop=True)

    # Concatenate the DataFrames
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
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2) 
    # model = RandomForestRegressor(n_estimators= 200, max_depth=5, random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True)

    # Cross-validation for R^2, MSE, and MAE
    cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    cv_mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')  # Convert negative MSE to positive

    # Calculate RMSE from MSE
    cv_rmse_scores = np.sqrt(cv_mse_scores)

    # Append the results to the list
    return {
        'Parameter': parameter_description,
        'MSE': cv_mse_scores.mean(),
        'RMSE': cv_rmse_scores.mean(),
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


# Plot R^2 scores as bars and data counts as a line graph
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the R^2 scores as bars on the primary y-axis
bars = ax1.bar(results_df['Parameter'], results_df['R^2'], color='skyblue', label='R^2 Score')

# Add labels and title for the primary y-axis
ax1.set_xlabel('Parameter', fontsize=12)
ax1.set_ylabel('R^2 Score', fontsize=12, color='blue')
ax1.set_title('R^2 Scores and Data Counts for Each Parameter', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(range(len(results_df['Parameter'])))
ax1.set_xticklabels(results_df['Parameter'], rotation=45, ha='right', fontsize=10)

# Annotate each bar with the R^2 score
for bar, r2_score in zip(bars, results_df['R^2']):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of the bar)
        bar.get_height() + 0.01,           # Y-coordinate (slightly above the bar)
        f'{r2_score:.2f}',                 # Text to display (R^2 score)
        ha='center',                       # Horizontal alignment
        va='bottom',                       # Vertical alignment
        fontsize=10,                       # Font size
        color='blue'                       # Text color
    )

# Create a secondary y-axis for the data count
ax2 = ax1.twinx()
ax2.plot(results_df['Parameter'], results_df['amount of data'], color='orange', marker='o', label='Data Count')

# Add labels for the secondary y-axis
ax2.set_ylabel('Data Count', fontsize=12, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add a legend for both axes
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()