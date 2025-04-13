import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
import numpy as np
from xgboost import XGBRegressor
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Load the cleaned CSV file
script_dir = os.path.dirname(__file__)
df_merged = pd.read_csv(os.path.join(script_dir, '../../csv_files/merged_data_cleaned.csv'))

# Filter the DataFrame for the selected descriptions
filtered_df = df_merged[df_merged['Description']=="lifespan"]

# Ensure the Unit is consistent for each description
def filter_by_most_abundant_unit(df):
    most_common_units = df.groupby('Description')['Unit'].agg(lambda x: x.value_counts().idxmax())
    return df[df.apply(lambda row: row['Unit'] == most_common_units[row['Description']], axis=1)]

filtered_df = filter_by_most_abundant_unit(filtered_df)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)


filtered_df['Target'] = np.log(filtered_df['Observed'])
    
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

# Get k-mer features along with the Species column
kmer_features = Parallel(n_jobs=-1)(
    delayed(lambda seq, species: {**get_kmers(str(seq)), 'Species': species})(
        seq, species
    )
    for seq, species in zip(df_parameter_selection['sequentie'], df_parameter_selection['Species'])
)

# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Ensure the Species column is preserved
valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
valid_kmer_columns.append('Species')  # Include the Species column
filtered_kmer_df = kmer_df[valid_kmer_columns]

# Combine features with the parameter DataFrame
df = pd.merge(df_parameter_selection, filtered_kmer_df, on='Species', how='inner')

# Prepare the data
filtered_kmer_df = filtered_kmer_df.drop(columns=['Species'])  # Drop the Species column for X
feature_columns = list(filtered_kmer_df.columns)
X = df[feature_columns]
y = df['Target']

# Ensure there are no missing values
X = X.fillna(0)
y = y.fillna(0)


# Train the model and perform cross-validation
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2)
kf = KFold(n_splits=5, shuffle=True)

# Cross-validation for R^2, MSE, and MAE
cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')


parameter = 'lifespan',
R_2 = cv_r2_scores.mean()

print(R_2)



# List of descriptions to process
descriptions = [
    "lifespan", "weight at birth", "maximum reproduction rate", "ultimate weight",
    "age at birth", "ultimate total length", "time since birth at puberty",
    "total length at puberty", "weight at puberty", "time since birth at weaning",
    "gestation time", "ultimate weight female", "ultimate weight male",
    "total length at birth", "time since birth at fledging",
    "time since birth at 1st brood", "time since birth at puberty female"
]

# Initialize a list to store results for each description
results = []

# Loop through each description
for description in descriptions:
    # Filter the DataFrame for the current description
    filtered_df = df_merged[df_merged['Description'] == description]

    # Ensure the Unit is consistent for each description
    filtered_df = filter_by_most_abundant_unit(filtered_df)

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

    # Get k-mer features along with the Species column
    kmer_features = Parallel(n_jobs=-1)(
        delayed(lambda seq, species: {**get_kmers(str(seq)), 'Species': species})(
            seq, species
        )
        for seq, species in zip(df_parameter_selection['sequentie'], df_parameter_selection['Species'])
    )

    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)

    # Ensure the Species column is preserved
    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    valid_kmer_columns.append('Species')  # Include the Species column
    filtered_kmer_df = kmer_df[valid_kmer_columns]

    # Combine features with the parameter DataFrame
    df = pd.merge(df_parameter_selection, filtered_kmer_df, on='Species', how='inner')

    # Prepare the data
    filtered_kmer_df = filtered_kmer_df.drop(columns=['Species'])  # Drop the Species column for X
    feature_columns = list(filtered_kmer_df.columns)
    X = df[feature_columns]
    y = df['Target']

    # Ensure there are no missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Train the model and perform cross-validation
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=2)
    kf = KFold(n_splits=5, shuffle=True)

    # Cross-validation for R^2
    cv_r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    # Store the results
    results.append({
        'Description': description,
        'R^2': cv_r2_scores.mean(),
        'Data Count': len(df_parameter_selection)
    })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Print the results
print(results_df)



# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot the R^2 scores as bars on the primary y-axis
bars = ax1.bar(results_df['Description'], results_df['R^2'], color='skyblue', label='R^2 Score')

# Add labels and title for the primary y-axis
ax1.set_xlabel('Description', fontsize=12)
ax1.set_ylabel('R^2 Score', fontsize=12, color='blue')
ax1.set_title('R^2 Scores and Data Counts for Different Descriptions', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(range(len(results_df['Description'])))
ax1.set_xticklabels(results_df['Description'], rotation=45, ha='right', fontsize=10)

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
ax2.plot(results_df['Description'], results_df['Data Count'], color='orange', marker='o', label='Data Count')

# Add labels for the secondary y-axis
ax2.set_ylabel('Data Count', fontsize=12, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# Add a legend for both axes
fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()