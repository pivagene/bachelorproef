import pandas as pd
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed

script_dir = os.path.dirname(__file__)

# Load the data
sequences_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_seq_12SrRNA_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_12SrRNA_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')

# Prepare features and target
def get_kmers(sequence, k=5):
    return {sequence[i:i+k]: 1 for i in range(len(sequence) - k + 1)}

kmer_features = [get_kmers(str(seq)) for seq in merged_df['sequentie']]
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Filter valid k-mers
valid_kmer_columns = [col for col in kmer_df.columns if set(col).issubset({'A', 'C', 'T', 'G'})]
filtered_kmer_df = kmer_df[valid_kmer_columns]

# Combine features with the merged DataFrame
df = pd.concat([merged_df.reset_index(drop=True), filtered_kmer_df.reset_index(drop=True)], axis=1)
df = df.dropna(subset=['Mod'])  # Ensure no missing target values

X = df[filtered_kmer_df.columns]
y = df['Mod']

print("Starting Leave-One-Out Cross-Validation (LOOCV)...")
# Perform LOOCV with parallel processing
def train_and_predict(train_index, test_index):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(n_estimators=100, random_state=37, n_jobs=-1)  # Use all cores for training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return df.iloc[test_index[0]]['ID'], y_test.values[0], y_pred[0]

loo = LeaveOneOut()
results = Parallel(n_jobs=-1)(
    delayed(train_and_predict)(train_index, test_index) 
    for i, (train_index, test_index) in enumerate(loo.split(X), start=1)
    if not print(f"Processing sample {i}/{len(X)}...")
)

print("LOOCV completed. Saving predictions...")
# Create a DataFrame for predictions
results_df = pd.DataFrame(results, columns=['ID', 'True_Label', 'Predicted_Label'])

# Save predictions to a CSV file
results_csv = os.path.join(script_dir, '../../csv_files/LOOCV_predictions_12SrRNA.csv')
results_df.to_csv(results_csv, index=False)

# Calculate accuracy of predictions
correct_predictions = results_df[results_df['True_Label'] == results_df['Predicted_Label']]
accuracy = len(correct_predictions) / len(results_df)

print(f"Accuracy of LOOCV predictions: {accuracy:.2%}")


# Define a function to map models to colors
def model_to_color(model):
    color_mapping = {
        'std': '#FF0000',  # Red
        'abj': '#00FF00',  # Green
        'stx': '#0000FF',  # Blue
        'stf': '#FFFF00',  # Yellow
        'abp': '#FF00FF',  # Magenta
        'hex': '#00FFFF',  # Cyan
        'ssj': '#800000',  # Maroon
        'hep': '#808000',  # Olive
        'hax': '#008000',  # Dark Green
        'asj': '#800080',  # Purple
        'sbp': '#000080'   # Navy
    }
    return color_mapping.get(model, '#000000')  # Default to black if model is not in the mapping

# Create the annotation file
annotation_file = os.path.join(script_dir, '12SrRNA_LOOCV_model_annotations.txt')

with open(annotation_file, 'w') as f:
    f.write('DATASET_COLORSTRIP\n')
    f.write('SEPARATOR TAB\n')
    f.write('DATASET_LABEL\tModel Predictions\n')
    f.write('COLOR\t#FF0000\n')
    f.write('LEGEND_TITLE\tModel\n')
    f.write('LEGEND_SHAPES\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\n')
    f.write('LEGEND_COLORS\t#FF0000\t#00FF00\t#0000FF\t#FFFF00\t#FF00FF\t#00FFFF\t#800000\t#808000\t#008000\t#800080\t#000080\n')
    f.write('LEGEND_LABELS\tstd\tabj\tstx\tstf\tabp\thex\tssj\thep\thax\tasj\tsbp\n')
    f.write('BORDER_WIDTH\t1\n')
    f.write('BORDER_COLOR\t#000000\n')
    f.write('DATA\n')

    for index, row in results_df.iterrows():
        color = model_to_color(row['Predicted_Label'])  # Map the predicted model to a color
        id_with_quotes = f"'{row['ID'].replace('_', ' ')}'"  # Add single quotes and replace underscores with spaces
        f.write(f"{id_with_quotes}\t{color}\n")
