import pandas as pd
import os
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
script_dir = os.path.dirname(__file__)

 # Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_seq_12SrRNA_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../csv_files/gene_IDS_12SrRNA_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'
merged_df.to_csv(os.path.join(script_dir, '../csv_files/AMP_species_list_12SrRNA.csv'), index=False)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k):
    kmers = [sequence[j:j+k] for j in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)



mean_cv_scores = []

kmer_lengths =[2]*30 + [3]*30 + [4]*30 + [5]*30 + [6]*30

for i in kmer_lengths:
    features = []
    kmer_features = []
    # Process each sequence one by one
    for seq in merged_df['sequentie']:
        sequence = str(seq)
        kmer_feature = get_kmers(sequence,i)
        kmer_features.append(kmer_feature)
    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)
    # Filter out columns with invalid k-mers
    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]
    # Combine features with the merged DataFrame
    df = pd.concat([merged_df, filtered_kmer_df], axis=1)

    # Prepare the data
    # Assuming 'Characteristic' is the column you want to predict
    # Specify the columns you want to use for X
    feature_columns = [] + list(filtered_kmer_df.columns)  # Add more features as needed length and gc doen ni veel kmer wint zelf zonder blast score
    X = df[feature_columns]
    y = df['Mod']  # Replace with your target column

    # Ensure there are no missing values
    X = X.fillna(0)

    # Split the data into training and testing sets
    
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100)
    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=37)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    mean_cv_scores.append(cv_scores.mean())

cross_validation_df = pd.DataFrame({'kmer_length': kmer_lengths, 'mean_cv_scores': mean_cv_scores})

## Plot the boxplot 
plt.figure(figsize=(10, 6))
cross_validation_df.boxplot(column='mean_cv_scores', by='kmer_length', grid=False)
plt.title('Boxplot of Mean CV Scores for Different k-mer Lengths')
plt.suptitle('')  # Suppress the default title to avoid duplication
plt.xlabel('k-mer Length')
plt.ylabel('Mean CV Score')
plt.show()

# Save the boxplot 
plt.savefig(os.path.join(script_dir, '../../grafieken/boxplot_mean_cv_scores.png'), bbox_inches='tight')
