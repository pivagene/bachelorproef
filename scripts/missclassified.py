import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from collections import Counter

script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Function to process a gene dataset, ensuring the 'Mod' column is added
def process_gene_with_mod(sequences_file, ids_file, mod_column_file):
    sequences_df = pd.read_csv(os.path.join(script_dir, sequences_file))
    ids_df = pd.read_csv(os.path.join(script_dir, ids_file))
    mod_column_df = pd.read_csv(os.path.join(script_dir, mod_column_file))
    merged_df = pd.merge(sequences_df, ids_df, on='Gene_ID', how='inner')
    merged_df = pd.merge(merged_df, mod_column_df, on='ID', how='inner')
    kmer_features = []
    for seq in merged_df['sequentie']:
        sequence = str(seq)
        kmer_feature = get_kmers(sequence)
        kmer_features.append(kmer_feature)

    kmer_df = pd.DataFrame(kmer_features).fillna(0)
    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]

    df = pd.concat([merged_df, filtered_kmer_df], axis=1)
    df = df.dropna(subset=['Mod'])

    return df

# Ensure proper feature and label selection
def prepare_features_and_labels(df):
    feature_columns = [col for col in df.columns if col not in ['Gene_ID', 'ID', 'Mod', 'sequentie','Range','Phylum','Class','Order','Family','Genus','Species','ScientificName','CommonName','rest','SMSE','COM','MRE']]
    X = df[feature_columns]
    y = df['Mod']
    return X.reset_index(drop=True), y.reset_index(drop=True)  # Reset indices to ensure alignment

# Initialize dictionaries to track misclassified instances
misclassified_instances = {'COX1': {}, '12S rRNA': {}, 'mitofish': {}}

# Repeat the cross-validation process 20 times
for iteration in range(20):
    print(f"Starting cross-validation iteration {iteration + 1}...")

    # Process COX1
    cox1_df = process_gene_with_mod('../csv_files/gene_seq_COX1_final.csv', '../csv_files/gene_IDS_COX1_final.csv', '../csv_files/AMP_species_list.csv')
    X_cox1, y_cox1 = prepare_features_and_labels(cox1_df)
    cox1_model = RandomForestClassifier(n_estimators=100)
    rkf = RepeatedKFold(n_splits=7, n_repeats=3)
    for train_idx, test_idx in rkf.split(X_cox1):
        X_train, X_test = X_cox1.iloc[train_idx], X_cox1.iloc[test_idx]
        y_train, y_test = y_cox1.iloc[train_idx], y_cox1.iloc[test_idx]
        cox1_model.fit(X_train, y_train)
        predictions = cox1_model.predict(X_test)
        for idx, (true_label, pred_label) in enumerate(zip(y_test, predictions)):
            if true_label != pred_label:
                gene_id = cox1_df.iloc[test_idx[idx]]['ScientificName']
                misclassified_instances['COX1'][gene_id] = misclassified_instances['COX1'].get(gene_id, 0) + 1

    # Process rRNA
    rRNA_df = process_gene_with_mod('../csv_files/gene_seq_12SrRNA_final.csv', '../csv_files/gene_IDS_12SrRNA_final.csv', '../csv_files/AMP_species_list.csv')
    X_rRNA, y_rRNA = prepare_features_and_labels(rRNA_df)
    rRNA_model = RandomForestClassifier(n_estimators=100)
    for train_idx, test_idx in rkf.split(X_rRNA):
        X_train, X_test = X_rRNA.iloc[train_idx], X_rRNA.iloc[test_idx]
        y_train, y_test = y_rRNA.iloc[train_idx], y_rRNA.iloc[test_idx]
        rRNA_model.fit(X_train, y_train)
        predictions = rRNA_model.predict(X_test)
        for idx, (true_label, pred_label) in enumerate(zip(y_test, predictions)):
            if true_label != pred_label:
                gene_id = rRNA_df.iloc[test_idx[idx]]['ScientificName']
                misclassified_instances['12S rRNA'][gene_id] = misclassified_instances['12S rRNA'].get(gene_id, 0) + 1

    # Process mitofish
    mitofish_df = pd.read_csv(os.path.join(script_dir, '../csv_files/mitofish_sequences.csv'))
    kmer_features_mitofish = []
    for seq in mitofish_df['sequentie']:
        sequence = str(seq)
        kmer_feature = get_kmers(sequence)
        kmer_features_mitofish.append(kmer_feature)

    kmer_df_mitofish = pd.DataFrame(kmer_features_mitofish).fillna(0)
    valid_kmer_columns_mitofish = [column for column in kmer_df_mitofish.columns if is_valid_kmer(column)]
    filtered_kmer_df_mitofish = kmer_df_mitofish[valid_kmer_columns_mitofish]

    df_mitofish = pd.concat([mitofish_df, filtered_kmer_df_mitofish], axis=1)
    df_mitofish = df_mitofish.dropna(subset=['Mod'])

    X_mitofish, y_mitofish = prepare_features_and_labels(df_mitofish)
    mitofish_model = RandomForestClassifier(n_estimators=100)
    for train_idx, test_idx in rkf.split(X_mitofish):
        X_train, X_test = X_mitofish.iloc[train_idx], X_mitofish.iloc[test_idx]
        y_train, y_test = y_mitofish.iloc[train_idx], y_mitofish.iloc[test_idx]
        mitofish_model.fit(X_train, y_train)
        predictions = mitofish_model.predict(X_test)
        for idx, (true_label, pred_label) in enumerate(zip(y_test, predictions)):
            if true_label != pred_label:
                gene_id = df_mitofish.iloc[test_idx[idx]]['ID']
                misclassified_instances['mitofish'][gene_id] = misclassified_instances['mitofish'].get(gene_id, 0) + 1

# Combine misclassified instances into a grouped DataFrame
misclassified_data = []
for gene_type, instances in misclassified_instances.items():
    for gene_id, frequency in instances.items():
        gene_id = gene_id.replace(" ", "_")  # Replace spaces with underscores
        misclassified_data.append({'Gene_ID': gene_id, 'Gene_Type': gene_type, 'Frequency': frequency})

misclassified_df = pd.DataFrame(misclassified_data)

# Pivot the data to group by Gene_ID and show counts for each gene type
grouped_df = misclassified_df.pivot_table(index='Gene_ID', columns='Gene_Type', values='Frequency', fill_value=0).reset_index()

# Add a column to indicate presence in each dataset
grouped_df['In_COX1'] = grouped_df['Gene_ID'].isin(cox1_df['ScientificName'].str.replace(" ", "_")).astype(int)
grouped_df['In_12S_rRNA'] = grouped_df['Gene_ID'].isin(rRNA_df['ScientificName'].str.replace(" ", "_")).astype(int)
grouped_df['In_mitofish'] = grouped_df['Gene_ID'].isin(df_mitofish['ID'].str.replace(" ", "_")).astype(int)

# Add the 'Mod' column by merging with the COX1 dataset (or another relevant dataset)
mod_column_df = cox1_df[['ScientificName', 'Mod']].copy()
mod_column_df['ScientificName'] = mod_column_df['ScientificName'].str.replace(" ", "_")
grouped_df = pd.merge(grouped_df, mod_column_df, left_on='Gene_ID', right_on='ScientificName', how='left')
grouped_df = grouped_df.drop(columns=['ScientificName'])  # Drop duplicate column after merge

# Add a total misclassification count column (exclude presence columns)
misclassification_columns = [col for col in grouped_df.columns if col not in ['Gene_ID', 'In_COX1', 'In_12S_rRNA', 'In_mitofish', 'Mod']]
grouped_df['Total_Frequency'] = grouped_df[misclassification_columns].sum(axis=1)

# Sort the DataFrame to show IDs present in all datasets at the top
grouped_df = grouped_df.sort_values(by=['In_COX1', 'In_12S_rRNA', 'In_mitofish', 'Total_Frequency'], ascending=[False, False, False, False])

# Save the grouped DataFrame to a CSV file
grouped_df.to_csv(os.path.join(script_dir, '../csv_files/misclassified_instances_all_genes.csv'), index=False)



