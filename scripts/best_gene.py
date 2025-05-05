import pandas as pd
import os
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, probplot, f_oneway
script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)

# Function to process a gene dataset and return cross-validation scores
def process_gene(sequences_file, ids_file, gene_name):
    sequences_df = pd.read_csv(os.path.join(script_dir, sequences_file))
    ids_df = pd.read_csv(os.path.join(script_dir, ids_file))
    characteristics_df = pd.read_csv(os.path.join(script_dir, '../csv_files/AMP_species_list.csv'))
    merged_df = pd.merge(sequences_df, ids_df, on='Gene_ID', how='inner')
    merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')

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

    feature_columns = list(filtered_kmer_df.columns)
    X = df[feature_columns]
    y = df['Mod']

    model = RandomForestClassifier(n_estimators=100)
    rkf = RepeatedKFold(n_splits=7, n_repeats=3)
    cv_scores = cross_val_score(model, X, y, cv=rkf, scoring='accuracy')

    return cv_scores

# Process COX1 gene
cox1_scores = process_gene('../csv_files/gene_seq_COX1_final.csv', '../csv_files/gene_IDS_COX1_final.csv', 'COX1')

# Process rRNA
rRNA_scores = process_gene('../csv_files/gene_seq_12SrRNA_final.csv', '../csv_files/gene_IDS_12SrRNA_final.csv', 'rRNA')

# Process mitofish (mitofishfish_sequences is already in the correct format)
mitofish_df = pd.read_csv(os.path.join(script_dir, '../csv_files/mitofish_sequences.csv'))

# Extract k-mer features for mitofish
kmer_features_mitofish = []
for seq in mitofish_df['sequentie']:
    sequence = str(seq)
    kmer_feature = get_kmers(sequence)
    kmer_features_mitofish.append(kmer_feature)

kmer_df_mitofish = pd.DataFrame(kmer_features_mitofish).fillna(0)
valid_kmer_columns_mitofish = [column for column in kmer_df_mitofish.columns if is_valid_kmer(column)]
filtered_kmer_df_mitofish = kmer_df_mitofish[valid_kmer_columns_mitofish]

# Combine features with mitofish DataFrame
df_mitofish = pd.concat([mitofish_df, filtered_kmer_df_mitofish], axis=1)
df_mitofish = df_mitofish.dropna(subset=['Mod'])

# Prepare data for mitofish
feature_columns_mitofish = list(filtered_kmer_df_mitofish.columns)
X_mitofish = df_mitofish[feature_columns_mitofish]
y_mitofish = df_mitofish['Mod']

# Perform cross-validation for mitofish
model = RandomForestClassifier(n_estimators=100)
rkf = RepeatedKFold(n_splits=7, n_repeats=3)
mitofish_scores = cross_val_score(model, X_mitofish, y_mitofish, cv=rkf, scoring='accuracy')

# Repeat cross-validation 20 times and collect means
cv_means_list = {'COX1': [], '12S rRNA': [], 'mitofish': []}

for _ in range(20):
    cox1_scores = process_gene('../csv_files/gene_seq_COX1_final.csv', '../csv_files/gene_IDS_COX1_final.csv', 'COX1')
    rRNA_scores = process_gene('../csv_files/gene_seq_12SrRNA_final.csv', '../csv_files/gene_IDS_12SrRNA_final.csv', 'rRNA')
    mitofish_scores = cross_val_score(model, X_mitofish, y_mitofish, cv=rkf, scoring='accuracy')

    cv_means_list['COX1'].append(cox1_scores.mean())
    cv_means_list['12S rRNA'].append(rRNA_scores.mean())
    cv_means_list['mitofish'].append(mitofish_scores.mean())

# Plot the means of the cross-validation scores
plt.boxplot([cv_means_list['COX1'], cv_means_list['12S rRNA'], cv_means_list['mitofish']], labels=['COX1', '12S rRNA', 'mitogenoom'])
plt.title('Cross-Validation Mean accuracy Comparison (20 Repeats)')
plt.ylabel('Mean Accuracy')
plt.show()

print("Cross-Validation Means (20 Repeats):")
for gene, means in cv_means_list.items():
    print(f"{gene}: {means}")
    print(f"Average {gene}: {sum(means) / len(means):.4f}")

# Generate QQ plots for normality checks
plt.figure(figsize=(12, 4))
for i, (gene, means) in enumerate(cv_means_list.items(), start=1):
    plt.subplot(1, 3, i)
    probplot(means, dist="norm", plot=plt)
    plt.title(f"QQ Plot for {gene}")

plt.tight_layout()
plt.show()

# Perform F-tests for variance comparison
f_stat_rRNA_vs_mitofish, p_value_rRNA_vs_mitofish = f_oneway(cv_means_list['12S rRNA'], cv_means_list['mitofish'])
f_stat_rRNA_vs_COX1, p_value_rRNA_vs_COX1 = f_oneway(cv_means_list['12S rRNA'], cv_means_list['COX1'])
f_stat_COX1_vs_mitofish, p_value_COX1_vs_mitofish = f_oneway(cv_means_list['COX1'], cv_means_list['mitofish'])

# Determine equal_var parameter based on F-test p-values
equal_var_rRNA_vs_mitofish = p_value_rRNA_vs_mitofish >= 0.05
equal_var_rRNA_vs_COX1 = p_value_rRNA_vs_COX1 >= 0.05
equal_var_COX1_vs_mitofish = p_value_COX1_vs_mitofish >= 0.05

# Perform one-sided t-tests
t_stat_rRNA_vs_mitofish, p_value_rRNA_vs_mitofish = ttest_ind(cv_means_list['12S rRNA'], cv_means_list['mitofish'], equal_var=equal_var_rRNA_vs_mitofish, alternative='greater')
t_stat_rRNA_vs_COX1, p_value_rRNA_vs_COX1 = ttest_ind(cv_means_list['12S rRNA'], cv_means_list['COX1'], equal_var=equal_var_rRNA_vs_COX1, alternative='greater')
t_stat_COX1_vs_mitofish, p_value_COX1_vs_mitofish = ttest_ind(cv_means_list['COX1'], cv_means_list['mitofish'], equal_var=equal_var_COX1_vs_mitofish, alternative='greater')

print("\nOne-Sided T-Test Results:")
print(f"rRNA > mitofish: t-statistic = {t_stat_rRNA_vs_mitofish:.4f}, p-value = {p_value_rRNA_vs_mitofish:.2e}")
print(f"rRNA > COX1: t-statistic = {t_stat_rRNA_vs_COX1:.4f}, p-value = {p_value_rRNA_vs_COX1:.2e}")
print(f"COX1 > mitofish: t-statistic = {t_stat_COX1_vs_mitofish:.4f}, p-value = {p_value_COX1_vs_mitofish:.2e}")

# Interpret results
if p_value_rRNA_vs_mitofish < 0.05:
    print("rRNA performs significantly better than mitofish (p < 0.05).")
else:
    print("No significant evidence that rRNA performs better than mitofish (p >= 0.05).")

if p_value_rRNA_vs_COX1 < 0.05:
    print("rRNA performs significantly better than COX1 (p < 0.05).")
else:
    print("No significant evidence that rRNA performs better than COX1 (p >= 0.05).")

if p_value_COX1_vs_mitofish < 0.05:
    print("COX1 performs significantly better than mitofish (p < 0.05).")
else:
    print("No significant evidence that COX1 performs better than mitofish (p >= 0.05).")

plt.savefig(os.path.join(script_dir, '../grafieken/best_gene_classifying.png'), bbox_inches='tight')