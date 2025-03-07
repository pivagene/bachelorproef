import pandas as pd
import subprocess
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Blast import NCBIXML
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from collections import Counter
import io
import joblib
import threading
import matplotlib.pyplot as plt
script_dir = os.path.dirname(__file__)

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=4):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return Counter(kmers)


# Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_seq_12SrRNA_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_12SrRNA_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'
merged_df.to_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list_12SrRNA.csv'), index=False)

# Extract features from the FASTA sequences
features = []
kmer_features = []
#blast_scores = []
#bit_scores = []

# Process each sequence one by one
for seq in merged_df['sequentie']:
    sequence = str(seq)
    kmer_feature = get_kmers(sequence)
    kmer_features.append(kmer_feature)

# Convert features to a DataFrame
features_df = pd.DataFrame(features)
# features_df = pd.read_csv("blast_scores.csv") als je de blast al hebt gedaan
# Convert k-mer features to a DataFrame
kmer_df = pd.DataFrame(kmer_features).fillna(0)

# Combine features with the merged DataFrame
df = pd.concat([merged_df, features_df, kmer_df], axis=1)

# Prepare the data
# Assuming 'Characteristic' is the column you want to predict
# Specify the columns you want to use for X
feature_columns = [] + list(kmer_df.columns)  # Add more features as needed length and gc doen ni veel kmer wint zelf zonder blast score
X = df[feature_columns]
y = df['Mod']  # Replace with your target column

# Ensure there are no missing values
X = X.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=37)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=37)
# Perform cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=37)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

# Print cross-validation results
print(f'Cross-Validation Accuracy: {cv_scores.mean()}')
print(f'Cross-Validation Standard Deviation: {cv_scores.std()}')

# Fit the model on the entire training set
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Compute Cohen's Kappa score
kappa_score = cohen_kappa_score(y_test, y_pred)
print(f"Cohen's Kappa Score: {kappa_score}")

# Get the unique labels in the test set
unique_labels = y_test.unique()

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

# Display the confusion matrix with correct labels
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size as needed
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(ax=ax)
# plt.savefig('confusion_matrix.png')
# Optionally, save the model for future use
joblib.dump(model, os.path.join('../../other_files/random_forest_model_kmer_COX1.pkl'))

# Save the blast scores and bit scores for future use
features_df.to_csv(os.path.join(script_dir, '../../other_files/blast_scores_COX1_1db'), index=False)


misclassified = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label': y_pred,
    'Correct': y_test == y_pred
}, index=X_test.index)

# Merge misclassified instances with the original DataFrame to include additional variables
misclassified = misclassified.merge(df[['Gene_ID', 'ID']], left_index=True, right_index=True)

# Filter misclassified instances where Correct is False
misclassified_false = misclassified[misclassified['Correct'] == False]

misclassified_false.to_csv(os.path.join(script_dir,'../../csv_files/misclassified_instances.csv'), index=False)

# cohens kappa sore voor class descrapency
# terugwerken van wat interessant is om te vertellen tijdens de verdediging nut van genomische data onderzoeksvragen