import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from collections import Counter
import joblib
import matplotlib.pyplot as plt
# Function to extract features from a sequence
def extract_features(sequence):
    features = {
        'length': len(sequence),
        'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence)
    }
    return features

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=5):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    total_kmers = len(kmers)
    for kmer in kmer_counts:
        kmer_counts[kmer] /= total_kmers
    return kmer_counts

seqCOX1 = pd.read_csv('AMP_species_list_COX1.csv')
seq12SrRNA = pd.read_csv('AMP_species_list_12SrRNA.csv')
seq12 = pd.merge(seqCOX1, seq12SrRNA, on='ScientificName' )
# seq12.to_csv('merged_species_list.csv', index=False)
merged_df = seq12
# Extract features from the FASTA sequences
kmer_features1 = []
kmer_features2 = []

for seq1, seq2 in zip(merged_df['sequentie_x'], merged_df['sequentie_y']):
    sequence1 = str(seq1)
    sequence2 = str(seq2)
    kmer_feature1 = get_kmers(sequence1)
    kmer_feature2 = get_kmers(sequence2)
    kmer_features1.append(kmer_feature1)
    kmer_features2.append(kmer_feature2)

# Convert k-mer features to DataFrames
kmer_df1 = pd.DataFrame(kmer_features1).fillna(0)
kmer_df2 = pd.DataFrame(kmer_features2).fillna(0)

# Combine k-mer features from both datasets
combined_kmer_df = pd.concat([kmer_df1, kmer_df2], axis=1)

# Combine features with the merged DataFrame
# Prepare the data
# Assuming 'Characteristic' is the column you want to predict
# Assuming 'Mod' is the column you want to predict
X = combined_kmer_df
y = merged_df['Mod_x']  # Replace with your target column

# Ensure there are no missing values
X = X.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=37)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=37)
# Perform cross-validation
rkf = RepeatedKFold(n_splits=7, n_repeats=3, random_state=37)
cv_scores = cross_val_score(model, X_train, y_train, cv=rkf, scoring='accuracy')

# Print cross-validation results
print(f'Repeated K-Fold Cross-Validation Accuracy: {cv_scores.mean()}')
print(f'Repeated K-Fold Cross-Validation Standard Deviation: {cv_scores.std()}')
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
# joblib.dump(model, 'random_forest_model_kmer_COX1.pkl')

# cohens kappa sore voor class descrapency
# terugwerken van wat interessant is om te vertellen tijdens de verdediging nut van genomische data onderzoeksvragen