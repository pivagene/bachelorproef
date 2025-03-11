import pandas as pd
import subprocess
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Blast import NCBIXML
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut,RepeatedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from collections import Counter
import io
import joblib
import threading
import matplotlib.pyplot as plt
# Step 1: Read the CSV file and write to a FASTA file and choose how to make the db
csv_file = 'AMP_species_list_COX1.csv'
df = pd.read_csv(csv_file)
# method 1: alle fasta bestanden uit csv halen om db te maken 
with open('COX1_db.fasta', 'w') as fasta_file:   # alle fasta bestanden uit csv halen om db te maken
    for index, row in df.iterrows():
        fasta_file.write(f">{row['FASTA']}\n")
# method 2: 1 fasta bestand uit csv halen om db te maken
# Extract the value from the 'fasta' column at row 604
fasta_value = df.at[604, 'FASTA']

# Save the extracted value as a FASTA file
with open('COX1_db.fasta', 'w') as fasta_file:
    fasta_file.write(f'>sequence_604\n{fasta_value}\n')
# voor deze stap moet je blast+ installeren en in path zetten bij de omgevingsvariabelen (https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
# Step 2: Create the BLAST database (hier verschillende groottes gebruiken kan ook via ncbi gdn worden expected time is 5 uur tho)
subprocess.run(['makeblastdb', '-in', 'COX1_db.fasta', '-dbtype', 'nucl', '-out', 'COX1_blastdb'])

# Function to extract features from a sequence
def extract_features(sequence):
    features = {
        'length': len(sequence),
        'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence)
    }
    return features

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k=4):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    total_kmers = len(kmers)
    for kmer in kmer_counts:
        kmer_counts[kmer] /= total_kmers
    return kmer_counts

# Function to perform local BLAST alignment and extract alignment score and bit score
def get_blast_scores(sequence, db='COX1_blastdb'):
    # Use a unique filename for each thread
    temp_fasta_filename = f'temp_query_{threading.get_ident()}.fasta'
    
    # Write the sequence to a temporary FASTA file
    with open(temp_fasta_filename, 'w') as temp_fasta:
        temp_fasta.write(f'>query\n{sequence}\n')
    
    # Run the local BLAST search
    result = subprocess.run(
        ['blastn', '-query', temp_fasta_filename, '-db', db, '-outfmt', '5'],
        capture_output=True, text=True
    )
    
    # Remove the temporary FASTA file
    os.remove(temp_fasta_filename)
    
    # Check if the BLAST search produced valid output
    if result.stdout.strip():
        # Parse the BLAST XML output
        blast_records = NCBIXML.parse(io.StringIO(result.stdout))
        for blast_record in blast_records:
            if blast_record.alignments:
                hsp = blast_record.alignments[0].hsps[0]
                return hsp.score, hsp.bits
    return 0, 0

# Load the CSV files
sequences_df = pd.read_csv('gene_seq_12SrRNA_final.csv')  # Contains sequence and ID
animals_df = pd.read_csv('gene_IDS_12SrRNA_final.csv')  # Contains ID and animal information
characteristics_df = pd.read_csv('AMP_species_list.csv')  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'
merged_df.to_csv('AMP_species_list_12SrRNA.csv', index=False)
merged_df = pd.read_csv('merged_species_list.csv')
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


# Add BLAST scores and bit scores to the features DataFrame
#features_df['blast_score'] = blast_scores
#features_df['bit_score'] = bit_scores

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
joblib.dump(model, 'random_forest_model_kmer_COX1.pkl')

# Save the blast scores and bit scores for future use
features_df.to_csv('blast_scores_COX1_1db.csv', index=False)


misclassified = pd.DataFrame({
    'True Label': y_test,
    'Predicted Label': y_pred,
    'Correct': y_test == y_pred
}, index=X_test.index)

# Merge misclassified instances with the original DataFrame to include additional variables
misclassified = misclassified.merge(df[['Gene_ID', 'ID']], left_index=True, right_index=True)

# Filter misclassified instances where Correct is False
misclassified_false = misclassified[misclassified['Correct'] == False]

misclassified_false.to_csv('misclassified_instances.csv', index=False)

# cohens kappa sore voor class descrapency
# terugwerken van wat interessant is om te vertellen tijdens de verdediging nut van genomische data onderzoeksvragen