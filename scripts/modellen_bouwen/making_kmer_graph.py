import pandas as pd
import os
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(__file__)

 # Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_seq_12SrRNA_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_12SrRNA_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k):
    kmers = [sequence[j:j+k] for j in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)



mean_cv_scores = []

kmer_lengths =[2]*10 + [3]*10 + [4]*10 + [5]*10 + [6]*10

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
    model = RandomForestClassifier(n_estimators=300)
    # Perform cross-validation
    kf = KFold(n_splits=2, shuffle=True)
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




# Initialize lists to store results
mean_training_scores = []  # To store training accuracy
mean_validation_scores = []  # To store validation accuracy

kmer_lengths = [1] + [2] + [3] + [4] + [5] + [6] + [7] + [8] + [9] + [10] + [11]
for i in kmer_lengths:
    features = []
    kmer_features = []
    # Process each sequence one by one
    for seq in merged_df['sequentie']:
        sequence = str(seq)
        kmer_feature = get_kmers(sequence, i)
        kmer_features.append(kmer_feature)
    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)
    # Filter out columns with invalid k-mers
    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]
    # Combine features with the merged DataFrame
    df = pd.concat([merged_df, filtered_kmer_df], axis=1)

    # Prepare the data
    feature_columns = list(filtered_kmer_df.columns)
    X = df[feature_columns]
    y = df['Mod']  # Replace with your target column

    # Ensure there are no missing values
    X = X.fillna(0)

    # Perform cross-validation
    kf = KFold(n_splits=2, shuffle=True)
    training_scores = []  # To store training accuracy for each fold
    validation_scores = []  # To store validation accuracy for each fold

    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Random Forest model
        model = RandomForestClassifier(n_estimators=300)
        model.fit(X_train, y_train)

        # Calculate training accuracy
        training_accuracy = model.score(X_train, y_train)
        training_scores.append(training_accuracy)

        # Calculate validation accuracy
        validation_accuracy = model.score(X_test, y_test)
        validation_scores.append(validation_accuracy)

    # Store the mean training and validation accuracy for this k-mer length
    mean_training_scores.append(np.mean(training_scores))
    mean_validation_scores.append(np.mean(validation_scores))

# Save results to a DataFrame
accuracy_df = pd.DataFrame({
    'kmer_length': kmer_lengths,
    'mean_training_scores': mean_training_scores,
    'mean_validation_scores': mean_validation_scores
})

# Plot training vs validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(accuracy_df['kmer_length'], accuracy_df['mean_training_scores'], label='Training Accuracy', marker='o', color='blue')
plt.plot(accuracy_df['kmer_length'], accuracy_df['mean_validation_scores'], label='Validation Accuracy', marker='o', color='orange')
plt.title('Training vs Validation Accuracy for Different k-mer Lengths')
plt.xlabel('k-mer Length')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()



import pandas as pd
import os
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(__file__)

# Load the CSV files
sequences_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_seq_12SrRNA_final.csv'))  # Contains sequence and ID
animals_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/gene_IDS_12SrRNA_final.csv'))  # Contains ID and animal information
characteristics_df = pd.read_csv(os.path.join(script_dir, '../../csv_files/AMP_species_list.csv'))  # Contains animal and characteristic

# Merge the DataFrames
merged_df = pd.merge(sequences_df, animals_df, on='Gene_ID', how='inner')  # Merge on 'ID'
merged_df = pd.merge(merged_df, characteristics_df, on='ID', how='inner')  # Merge on 'Animal'

# Function to extract k-mer counts from a sequence
def get_kmers(sequence, k):
    kmers = [sequence[j:j+k] for j in range(len(sequence) - k + 1)]
    return Counter(kmers)

# Function to check if a k-mer contains only A, C, T, or G
def is_valid_kmer(kmer):
    valid_bases = {'A', 'C', 'T', 'G'}
    return all(base in valid_bases for base in kmer)


# Initialize lists to store results
mean_training_scores = []  # To store training accuracy
mean_validation_scores = []  # To store validation accuracy

kmer_lengths = [2, 3, 4, 5, 6, 7, 8, 9]

for i in kmer_lengths:
    features = []
    kmer_features = []
    # Process each sequence one by one
    for seq in merged_df['sequentie']:
        sequence = str(seq)
        kmer_feature = get_kmers(sequence, i)
        kmer_features.append(kmer_feature)
    # Convert k-mer features to a DataFrame
    kmer_df = pd.DataFrame(kmer_features).fillna(0)
    # Filter out columns with invalid k-mers
    valid_kmer_columns = [column for column in kmer_df.columns if is_valid_kmer(column)]
    filtered_kmer_df = kmer_df[valid_kmer_columns]
    # Combine features with the merged DataFrame
    df = pd.concat([merged_df, filtered_kmer_df], axis=1)

    # Prepare the data
    feature_columns = list(filtered_kmer_df.columns)
    X = df[feature_columns]
    y = df['Mod']  # Replace with your target column

    # Ensure there are no missing values
    X = X.fillna(0)

    # Perform cross-validation
    kf = KFold(n_splits=2, shuffle=True)
    training_scores = []  # To store training accuracy for each fold
    validation_scores = []  # To store validation accuracy for each fold

    for train_index, test_index in kf.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the Decision Tree model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Calculate training accuracy
        training_accuracy = model.score(X_train, y_train)
        training_scores.append(training_accuracy)

        # Calculate validation accuracy
        validation_accuracy = model.score(X_test, y_test)
        validation_scores.append(validation_accuracy)

    # Store the mean training and validation accuracy for this k-mer length
    mean_training_scores.append(np.mean(training_scores))
    mean_validation_scores.append(np.mean(validation_scores))

# Save results to a DataFrame
accuracy_df = pd.DataFrame({
    'kmer_length': kmer_lengths,
    'mean_training_scores': mean_training_scores,
    'mean_validation_scores': mean_validation_scores
})

# Plot training vs validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(accuracy_df['kmer_length'], accuracy_df['mean_training_scores'], label='Training Accuracy', marker='o', color='blue')
plt.plot(accuracy_df['kmer_length'], accuracy_df['mean_validation_scores'], label='Validation Accuracy', marker='o', color='orange')
plt.title('Training vs Validation Accuracy for Different k-mer Lengths (Decision Tree)')
plt.xlabel('k-mer Length')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()