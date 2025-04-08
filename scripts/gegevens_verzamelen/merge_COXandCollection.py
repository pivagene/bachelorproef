import pandas as pd

# Load the CSV files
cox1_df = pd.read_csv('c:\\Users\\devos\\OneDrive - UGent\\3de bach\\bachelorproef\\Bachelorproef_AMP\\python\\bachelorproef github\\csv_files\\AMP_species_list_COX1.csv')
collection_df = pd.read_csv('c:\\Users\\devos\\OneDrive - UGent\\3de bach\\bachelorproef\\Bachelorproef_AMP\\python\\bachelorproef github\\csv_files\\AMP_collection.csv')

# Replace underscores with spaces in the 'Species' column of the collection DataFrame
collection_df['Species'] = collection_df['Species'].str.replace('_', ' ')

# Merge the two DataFrames on 'ScientificName' and 'Species'
merged_df = pd.merge(cox1_df, collection_df, left_on='ScientificName', right_on='Species', how='inner')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('c:\\Users\\devos\\OneDrive - UGent\\3de bach\\bachelorproef\\Bachelorproef_AMP\\python\\bachelorproef github\\csv_files\\merged_data.csv', index=False)

print("Merged CSV file saved successfully!")