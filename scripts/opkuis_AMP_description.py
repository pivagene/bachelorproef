import os
import pandas as pd

script_dir = os.path.dirname(__file__)

df_merged = pd.read_csv(os.path.join(script_dir, '../csv_files/merged_data.csv'))
df_merged['Description'] = df_merged['Description'].str.lower()
df_merged['Description'] = df_merged['Description'].str.replace('reprod', 'reproduction', regex=False)
df_merged['Description'] = df_merged['Description'].str.replace('for', '', regex=False)
df_merged['Description'] = df_merged['Description'].str.replace('males', 'male', regex=False)
df_merged['Description'] = df_merged['Description'].str.replace('females','female', regex=False)
df_merged['Description'] = df_merged['Description'].str.replace('juveniles', 'juvenile', regex=False)
df_merged['Description'] = df_merged['Description'].str.replace('adults', 'adult', regex=False)
unique_values = sorted(df_merged['Description'].unique())

description_counts = df_merged['Description'].value_counts()

# Create a DataFrame with sorted unique values and their counts
unique_values_with_counts = pd.DataFrame({
    'Description': unique_values,
    'Count': [description_counts[desc] for desc in unique_values]
})