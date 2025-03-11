import pandas as pd
import os

# Get the current script directory
script_dir = os.path.dirname(__file__)

# Construct the path to the CSV file
csv_file_path = os.path.join(script_dir, '../csv_files/AMP_collection.csv')

# Read the CSV file
df = pd.read_csv(csv_file_path)