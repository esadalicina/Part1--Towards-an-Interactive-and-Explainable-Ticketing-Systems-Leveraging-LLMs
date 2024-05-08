import json

import pandas as pd
from tabulate import tabulate

# -------------------------------------------------- Data Understanding ------------------------------------------------


# Opening JSON file
f = open('../../Dataset/complaints-2021-05-14_08_16.json')

# Returns JSON object as a dictionary
data = json.load(f)
df = pd.json_normalize(data)

# First 5 rows of the dataframe
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))


# Specify the file path of your CSV file
file_path = '../../Dataset/Cleaned_Dataset.csv'

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)

# Keep the columns"complaint_what_happened" & "Topic" only in the new dataframe --> training_data
training_data = df_clean[['complaint_what_happened', 'Topic']]

# Display the first few rows of the DataFrame
print(training_data.head())

# ------------------------------------------------------- Clustering --------------------------------------------------

