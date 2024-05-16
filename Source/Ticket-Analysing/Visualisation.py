import json

import numpy as np
import pandas as pd
from tabulate import tabulate


# -------------------------------------------------- Loading the data -------------------------------------------------

# Opening JSON file
f = open('../../Dataset/complaints-2021-05-14_08_16.json')

# Returns JSON object as a dictionary
data = json.load(f)
df = pd.json_normalize(data)

print("------------------------------------------------ Dataset ------------------------------------------------")
# First 5 rows of the dataframe
print(tabulate(df[100:120], headers='keys', tablefmt='pretty'))


print("---------------------------------------------- Dataset Info  ------------------------------------------------")
# Inspect the dataframe to understand the given data.
print(df.info())


print("------------------------------------------------ Columns ------------------------------------------------")
# Print the column names
print("Columns are: ", df.columns.values)

print("---------------------------------------------- Delete Columns  ------------------------------------------------")
df = df.drop(columns=['_index', "_id", "_source.state", '_type', '_source.zip_code', '_score', '_source.complaint_id', "_source.date_received",
                      "_source.consumer_disputed", "_source.company_response", "_source.company", "_source.submitted_via",
                      "_source.date_sent_to_company", "_source.company_public_response", "_source.timely", "_source.consumer_consent_provided"])

# Assign new column names
df.rename(columns={
    '_source.tags': 'tags',
    '_source.issue': 'issue',
    '_source.product': 'product',
    '_source.sub_product': 'sub_product',
    '_source.complaint_what_happened': 'complaint_what_happened',
    '_source.sub_issue': 'sub_issue',
}, inplace=True)

print("------------------------------------------------ Columns ------------------------------------------------")
# Print the column names
print("Columns are: ", df.columns.values)

# Replace empty rows with NaN
df = df.apply(lambda x: x.replace('', np.nan))

print("---------------------------------------------- Dataset Info  ------------------------------------------------")
# Inspect the dataframe to understand the given data.
print(df.info())


# Count the number of unique values in each column
unique_value_counts = df.nunique()

# Print the number of unique values in each column
print("Number of unique values in each column:")
print(unique_value_counts)

# Print unique values of each column
unique_values = df["tags"].unique()
print(f"Unique values of column tags': {unique_values}")

df = df.drop(columns=["tags"])

# Remove rows with NaN values in the 'tags' column
df = df.dropna(subset=['complaint_what_happened'])

# Inspect the dataframe to understand the given data.
print(df.info())


# Remove duplicate values from column 'A'
df = df.drop_duplicates(subset=['complaint_what_happened'])

# Inspect the dataframe to understand the given data.
print(df.info())

df = df.drop(columns=["sub_issue"])

# Inspect the dataframe to understand the given data.
print(df.info())


# Print unique values of column
unique_values = df["product"].unique()
print(f"Unique values of column tags': {unique_values}")

temp_series = df.groupby('product').size()
print(temp_series)

temp_series.sort_values(ascending = False, inplace = True)
print("\nAfter sorting in Descending Order :\n\n",temp_series)


temp_series = df.groupby('issue').size()
print(temp_series)

temp_series.sort_values(ascending = False, inplace = True)
print("\nAfter sorting in Descending Order :\n\n",temp_series)

# Issue = Summary of complaint (Free field to fill in)
# Product = Selecting box



