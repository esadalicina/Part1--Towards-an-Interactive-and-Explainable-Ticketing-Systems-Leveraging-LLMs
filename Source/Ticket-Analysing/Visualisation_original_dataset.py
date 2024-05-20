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

# Print unique values of product column
unique_values = df["product"].unique()
print(f"Unique values of column tags': {unique_values}")

df = df.drop(columns=["tags"])

# Remove rows with NaN values in the 'tags' column
df = df.dropna(subset=['complaint_what_happened'])

# Remove duplicate values from column
df = df.drop_duplicates(subset=['complaint_what_happened'])

# Inspect the dataframe to understand the given data.
print(df.info())

# Remove sub_issue column
df = df.drop(columns=["sub_issue"])

# Remove issue column
df = df.drop(columns=["issue"])

# Inspect the dataframe to understand the given data.
print(df.info())

# Group by 'Product' and aggregate unique 'Sub-Product' values
product_subproduct_mapping = df.groupby('product')['sub_product'].unique().reset_index()

# Display the results
print(product_subproduct_mapping)

# Group by 'Product' and count unique 'Sub-Product' values
unique_subproduct_counts = df.groupby('product')['sub_product'].nunique().reset_index()

# Rename the column for clarity
unique_subproduct_counts.columns = ['product', 'Unique Sub-Product Count']

# Display the results
print(unique_subproduct_counts)


# Group by 'Product' and count unique 'complaints' values
unique_complaints_counts = df.groupby('product')['complaint_what_happened'].nunique().reset_index()

# Rename the column for clarity
unique_complaints_counts.columns = ['product', 'Unique complaints Count']

# Display the results
print(unique_complaints_counts)


# Define categories out of the subcategories:

# Define the categories and their subcategories
categories = {
    "Bank Account or Service": [
        "Checking or savings account",
        "Bank account or service"
    ],
    "Loans": [
        "Consumer Loan",
        "Mortgage",
        "Payday loan",
        "Payday loan, title loan, or personal loan",
        "Student loan",
        "Vehicle loan or lease"
    ],
    "Credit Cards and Prepaid Cards": [
        "Credit card",
        "Credit card or prepaid card",
        "Prepaid card"
    ],
    "Credit Reporting and Debt Collection": [
        "Credit reporting",
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Debt collection"
    ],
    "Money Transfers and Financial Services": [
        "Money transfer, virtual currency, or money service",
        "Money transfers",
        "Other financial service"
    ]
}

# Create a reverse mapping from subcategory to category
subcategory_to_category = {subcat: cat for cat, subcats in categories.items() for subcat in subcats}

# Map the subcategories to categories
df['category'] = df['product'].map(subcategory_to_category)

# Remove rows with NaN values in the 'tags' column
df = df.dropna(subset=['product'])

df = df.drop(columns=["sub_product"])

# Print the updated DataFrame to verify
print(df.info())


# Group by 'Product' and count unique 'Sub-Product' values
unique_category_counts = df.groupby('category')['complaint_what_happened'].nunique().reset_index()

# Rename the column for clarity
unique_category_counts.columns = ['category', 'Unique Sub-Product Count']

# Display the results
print(unique_category_counts)

# Using pandas' factorize method
df['category_encoded'] = pd.factorize(df['category'])[0]

print(df)

# Create a dictionary to map encoded categories to original category names
category_mapping = dict(zip(df['category_encoded'], df['category']))

print("\nMapping of Encoded Categories to Original Category Names:")
print(category_mapping)

# Specify the file path where you want to save the modified DataFrame as a CSV file
# output_file = '../../Dataset/clean.csv'

# Save the modified DataFrame to a CSV file
# df.to_csv(output_file, index=False)







