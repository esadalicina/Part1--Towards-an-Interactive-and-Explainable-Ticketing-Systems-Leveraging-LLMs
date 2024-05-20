import json
import numpy as np
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------- Load the Data ----------------------------------------------

# Opening JSON file and loading the data into a dictionary
with open('../../Dataset/complaints-2021-05-14_08_16.json') as f:
    data = json.load(f)

# Normalize JSON data into a flat table
df = pd.json_normalize(data)

# Display a subset of the dataframe
print("------------------------------ Displaying Dataset Sample ------------------------------")
print(tabulate(df[100:120], headers='keys', tablefmt='pretty'))

# Display information about the dataframe to understand its structure
print("------------------------------ Dataset Information ------------------------------")
print(df.info())

# Print the column names
print("------------------------------ Column Names ------------------------------")
print("Columns are: ", df.columns.values)

# ---------------------------------------------- Data Cleaning ----------------------------------------------

# Drop unnecessary columns
df = df.drop(columns=[
    '_index', "_id", "_source.state", '_type', '_source.zip_code', '_score', '_source.complaint_id',
    "_source.date_received", "_source.consumer_disputed", "_source.company_response", "_source.company",
    "_source.submitted_via", "_source.date_sent_to_company", "_source.company_public_response",
    "_source.timely", "_source.consumer_consent_provided"
])

# Rename columns for better readability
df.rename(columns={
    '_source.tags': 'tags',
    '_source.issue': 'issue',
    '_source.product': 'product',
    '_source.sub_product': 'sub_product',
    '_source.complaint_what_happened': 'complaint_what_happened',
    '_source.sub_issue': 'sub_issue',
}, inplace=True)

# Print the updated column names
print("------------------------------ Updated Column Names ------------------------------")
print("Columns are: ", df.columns.values)

# Replace empty strings with NaN
df = df.apply(lambda x: x.replace('', np.nan))

# Display information about the dataframe after modifications
print("------------------------------ Dataset Information After Initial Cleaning ------------------------------")
print(df.info())

# Count the number of unique values in each column and print the results
print("------------------------------ Unique Value Counts ------------------------------")
unique_value_counts = df.nunique()
print(unique_value_counts)

# Print unique values of specific columns
print(f"Unique values of column 'tags': {df['tags'].unique()}")
print(f"Unique values of column 'product': {df['product'].unique()}")

# Drop the 'tags' column as it's no longer needed
df = df.drop(columns=["tags"])

# Remove rows where 'complaint_what_happened' is NaN
df = df.dropna(subset=['complaint_what_happened'])

# Remove duplicate entries based on 'complaint_what_happened'
df = df.drop_duplicates(subset=['complaint_what_happened'])

# Display information about the dataframe after further cleaning
print("------------------------------ Dataset Information After Further Cleaning ------------------------------")
print(df.info())

# Drop columns 'sub_issue' and 'issue' as they are no longer needed
df = df.drop(columns=["sub_issue", "issue"])

# Display information about the dataframe after dropping additional columns
print("------------------------------ Dataset Information After Dropping Columns ------------------------------")
print(df.info())

# ---------------------------------------------- Grouping and Aggregation ----------------------------------------------

# Group by 'product' and aggregate unique 'sub_product' values
product_subproduct_mapping = df.groupby('product')['sub_product'].unique().reset_index()
print("------------------------------ Product to Sub-Product Mapping ------------------------------")
print(product_subproduct_mapping)

# Group by 'product' and count unique 'sub_product' values
unique_subproduct_counts = df.groupby('product')['sub_product'].nunique().reset_index()
unique_subproduct_counts.columns = ['product', 'Unique Sub-Product Count']
print("------------------------------ Unique Sub-Product Counts ------------------------------")
print(unique_subproduct_counts)

# Group by 'product' and count unique 'complaint_what_happened' values
unique_complaints_counts = df.groupby('product')['complaint_what_happened'].nunique().reset_index()
unique_complaints_counts.columns = ['product', 'Unique Complaints Count']
print("------------------------------ Unique Complaints Counts ------------------------------")
print(unique_complaints_counts)

# ---------------------------------------------- Category Mapping ----------------------------------------------

# Define categories and their subcategories
categories = {
    "Bank Account or Service": ["Checking or savings account", "Bank account or service"],
    "Loans": ["Consumer Loan", "Mortgage", "Payday loan", "Payday loan, title loan, or personal loan", "Student loan", "Vehicle loan or lease"],
    "Credit Cards and Prepaid Cards": ["Credit card", "Credit card or prepaid card", "Prepaid card"],
    "Credit Reporting and Debt Collection": ["Credit reporting", "Credit reporting, credit repair services, or other personal consumer reports", "Debt collection"],
    "Money Transfers and Financial Services": ["Money transfer, virtual currency, or money service", "Money transfers", "Other financial service"]
}

# Create a reverse mapping from subcategory to category
subcategory_to_category = {subcat: cat for cat, subcats in categories.items() for subcat in subcats}

# Map the subcategories to categories
df['category'] = df['product'].map(subcategory_to_category)

# Remove rows with NaN values in the 'product' column
df = df.dropna(subset=['product'])

# Drop the 'sub_product' column as it's no longer needed
df = df.drop(columns=["sub_product"])

# Display the updated DataFrame information
print("------------------------------ Dataset Information After Category Mapping ------------------------------")
print(df.info())

# Group by 'category' and count unique 'complaint_what_happened' values
unique_category_counts = df.groupby('category')['complaint_what_happened'].nunique().reset_index()
unique_category_counts.columns = ['category', 'Unique Complaint Count']
print("------------------------------ Unique Complaint Counts by Category ------------------------------")
print(unique_category_counts)

# Encode categories using pandas' factorize method
df['category_encoded'] = pd.factorize(df['category'])[0]

print("------------------------------ Final DataFrame with Encoded Categories ------------------------------")
print(df)

# Create a dictionary to map encoded categories to original category names
category_mapping = dict(zip(df['category_encoded'], df['category']))

print("\n----------------------Mapping of Encoded Categories to Original Category Names ------------------------------")
print(category_mapping)


# Specify the file path where you want to save the modified DataFrame as a CSV file
output_file = '../../Dataset/Cleaned_Dataset.csv'

# Save the modified DataFrame to a CSV file
df.to_csv(output_file, index=False)






