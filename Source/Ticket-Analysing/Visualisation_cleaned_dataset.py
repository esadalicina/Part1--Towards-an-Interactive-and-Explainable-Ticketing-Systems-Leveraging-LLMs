import pandas as pd

print("Visualization of cleaned dataset")

# Specify the file path of your CSV file
file_path = '../../Dataset/Cleaned_Dataset.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Group by 'Product' and aggregate unique 'Sub-Product' values
product_subproduct_mapping = df.groupby('Topic_category')['category'].unique().reset_index()

# Display the results
print(product_subproduct_mapping)

# Group by 'Product' and count unique 'Sub-Product' values
unique_subproduct_counts = df.groupby('Topic_category')['category'].nunique().reset_index()

# Rename the column for clarity
unique_subproduct_counts.columns = ['Topic_category', 'Unique Sub-Product Count']

# Display the results
print(unique_subproduct_counts)


# Group by 'Topic_category' and aggregate up to 5 examples from 'Complaint_clean'
product_examples = df.groupby('category')['complaint_what_happened'].apply(lambda x: x.head(1).tolist()).reset_index()

# Adjust pandas display options to ensure the full text is shown
pd.set_option('display.max_colwidth', None)

# Display the results
print(product_examples)

