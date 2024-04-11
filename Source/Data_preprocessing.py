import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate

ticket = pd.read_csv('/Users/esada/Documents/UNI.lu/MICS/Sem3/Master-Thesis/Dataset/customer_support_tickets.csv', )
print(tabulate(ticket.head(), headers='keys', tablefmt='pretty'))
print(tabulate(ticket.tail(), headers='keys', tablefmt='pretty'))

# Get the shape and size of the dataset
print('No of rows:\033[1m', ticket.shape[0], '\033[0m')
print('No of cols:\033[1m', ticket.shape[1], '\033[0m')

# Get more info on it
# 1. Name of the columns
# 2. Find the data types of each columns
# 3. Look for any null/missing values
print(ticket.info())

# Describe the dataset with various summary and statistics
print(ticket.describe())


# Remove columns
columns_to_remove = ['First Response Time', 'Time to Resolution', 'Customer Satisfaction Rating', 'Date of Purchase', 'Customer Age', 'Customer Gender']
df = ticket.drop(columns=columns_to_remove)

print(df.info())


# Text normalization:
# Convert text to lowercase



# remove punctuation
# perform stemming or lemmatization to reduce words to their base form.