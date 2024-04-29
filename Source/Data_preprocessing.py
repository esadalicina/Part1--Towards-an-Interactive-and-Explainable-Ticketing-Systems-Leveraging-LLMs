import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
ticket = pd.read_csv('/Users/esada/Documents/UNI.lu/MICS/Sem3/Master-Thesis/Dataset/customer_support_tickets.csv')

# ----------------------------------------------------- Data Analysis ------------------------------------------------

print(tabulate(ticket.head(50), headers='keys', tablefmt='pretty'))
print(tabulate(ticket.tail(), headers='keys', tablefmt='pretty'))

print('No of rows:\033[1m', ticket.shape[0], '\033[0m')
print('No of cols:\033[1m', ticket.shape[1], '\033[0m')

print(ticket.info())
print(ticket.describe())

# ----------------------------------------------------- Data Cleaning ------------------------------------------------

columns_to_remove = ['First Response Time', 'Time to Resolution', 'Customer Satisfaction Rating', 'Date of Purchase',
                     'Customer Age', 'Customer Gender']
df = ticket.drop(columns=columns_to_remove)

df['Ticket Description'] = df.apply(lambda row: row['Ticket Description'].replace('{product_purchased}', str(row['Product Purchased'])), axis=1)


# Define a function for cleaning text
def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove email IDs
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# Apply the cleaning function to the 'Ticket Description' column
df['Cleaned_Description'] = df['Ticket Description'].apply(clean_text)

print(df['Cleaned_Description'])

unique_counts = df['Ticket Type'].value_counts()

# Print the unique value counts
print(unique_counts)


# ----------------------------------------------------- Model Selection ------------------------------------------------


# ----------------------------------------------------- Feature Extraction --------------------------------------------


# -------------------------------------------------------- Classifier -------------------------------------------------
