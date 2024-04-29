import re
import numpy as np
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from tabulate import tabulate

# Load the dataset
ticket = pd.read_csv('/Users/esada/Documents/UNI.lu/MICS/Sem3/Master-Thesis/Dataset/export-collection-all-properties-new-dataset-with-ticket-description-2024-04-28.csv')


print(tabulate(ticket.head(), headers='keys', tablefmt='pretty'))
print(tabulate(ticket.tail(), headers='keys', tablefmt='pretty'))


print('No of rows:\033[1m', ticket.shape[0], '\033[0m')
print('No of cols:\033[1m', ticket.shape[1], '\033[0m')


print(ticket.info())
print(ticket.describe())


# Group the DataFrame by 'Ticket Type' and sample 5 rows from each group
sampled_rows = ticket.groupby('Ticket Type').apply(lambda x: x.sample(5))

# Reset index to remove the multi-index created by the groupby operation
sampled_rows.reset_index(drop=True, inplace=True)

# Display the sampled rows
print(tabulate(sampled_rows, headers='keys', tablefmt='pretty'))



# ----------------------------------------------------- Data Cleaning ------------------------------------------------

#columns_to_remove = ['First Response Time', 'Time to Resolution', 'Customer Satisfaction Rating', 'Date of Purchase',
                     #'Customer Age', 'Customer Gender']
#df = ticket.drop(columns=columns_to_remove)

#df['Ticket Description'] = df.apply(lambda row: row['Ticket Description'].replace('{product_purchased}', str(row['Product Purchased'])), axis=1)


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
ticket['Cleaned_Description'] = ticket['Ticket Description'].apply(clean_text)

print(ticket['Cleaned_Description'])

lemmatizer = WordNetLemmatizer()


# Preprocess text data
def preprocess_text(text):

    tokens = word_tokenize(text.lower())  # Tokenization and convert to lowercase

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Filter out unique lemmatized tokens
    unique_lemmatized_tokens = list(set(lemmatized_tokens))

    unique_lemmatized_tokens = [token for token in unique_lemmatized_tokens if token.isalpha()]  # Remove non-alphabetic tokens
    unique_lemmatized_tokens = [token for token in unique_lemmatized_tokens if token not in stopwords.words('english')]  # Remove stopwords

    tagged_tokens = pos_tag(unique_lemmatized_tokens)
    # Filter out verbs (tags starting with 'VB')
    filtered_tokens = [token for token, tag in tagged_tokens if not tag.startswith('VB')]

    return filtered_tokens


ticket['Cleaned_Descriptions_right'] = ticket['Cleaned_Description'].apply(preprocess_text)


# Descriptive Statistics
ticket['Word_Count'] = ticket['Cleaned_Descriptions_right'].apply(len)
print("Descriptive Statistics:")
print(ticket['Word_Count'].describe())

# Print the column
print(ticket['Cleaned_Descriptions_right'])


# Train RandomForestClassifier model
cleaned_data_vector = ticket['Cleaned_Descriptions_right']
labels = ticket['Ticket Type']

# Create a pipeline for text vectorization and model training
text_classifier_RF = make_pipeline(
    TfidfVectorizer(analyzer=lambda x: x),  # Use tokenized words as analyzer
    RandomForestClassifier(n_estimators=400, random_state=0)
)


# Perform cross-validation and compute accuracy scores
cv_scores = cross_val_score(text_classifier_RF, cleaned_data_vector, labels, cv=7)
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))

text_classifier_SVM = make_pipeline(
    TfidfVectorizer(analyzer=lambda x: x),
    SVC(kernel='linear')  # You can choose different kernel functions like 'linear', 'poly', 'rbf', etc.
)


# Perform cross-validation and compute accuracy scores
cv_scores = cross_val_score(text_classifier_SVM, cleaned_data_vector, labels, cv=7)
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))


text_classifier_NB = make_pipeline(
    TfidfVectorizer(analyzer=lambda x: x),
    MultinomialNB()
)






