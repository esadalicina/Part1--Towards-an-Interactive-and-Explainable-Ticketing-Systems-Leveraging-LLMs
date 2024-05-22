import re
import warnings
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Analysing_and_Cleaning import *


warnings.filterwarnings('ignore')

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Write your function to Lemmatize the texts
stopwords = nlp.Defaults.stop_words

# change the display properties of pandas to max
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ----------------------------------------- Prepare the text for classification ---------------------------------------


def preprocess_text(text):
    text = text.lower()  # Convert to lower case
    text = re.sub(r'^\[[\w\s]\]+$', ' ', text)  # Remove text in square brackets
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'^[a-zA-Z]\d+\w*$', ' ', text)  # Remove words with numbers
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords]
    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Preprocess text columns
df['complaint_what_happened_lemmatized'] = df['complaint_what_happened'].progress_apply(lambda x: preprocess_text(x)) # type: ignore
# Remove the 'xxxx' and 'xxx' from the text columns
df['complaint_what_happened_lemmatized'] = df['complaint_what_happened_lemmatized'].str.replace('xxx', '')
df['complaint_what_happened_lemmatized'] = df['complaint_what_happened_lemmatized'].str.replace('xxxx', '')


# The clean dataframe should now contain the raw and lemmatized complaint with the category and product columns
print(tabulate(df.head(), headers='keys', tablefmt='pretty')) # type: ignore


# Specify the file path where you want to save the modified DataFrame as a CSV file
# output_file = '../../Dataset/Cleaned_Dataset.csv'

# Save the modified DataFrame to a CSV file
# df.to_csv(output_file, index=False)
