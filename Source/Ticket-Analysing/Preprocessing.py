import re
import warnings
from tqdm import tqdm
import spacy
from Analysing_and_Cleaning import *


tqdm.pandas()
warnings.filterwarnings('ignore')

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# change the display properties of pandas to max
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Write your function to Lemmatize the texts
stopwords = nlp.Defaults.stop_words


# ----------------------------------------- Prepare the text for classification ---------------------------------------


def clean_text(text):
    text = text.lower()  # Convert to lower case
    text = re.sub(r'^\[[\w\s]\]+$', ' ', text)  # Remove text in square brackets
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'^[a-zA-Z]\d+\w*$', ' ', text)  # Remove words with numbers
    return text


def lemmatization(texts):
    lemma_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.lemma_ for token in doc if token.text not in set(stopwords)]
        lemma_sentences.append(' '.join(sent))
    return lemma_sentences


# Clean text columns
df['complaint_what_happened_clean'] = df['complaint_what_happened'].progress_apply(lambda x: clean_text(x))

# lemmitize the text columns
df['complaint_what_happened_lemmatized'] = lemmatization(df['complaint_what_happened_clean'])

# Remove the 'xxxx' from the text columns
df['complaint_what_happened_clean'] = df['complaint_what_happened_clean'].str.replace('xxxx', '')
df['complaint_what_happened_lemmatized'] = df['complaint_what_happened_lemmatized'].str.replace('xxxx', '')


# The clean dataframe should now contain the raw complaint, clean and lemmatized complaint
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))


# Specify the file path where you want to save the modified DataFrame as a CSV file
# output_file = '../../Dataset/Cleaned_Dataset.csv'

# Save the modified DataFrame to a CSV file
# df.to_csv(output_file, index=False)
