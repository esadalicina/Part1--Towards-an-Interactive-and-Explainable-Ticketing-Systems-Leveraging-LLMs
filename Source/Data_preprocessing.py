import json
import re
import numpy as np
import pandas as pd
import warnings
from tabulate import tabulate
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()
warnings.filterwarnings('ignore')
import spacy
# Load the English language model
nlp = spacy.load("en_core_web_sm")

# change the display properties of pandas to max
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Opening JSON file
f = open('../Dataset/complaints-2021-05-14_08_16.json')

# returns JSON object as
# a dictionary
data = json.load(f)
df = pd.json_normalize(data)

# First 5 rows of the dataframe
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))


# Inspect the dataframe to understand the given data.
print(df.info())


#print the column names
print("Columns are: ", df.columns.values)

#Assign new column names
df.rename(columns={
    '_index': 'index',
    '_type': 'type',
    '_id': 'id',
    '_score': 'score',
    '_source.tags': 'tags',
    '_source.zip_code': 'zip_code',  # Rename this column correctly
    '_source.complaint_id': 'complaint_id',
    '_source.issue': 'issue',
    '_source.date_received': 'date_received',
    '_source.state': 'state',
    '_source.consumer_disputed': 'consumer_disputed',
    '_source.product': 'product',
    '_source.company_response': 'company_response',
    '_source.company': 'company',
    '_source.submitted_via': 'submitted_via',
    '_source.date_sent_to_company': 'date_sent_to_company',
    '_source.company_public_response': 'company_public_response',
    '_source.sub_product': 'sub_product',
    '_source.timely': 'timely',
    '_source.complaint_what_happened': 'complaint_what_happened',
    '_source.sub_issue': 'sub_issue',
    '_source.consumer_consent_provided': 'consumer_consent_provided'
}, inplace=True)



#Assign nan in place of blanks in the complaint_what_happened column
df['complaint_what_happened'].replace('', np.nan, inplace=True)

# Null values count after replacing blanks with nan
df['complaint_what_happened'].isnull().sum()

#Remove all rows where complaint_what_happened column is nan
df.dropna(subset=['complaint_what_happened'],inplace=True)


# Write your function here to clean the text and remove all the unnecessary elements.
def clean_text(text):
  text=text.lower()  #convert to lower case
  text=re.sub(r'^\[[\w\s]\]+$',' ',text) #Remove text in square brackets
  text=re.sub(r'[^\w\s]',' ',text) #Remove punctuation
  text=re.sub(r'^[a-zA-Z]\d+\w*$',' ',text) #Remove words with numbers
  return text



#Write your function to Lemmatize the texts
stopwords = nlp.Defaults.stop_words
def lemmatization(texts):
    lemma_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.lemma_ for token in doc if token.text not in set(stopwords)]
        lemma_sentences.append(' '.join(sent))
    return lemma_sentences


#Create a dataframe('df_clean') that will have only the complaints and the lemmatized complaints
df_clean = pd.DataFrame()

# Clean text columns
df_clean['complaint_what_happened'] = df['complaint_what_happened'].progress_apply(lambda x: clean_text(x))

# lemmitize the text columns
df_clean['complaint_what_happened_lemmatized'] = lemmatization(df_clean['complaint_what_happened'])



# adding category and sub_category columns to the dataframe for better topic identification
df_clean['category'] = df['product']
df_clean['sub_category'] = df['sub_product']

#Write your function to extract the POS tags only for NN
def extract_pos_tags(texts):
    pos_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.text for token in doc if token.tag_ == 'NN']
        pos_sentences.append(' '.join(sent))
    return pos_sentences

df_clean["complaint_POS_removed"] = extract_pos_tags(df_clean['complaint_what_happened_lemmatized'])



#The clean dataframe should now contain the raw complaint, lemmatized complaint and the complaint after removing POS tags.
print(tabulate(df_clean.head(), headers='keys', tablefmt='pretty'))