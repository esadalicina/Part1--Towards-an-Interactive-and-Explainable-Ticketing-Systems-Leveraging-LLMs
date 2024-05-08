import json
import re
import numpy as np
import pandas as pd
import warnings
import plotly.express as px
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from tabulate import tabulate
from tqdm import tqdm
from wordcloud import WordCloud
import spacy

tqdm.pandas()
warnings.filterwarnings('ignore')

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# change the display properties of pandas to max
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# -------------------------------------------------- Loading the data -------------------------------------------------

# Opening JSON file
f = open('../../Dataset/complaints-2021-05-14_08_16.json')

# Returns JSON object as a dictionary
data = json.load(f)
df = pd.json_normalize(data)

# First 5 rows of the dataframe
print(tabulate(df.head(), headers='keys', tablefmt='pretty'))

# -------------------------------------------------- Data preparation -------------------------------------------------

# Inspect the dataframe to understand the given data.
print(df.info())

# Print the column names
print("Columns are: ", df.columns.values)

# Assign new column names
df.rename(columns={
    '_index': 'index',
    '_type': 'type',
    '_id': 'id',
    '_score': 'score',
    '_source.tags': 'tags',
    '_source.zip_code': 'zip_code',
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

# Assign nan in place of blanks in the complaint_what_happened column
df['complaint_what_happened'].replace('', np.nan, inplace=True)

# Null values count after replacing blanks with nan
df['complaint_what_happened'].isnull().sum()

# Remove all rows where complaint_what_happened column is nan
df.dropna(subset=['complaint_what_happened'], inplace=True)


# ----------------------------------------- Prepare the text for topic modeling ----------------------------------------

# Write your function here to clean the text and remove all the unnecessary elements.
def clean_text(text):
    text = text.lower()  # Convert to lower case
    text = re.sub(r'^\[[\w\s]\]+$', ' ', text)  # Remove text in square brackets
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'^[a-zA-Z]\d+\w*$', ' ', text)  # Remove words with numbers
    return text


# Write your function to Lemmatize the texts
stopwords = nlp.Defaults.stop_words


def lemmatization(texts):
    lemma_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.lemma_ for token in doc if token.text not in set(stopwords)]
        lemma_sentences.append(' '.join(sent))
    return lemma_sentences


# Create a dataframe('df_clean') that will have only the complaints and the lemmatized complaints
df_clean = pd.DataFrame()

# Clean text columns
df_clean['complaint_what_happened'] = df['complaint_what_happened'].progress_apply(lambda x: clean_text(x))

# lemmitize the text columns
df_clean['complaint_what_happened_lemmatized'] = lemmatization(df_clean['complaint_what_happened'])

# adding category and sub_category columns to the dataframe for better topic identification
df_clean['category'] = df['product']
df_clean['sub_category'] = df['sub_product']


# Write your function to extract the POS tags only for NN
def extract_pos_tags(texts):
    pos_sentences = []
    for doc in tqdm(nlp.pipe(texts)):
        sent = [token.text for token in doc if token.tag_ == 'NN']
        pos_sentences.append(' '.join(sent))
    return pos_sentences


df_clean["complaint_POS_removed"] = extract_pos_tags(df_clean['complaint_what_happened_lemmatized'])

# The clean dataframe should now contain the raw complaint,
# lemmatized complaint and the complaint after removing POS tags.
print(tabulate(df_clean.head(), headers='keys', tablefmt='pretty'))


# ------------------------------ Exploratory data analysis to get familiar with the data -------------------------------

# Write your code here to visualise the data according to the 'Complaint' character length
df_clean['complaint_length'] = df_clean['complaint_what_happened'].str.len()
df_clean['complaint_what_happened_lemmatized_length'] = df_clean['complaint_what_happened_lemmatized'].str.len()
df_clean['complaint_POS_removed_length'] = df_clean['complaint_POS_removed'].str.len()

fig = go.Figure()
fig.add_trace(go.Histogram(x=df_clean['complaint_length'], name='Complaint'))
fig.add_trace(go.Histogram(x=df_clean['complaint_what_happened_lemmatized_length'], name='Complaint Lemmatized'))
fig.add_trace(go.Histogram(x=df_clean['complaint_POS_removed_length'], name='Complaint POS Removed'))
fig.update_layout(barmode='overlay', title='Complaint Character Length', xaxis_title='Character Length',
                  yaxis_title='Count')
fig.update_traces(opacity=0.75)
fig.show()

# Using a word cloud find the top 40 words by frequency among all the articles after processing the text
wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=2000, height=1500, max_words=40).generate(
    ' '.join(df_clean['complaint_POS_removed']))
plt.imshow(wordcloud, interpolation='bilinear', aspect='auto')
plt.axis("off")
plt.show()

# Removing -PRON- from the text corpus
df_clean['Complaint_clean'] = df_clean['complaint_POS_removed'].str.replace('-PRON-', '')


# function to get the specified top n-grams
def get_top_n_words(corpus, n=None, count=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:count]


# Print the top 10 words in the unigram frequency and plot the same using a bar graph
unigram = get_top_n_words(df_clean['Complaint_clean'], 1, 10)
for word, freq in unigram:
    print(word, freq)
px.bar(x=[word for word, freq in unigram], y=[freq for word, freq in unigram], title='Top 10 Unigrams')

# Print the top 10 words in the bigram frequency and plot the same using a bar graph
bigram = get_top_n_words(df_clean['Complaint_clean'], 2, 10)
for word, freq in bigram:
    print(word, freq)
px.bar(x=[word for word, freq in bigram], y=[freq for word, freq in bigram], title='Top 10 Bigrams')

# Print the top 10 words in the trigram frequency and plot the same using a bar graph
trigram = get_top_n_words(df_clean['Complaint_clean'], 3, 10)
for word, freq in trigram:
    print(word, freq)
px.bar(x=[word for word, freq in trigram], y=[freq for word, freq in trigram], title='Top 10 Trigram')

df_clean['Complaint_clean'] = df_clean['Complaint_clean'].str.replace('xxxx', '')

# All masked texts has been removed
print(tabulate(df_clean.head(), headers='keys', tablefmt='pretty'))
