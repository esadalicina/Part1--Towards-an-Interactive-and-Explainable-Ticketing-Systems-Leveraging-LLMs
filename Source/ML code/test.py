import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
ticket_data = pd.read_csv('/Users/esada/Documents/UNI.lu/MICS/Sem3/Master-Thesis/Dataset/customer_support_tickets.csv')

# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and convert to lowercase
    tokens = [token for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    return tokens

ticket_data['Cleaned_Description'] = ticket_data['Ticket Description'].apply(preprocess_text)

# Descriptive Statistics
ticket_data['Word_Count'] = ticket_data['Cleaned_Description'].apply(len)
print("Descriptive Statistics:")
print(ticket_data['Word_Count'].describe())

# Word Frequency Analysis
all_words = [word for desc in ticket_data['Cleaned_Description'] for word in desc]
freq_dist = FreqDist(all_words)
print("\nMost Common Words:")
print(freq_dist.most_common(20))

# Word Cloud Visualization
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.show()



pd.set_option('display.max_colwidth', None)

# Print the column
print(ticket_data['Cleaned_Description'][3000:8000])