import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
nltk.download('punkt')
import re
from gensim import models, corpora
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
nltk.download('brown')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
import re
from sklearn import preprocessing

# datasource=pd.read_csv('/Users/esada/Documents/UNI.lu/MICS/Sem3/Master-Thesis/Dataset/customer_support_tickets.csv')
# print(datasource.head())
#
# datasource['Ticket Description'] = datasource.apply(lambda row: row['Ticket Description'].replace('{product_purchased}', str(row['Product Purchased'])), axis=1)
#
# features=datasource['Ticket Description']
# labels=datasource['Ticket Type']
#
# # Assuming 'df' is your DataFrame
# nan_counts = datasource.isnull().sum()
#
# # Print the counts of NaN values for each column
# print(nan_counts)

#datasource=pd.read_csv('https://raw.githubusercontent.com/krushnapavan9/Support-Ticket-Classification/master/Deep%20Learning/latest_ticket_data.csv')
#datasource.head()

#features=datasource['Description']
#labels=datasource['Category']
#
#
# cList = {
#     "ain't": "am not",
#     "aren't": "are not",
#     "can't": "cannot",
#     "can't've": "cannot have",
#     "'cause": "because",
#     "could've": "could have",
#     "couldn't": "could not",
#     "couldn't've": "could not have",
#     "didn't": "did not",
#     "doesn't": "does not",
#     "don't": "do not",
#     "hadn't": "had not",
#     "hadn't've": "had not have",
#     "hasn't": "has not",
#     "haven't": "have not",
#     "he'd": "he would",
#     "he'd've": "he would have",
#     "he'll": "he will",
#     "he'll've": "he will have",
#     "he's": "he is",
#     "how'd": "how did",
#     "how'd'y": "how do you",
#     "how'll": "how will",
#     "how's": "how is",
#     "I'd": "I would",
#     "I'd've": "I would have",
#     "I'll": "I will",
#     "I'll've": "I will have",
#     "I'm": "I am",
#     "I've": "I have",
#     "isn't": "is not",
#     "it'd": "it had",
#     "it'd've": "it would have",
#     "it'll": "it will",
#     "it'll've": "it will have",
#     "it's": "it is",
#     "let's": "let us",
#     "ma'am": "madam",
#     "mayn't": "may not",
#     "might've": "might have",
#     "mightn't": "might not",
#     "mightn't've": "might not have",
#     "must've": "must have",
#     "mustn't": "must not",
#     "mustn't've": "must not have",
#     "needn't": "need not",
#     "needn't've": "need not have",
#     "o'clock": "of the clock",
#     "oughtn't": "ought not",
#     "oughtn't've": "ought not have",
#     "shan't": "shall not",
#     "sha'n't": "shall not",
#     "shan't've": "shall not have",
#     "she'd": "she would",
#     "she'd've": "she would have",
#     "she'll": "she will",
#     "she'll've": "she will have",
#     "she's": "she is",
#     "should've": "should have",
#     "shouldn't": "should not",
#     "shouldn't've": "should not have",
#     "so've": "so have",
#     "so's": "so is",
#     "that'd": "that would",
#     "that'd've": "that would have",
#     "that's": "that is",
#     "there'd": "there had",
#     "there'd've": "there would have",
#     "there's": "there is",
#     "they'd": "they would",
#     "they'd've": "they would have",
#     "they'll": "they will",
#     "they'll've": "they will have",
#     "they're": "they are",
#     "they've": "they have",
#     "to've": "to have",
#     "wasn't": "was not",
#     "we'd": "we had",
#     "we'd've": "we would have",
#     "we'll": "we will",
#     "we'll've": "we will have",
#     "we're": "we are",
#     "we've": "we have",
#     "weren't": "were not",
#     "what'll": "what will",
#     "what'll've": "what will have",
#     "what're": "what are",
#     "what's": "what is",
#     "what've": "what have",
#     "when's": "when is",
#     "when've": "when have",
#     "where'd": "where did",
#     "where's": "where is",
#     "where've": "where have",
#     "who'll": "who will",
#     "who'll've": "who will have",
#     "who's": "who is",
#     "who've": "who have",
#     "why's": "why is",
#     "why've": "why have",
#     "will've": "will have",
#     "won't": "will not",
#     "won't've": "will not have",
#     "would've": "would have",
#     "wouldn't": "would not",
#     "wouldn't've": "would not have",
#     "y'all": "you all",
#     "y'alls": "you alls",
#     "y'all'd": "you all would",
#     "y'all'd've": "you all would have",
#     "y'all're": "you all are",
#     "y'all've": "you all have",
#     "you'd": "you had",
#     "you'd've": "you would have",
#     "you'll": "you you will",
#     "you'll've": "you you will have",
#     "you're": "you are",
#     "you've": "you have"
# }
#
# c_re = re.compile('(%s)' % '|'.join(cList.keys()))
#
#
# def expandContractions(text, c_re=c_re):
#     def replace(match):
#         return cList[match.group(0)]
#
#     return c_re.sub(replace, text)
#
#
# print(expandContractions("I ain't going anywhere"))
#
#
# def clean_text(text):
#     # Removing comments fom body
#     cleaned_text = text.replace(r'From:.*$', '')
#
#     # Expanding the sentence
#     cleaned_text = (re.sub("([A-Z]{1}[a-z])", " \1", i) for i in cleaned_text)
#
#     # To Lower case
#     # cleaned_text = cleaned_text.lower()
#
#     # Removing email ID's
#     cleaned_text = (re.sub("[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.]+", "", i) for i in cleaned_text)
#
#     # Removing hyperlinks
#     cleaned_text = (re.sub(
#         "(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-zA-Z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?",
#         "", i) for i in cleaned_text)
#
#     cleaned_text = (re.sub("http\S+", "", i) for i in cleaned_text)
#
#     # cleaned_text = cleaned_text.replace(r'url= .', '')
#
#     # Replacing Contractions
#     cleaned_text = (re.sub("’", "'", i) for i in cleaned_text)
#
#     cleaned_text = (expandContractions(i) for i in cleaned_text)
#
#     # Removing unnecessary punctuations
#     cleaned_text = (re.sub("[\\{\}\;\:\"\\<\>\/\@\#$\%\^\&\*\_\~\–]+", " ", i)
#     for i in cleaned_text)
#
#     cleaned_text = (re.sub("(\?)([a-zA-Z0-9]+)", "\1 \2", i) for i in cleaned_text)
#
#     cleaned_text = (re.sub("([A-Za-z]{1,})(\.)([A-Za-z]{2,})", "\1 \2 \3", i) for i in cleaned_text)
#
#     cleaned_text = (re.sub("[\']+", "", i) for i in cleaned_text)
#
#     # cleaned_text = cleaned_text.replace(r'\xa0', ' ')
#
#     cleaned_text = (re.sub("(\.)(\,|\?|\'|\!|\s){1,}", "\1 ", i) for i in cleaned_text)
#
#     cleaned_text = (re.sub("(!){1}(\s)*(\=)", " not equals ", i) for i in cleaned_text)
#
#     cleaned_text = (re.sub("(\=)", " equals ", i) for i in cleaned_text)
#
#     # Replacing _ with " "
#     cleaned_text = (re.sub("_", " ", i) for i in cleaned_text)
#
#     # Replacing @–
#     # cleaned_text = cleaned_text.apply(lambda i: re.sub(r'[@–]','',i))
#
#     # Removing ASCII
#     # cleaned_text = cleaned_text.apply(lambda i:  re.sub(r'[^\x00-\x7F]+',' ', i))
#
#     # Replacing multiple '.' with single '.'
#     cleaned_text = (re.sub("\s\.+", " .", i) for i in cleaned_text)
#
#     cleaned_text = (re.sub("\.+", ".", i) for i in cleaned_text)
#
#     # Replacing . follower by characters with spaces followed by . followed by characters
#     cleaned_text = (re.sub("([0-9]+)(\.{1})([^0-9]+)", "\1 \2 \3", i) for i in cleaned_text)
#
#     # remove after regards
#     cleaned_text = (re.sub(r'regards.*$', '', i) for i in cleaned_text)
#     cleaned_text = (re.sub(r'tel \+.*$', '', i) for i in cleaned_text)
#     cleaned_text = (re.sub(r'fax \+.*$', '', i) for i in cleaned_text)
#
#     # print (cleaned_text)
#
#     # Remove all the special characters
#     cleaned_text = re.sub(r'\W', ' ', ("").join(cleaned_text))
#     # print (cleaned_text)
#
#     # Remove single characters from the start
#     cleaned_text = re.sub(r'\^[a-zA-Z]\s+', ' ', cleaned_text)
#
#     # Substituting multiple spaces with single space
#     cleaned_text = re.sub(r'\s+', ' ', cleaned_text, flags=re.I)
#
#     # remove all single characters
#     cleaned_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', cleaned_text)
#
#     # Getting to root word
#     # cleaned_text = (lemmatizeCust(i) for i in  cleaned_text)
#
#     # Removing stopwords
#     # cleaned_text = (removeStopWord(i) for i in  cleaned_text)
#
#     # Replacing muiltple white spaces
#     # cleaned_text = (re.sub(" +"," ",i) for i in  cleaned_text)
#
#     return (cleaned_text)
#
#
# stopWordlist =[
#  'thank',
#  'regards',
#  'ocp',
#  'thanks',
#  'hi',
#  'hello',
#  'team',
#  'team',
#  'team',
#  'team',
#  'team',
#  'team',
#  'team',
#  'team',
#  'i',
#  'me',
#  'my',
#  'myself',
#  'we',
#  'our',
#  'ours',
#  'ourselves',
#  'you',
#  "you're",
#  "you've",
#  "you'll",
#  "you'd",
#  'your',
#  'yours',
#  'yourself',
#  'yourselves',
#  'he',
#  'him',
#  'his',
#  'himself',
#  'she',
#  "she's",
#  'her',
#  'hers',
#  'herself',
#  'it',
#  "it's",
#  'its',
#  'itself',
#  'they',
#  'them',
#  'their',
#  'theirs',
#  'themselves',
#  'what',
#  'which',
#  'who',
#  'whom',
#  'this',
#  'that',
#  "that'll",
#  'these',
#  'those',
#  'am',
#  'is',
#  'are',
#  'was',
#  'were',
#  'be',
#  'been',
#  'being',
#  'have',
#  'has',
#  'had',
#  'having',
#  'do',
#  'does',
#  'did',
#  'doing',
#  'a',
#  'an',
#  'the',
#  'and',
#  'if',
#  'or',
#  'because',
#  'as',
#  'until',
#  'while',
#  'of',
#  'at',
#  'by',
#  'for',
#  'with',
#  'about',
#  'against',
#  'between',
#  'into',
#  'through',
#  'during',
#  'before',
#  'after',
#  'above',
#  'below',
#  'to',
#  'from',
#  'up',
#  'down',
#  'in',
#  'out',
#  'on',
#  'off',
#  'over',
#  'under',
#  'again',
#  'further',
#  'then',
#  'once',
#  'here',
#  'there',
#  'when',
#  'where',
#  'why',
#  'how',
#  'all',
#  'any',
#  'both',
#  'each',
#  'few',
#  'more',
#  'most',
#  'other',
#  'some',
#  'such',
#  'no',
#  'nor',
#  'only',
#  'own',
#  'same',
#  'so',
#  'than',
#  'too',
#  'very',
#  's',
#  't',
#  'can',
#  'will',
#  'just',
#  'don',
#  "don't",
#  'should',
#  "should've",
#  'now',
#  'd',
#  'll',
#  'm',
#  'o',
#  're',
#  've',
#  'y',
#  'ain',
#  'aren',
#  "aren't",
#  'couldn',
#  "couldn't",
#  'didn',
#  "didn't",
#  'doesn',
#  "doesn't",
#  'hadn',
#  "hadn't",
#  'hasn',
#  "hasn't",
#  'haven',
#  "haven't",
#  'isn',
#  "isn't",
#  'ma',
#  'mightn',
#  "mightn't",
#  'mustn',
#  "mustn't",
#  'needn',
#  "needn't",
#  'shan',
#  "shan't",
#  'shouldn',
#  "shouldn't",
#  'wasn',
#  "wasn't",
#  'weren',
#  "weren't",
#  'won',
#  "won't",
#  'wouldn',
#  "wouldn't",
#   "xa0",
#   "nbsp"]
#
#
# def removeStopWord(text):
#     string = ""
#     for word in text.split(" "):
#         if (word in stopWordlist):
#             string = string + ""
#         else:
#             string = string + " " + word
#     return string
#
# from nltk.stem import PorterStemmer
# from nltk.tokenize import sent_tokenize, word_tokenize
#
# porter=PorterStemmer()
# def stemSentence(sentence):
#     token_words=word_tokenize(sentence)
#     token_words
#     stem_sentence=[]
#     for word in token_words:
#         stem_sentence.append(porter.stem(word))
#         stem_sentence.append(" ")
#     return "".join(stem_sentence)
#
# x=stemSentence("I am loving it")
# print(x)
#
# import nltk
# nltk.download('punkt')
#
# cleaned_data_train = []
# for sentence in range(0, len(features)):  # Define Your features above
#     cleaned_sentence = expandContractions(str(features[sentence]))
#
#     cleaned_sentence = clean_text(cleaned_sentence)
#     cleaned_sentence = stemSentence(cleaned_sentence)
#     cleaned_sentence = removeStopWord(cleaned_sentence)
#
#     cleaned_data_train.append(cleaned_sentence)
#
#
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import stopwords
#
# stop_words = ['in', 'of', 'at', 'a', 'the','an','or','i','and','has','he','will','was','with','is','its','if']
#
# cv_vect = CountVectorizer(binary=True,ngram_range=(1,2),stop_words=stop_words)
#
# cv_vect.fit(features)
# cleaned_data_vector= cv_vect.transform(features)
# labels = labels.factorize()[0]
#
# print(labels[:100])
# #print(labels.factorize()[1])
# print(labels.shape)
# y = np.bincount(labels[:8000])
# ii = np.nonzero(y)[0]
# for a,b in zip(ii,y[ii]):
#   print(a,b)

nltk.download('stopwords')

# Load the dataset
ticket_data = pd.read_csv('/Users/esada/Documents/UNI.lu/MICS/Sem3/Master-Thesis/Dataset/customer_support_tickets.csv')
ticket_data['Ticket Description'] = ticket_data.apply(lambda row: row['Ticket Description'].replace('{product_purchased}', str(row['Product Purchased'])), axis=1)

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
print(ticket_data['Cleaned_Description'][100:2000])


# Train RandomForestClassifier model
cleaned_data_vector = ticket_data['Cleaned_Description'][:3000]
labels = ticket_data['Ticket Type'][:3000]

print(ticket_data["Ticket Type"])

# Create a pipeline for text vectorization and model training
text_classifier_RF = make_pipeline(
    TfidfVectorizer(analyzer=lambda x: x),  # Use tokenized words as analyzer
    RandomForestClassifier(n_estimators=300, random_state=0)
)

# Perform cross-validation and compute accuracy scores
cv_scores = cross_val_score(text_classifier_RF, cleaned_data_vector, labels, cv=7)
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))