from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

# ---------------------------------------------------------------- Choose right columns ----------------------------------------------------

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data = df_clean['complaint_what_happened_without_stopwords']
label_data = df_clean['category_encoded']

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.2, random_state=42, shuffle=True)


# ----------------------------------------------------------------- Tokenization with Tfidf ----------------------------------------------------------

def Tfidf_method(training_data, count_vect=None, tfidf_transformer=None):
    if count_vect is None:
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(training_data)
        tfidf_transformer = TfidfTransformer()
        X_train_tf = tfidf_transformer.fit_transform(X_train_counts)
    else:
        X_train_counts = count_vect.transform(training_data)
        X_train_tf = tfidf_transformer.transform(X_train_counts)
    return X_train_tf, count_vect, tfidf_transformer

# Apply TF-IDF method to train and test data
X_train_tf, count_vect, tfidf_transformer = Tfidf_method(train_texts)
X_test_tf, _, _ = Tfidf_method(test_texts, count_vect, tfidf_transformer)


import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Handle class imbalance using SMOTE on TF-IDF features
smote = SMOTE(random_state=42)
X_train_tf_resampled, train_labels_resampled = smote.fit_resample(X_train_tf, train_labels) # type: ignore


# ----------------------------------------------------------------- Tokenization with Word2Vec ----------------------------------------------------------

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data_w2v = df_clean['complaint_what_happened_basic_clean_DL']
label_data_w2v = df_clean['category_encoded']

# Split the data into training and testing sets
train_texts_w2v, test_texts_w2v, train_labels_w2v, test_labels_w2v = train_test_split(ticket_data_w2v, label_data_w2v, test_size=0.2, random_state=42, shuffle=True)


def Word2vec_method(train_texts):
    # Tokenize the training texts
    data = []
    for text in train_texts:
        temp = []
        # tokenize the sentence into words
        for j in word_tokenize(text):
            temp.append(j.lower())
        data.append(temp)
    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=data, vector_size=250, window=5, min_count=7, workers=4)
    return w2v_model

# Function to average word vectors for each document
def get_word2vec_embeddings(texts, model):
    embeddings = []
    for text in texts:
        tokens = word_tokenize(text)
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            vector = np.mean(vectors, axis=0)
        else:
            vector = np.zeros(model.vector_size)
        embeddings.append(vector)
    return np.array(embeddings)

# Apply Word2Vec method and create the 'tokens' 
w2v_model = Word2vec_method(train_texts_w2v)

train_embeddings = get_word2vec_embeddings(train_texts_w2v, w2v_model)
test_embeddings = get_word2vec_embeddings(test_texts_w2v, w2v_model)

# Handle class imbalance using SMOTE on Word2Vec embeddings
# train_embeddings_resampled, train_labels_resampled_w2v = smote.fit_resample(train_embeddings, train_labels) # type: ignore


# ----------------------------------------------------------------- View Data ----------------------------------------------------------

# Check for get_feature_names_out or fall back to get_feature_names
try:
    feature_names = count_vect.get_feature_names_out()
except AttributeError:
    feature_names = count_vect.get_feature_names() # type: ignore

# Convert the resampled TF-IDF data to a DataFrame for viewing
tfidf_df = pd.DataFrame(X_train_tf_resampled.toarray(), columns=feature_names) # type: ignore
tfidf_df['label'] = train_labels_resampled.values

# Display a few rows of the TF-IDF DataFrame
print("TF-IDF Features with Resampled Labels:")
print(tfidf_df.head())

# Convert the resampled Word2Vec embeddings to a DataFrame for viewing
w2v_df = pd.DataFrame(test_embeddings)
w2v_df['label'] = test_labels.values

# Display a few rows of the Word2Vec DataFrame
print("\nWord2Vec Embeddings with Resampled Labels:")
print(w2v_df.head(17))


# Convert the resampled Word2Vec embeddings to a DataFrame for viewing
w2v_df = pd.DataFrame(train_embeddings)
w2v_df['label'] = train_labels.values

# Display a few rows of the Word2Vec DataFrame
print("\nWord2Vec Embeddings with Resampled Labels:")
print(w2v_df.head(17))

print("Train", train_embeddings.shape)
print("label", train_labels.shape)


