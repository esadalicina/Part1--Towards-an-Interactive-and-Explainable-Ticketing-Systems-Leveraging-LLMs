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



# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data_w2v = df_clean['complaint_what_happened_basic_clean_DL']
label_data_w2v = df_clean['category_encoded']

smote = SMOTE(random_state=42)

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
train_embeddings_resampled, train_labels_resampled_w2v = smote.fit_resample(train_embeddings, train_labels) # type: ignore




# Retrieve the first row text
original_text_row = ticket_data_w2v.iloc[0]

# Function to get Word2Vec embeddings for a single text
def get_word2vec_embedding_for_text(text, model):
    tokens = word_tokenize(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Get the Word2Vec embedding for the example text
example_embedding = get_word2vec_embedding_for_text(original_text_row, w2v_model)

# Print the Word2Vec values for the example
print("Original Text:", original_text_row)
print("Word2Vec Embedding:", example_embedding)