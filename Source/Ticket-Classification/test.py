from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')

# ---------------------------------------------------------------- Choose right columns ----------------------------------------------------

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category_encoded']

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.3, random_state=42, shuffle=True)


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

# Handle class imbalance using SMOTE on TF-IDF features
smote = SMOTE(random_state=42)
X_train_tf_resampled, train_labels_resampled = smote.fit_resample(X_train_tf, train_labels) # type: ignore

# ----------------------------------------------------------------- Tokenization with Word2Vec ----------------------------------------------------------

def Word2vec_method(train_texts):
    # Tokenize the training texts
    data = []
    for text in train_texts:
        # iterate through each sentence in the file
        for i in sent_tokenize(text):
            temp = []
            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())
            data.append(temp)
    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=2, workers=4)
    return w2v_model

# Function to average word vectors for each document
def get_word2vec_embeddings(texts, model):
    embeddings = []
    for text in texts:
        tokens = text.split()
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            vector = np.mean(vectors, axis=0)
        else:
            vector = np.zeros(model.vector_size)
        embeddings.append(vector)
    return np.array(embeddings)


def smoting(train_embeddings, train_labels):
    train_embeddings_resampled, train_labels_resampled_w2v = smote.fit_resample(train_embeddings, train_labels) # type: ignore
    return train_embeddings_resampled, train_labels_resampled_w2v


text_clf = Pipeline([
    ('vect', Word2vec_method()), # type: ignore
    ('emb', get_word2vec_embeddings()), # type: ignore
    ('smo', smoting()), # type: ignore
    ('clf', RandomForestClassifier())
])
text_clf = text_clf.fit(train_texts, train_labels)

if use_grid_search:
    
    parameters = {
        'vect': ,
        'emb': ,
        'smo': ,
        'clf__alpha': 
    }

    # Next, we create an instance of the grid search by passing the classifier, parameters
    # and n_jobs=-1 which tells to use multiple cores from user machine.
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5)
    gs_clf = gs_clf.fit(train_texts, train_labels)

    # To see the best mean score and the params, run the following code
    gs_clf.best_score_
    gs_clf.best_params_