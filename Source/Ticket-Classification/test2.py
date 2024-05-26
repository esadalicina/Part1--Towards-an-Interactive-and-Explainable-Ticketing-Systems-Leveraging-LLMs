import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

nltk.download('punkt')

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Clean the DataFrame
df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Extract the relevant columns
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category_encoded']

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.3, random_state=42, shuffle=True)

# Define the Word2Vec transformer
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, X, y=None):
        data = []
        for text in X:
            for sentence in sent_tokenize(text):
                words = [word.lower() for word in word_tokenize(sentence)]
                data.append(words)
        self.model = Word2Vec(sentences=data, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, 'model')
        embeddings = []
        for text in X:
            tokens = text.split()
            vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
            if vectors:
                vector = np.mean(vectors, axis=0)
            else:
                vector = np.zeros(self.vector_size)
            embeddings.append(vector)
        return np.array(embeddings)

# Define the pipeline
text_clf = ImbPipeline([
    ('w2v', Word2VecTransformer()), 
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier())
])

# Fit the pipeline
text_clf.fit(train_texts, train_labels)

# Define grid search parameters
parameters = {
    'w2v__vector_size': [100, 200],
    'w2v__window': [5, 10],
    'w2v__min_count': [1, 2],
}

# Perform grid search
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5)
gs_clf.fit(train_texts, train_labels)

# Output the best score and parameters
best_score = gs_clf.best_score_
best_params = gs_clf.best_params_

print(f'Best score: {best_score}')
print(f'Best parameters: {best_params}')


