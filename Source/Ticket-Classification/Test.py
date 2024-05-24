from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from scipy.stats import randint
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Load and preprocess data
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"
df_clean = pd.read_csv(file_path)
training_data = pd.DataFrame(df_clean[['complaint_what_happened_lemmatized', 'category_encoded']])

# Define the Word2Vec transformer
class Word2VecTransformer(FunctionTransformer):
    def __init__(self, size=100, window=5, min_count=1, sg=0):
        self.size = size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.w2v_model = None
        super().__init__(self.fit)

    def fit(self, X):
        self.w2v_model = Word2Vec(sentences=X, vector_size=self.size, window=self.window, min_count=self.min_count, sg=self.sg)
        return self

    def transform(self, X):
        return np.array([self.document_vector(doc) for doc in X]) # type: ignore

    def document_vector(self, doc):
        doc = [word for word in doc if word in self.w2v_model.wv.key_to_index]
        return np.mean(self.w2v_model.wv[doc], axis=0)


class SMOTETransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, X, y=None):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y) # type: ignore
        return X_resampled, y_resampled


# Define a custom scoring function
def custom_scorer(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Define the pipeline
pipeline = Pipeline([
    ('sampling', SMOTETransformer()),  # Using custom SMOTE transformer
    ('word2vec', Word2VecTransformer()),
    ('classifier', RandomForestClassifier())  # Using RandomForestClassifier
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(training_data['complaint_what_happened_lemmatized'], 
                                                    training_data['category_encoded'], test_size=0.2, random_state=42)

# Tokenize and transform training data
X_train = X_train.apply(lambda x: x.split())
X_test = X_test.apply(lambda x: x.split())

param_dist = {
    'word2vec__size': randint(50, 300),  # Range of vector_size values to try
    'word2vec__window': randint(2, 10)          # Range of window values to try
}

# Initialize RandomizedSearchCV with cross-validation
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, 
                                   scoring=make_scorer(custom_scorer), cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                                   n_iter=10, random_state=42, n_jobs=-1)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_

# Tokenize and transform test data
X_test = best_estimator.named_steps['word2vec'].transform(X_test) # type: ignore

# Evaluate the best estimator
y_pred = best_estimator.predict(X_test) # type: ignore
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Best Parameters:", best_params)
print("Test Accuracy:", accuracy)
print("Classification Report:\n", report)
