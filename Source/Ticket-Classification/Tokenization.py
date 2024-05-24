from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import randint
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer


# ---------------------------------------------------------------- Choose right columns ----------------------------------------------------

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
training_data = pd.DataFrame(df_clean[['complaint_what_happened_lemmatized', 'category_encoded']])

# --------------------------------------------------------------- Displaying data inbalance --------------------------------------------------

# Checking for class imbalance before any processing
# fig = px.bar(x=training_data['category_encoded'].value_counts().index,
             # y=training_data['category_encoded'].value_counts(normalize=True),
             # title='Class Imbalance')
# fig.write_html('plot.html')


# ----------------------------------------------------------------- Tokenization with Tfidf ----------------------------------------------------------


def Tfidf_method(training_data):
    count_vect = CountVectorizer()
    # Write your code to get the Vector count
    X_train_counts = count_vect.fit_transform(training_data['complaint_what_happened_lemmatized'])

    # Write your code here to transform the word vector to tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_tf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tf, count_vect, tfidf_transformer


# Apply TF-IDF method
X_train_tf, count_vect, tfidf_transformer = Tfidf_method(training_data)

# Split the data into training and testing sets
X_train_T, X_test_T, y_train_T, y_test_T = train_test_split(X_train_tf, training_data['category_encoded'], test_size=0.2, random_state=42)



# ----------------------------------------------------------------- Tokenization with Word2Vec ----------------------------------------------------------


# Function to calculate document vector
def document_vector(doc, model):
    doc = [word for word in doc if word in model.wv.key_to_index]
    return np.mean(model.wv[doc], axis=0)

# Function to train Word2Vec model
def fit_word2vec(X, **kwargs):
    model = Word2Vec(sentences=X, **kwargs)
    return model

# Function to transform data to document vectors
def transform_word2vec(model):
    # Get vector representations for each document
    X_train_vectors = np.array([document_vector(doc, model) for doc in training_data['tokens']])
    X_train_vectors = normalize(X_train_vectors)
    return X_train_vectors


# Define the hyperparameter space
param_dist = {
    'size': randint(50, 300),  # Range of vector_size values to try
    'window': randint(2, 10),          # Range of window values to try
    'min_count': randint(2, 100)       # Range of min_count values to try
}

pipeline = Pipeline([
    ('word2vec', FunctionTransformer(func=fit_word2vec, validate=False))
])


# Tokenize the data
training_data['tokens'] = training_data['complaint_what_happened_lemmatized'].apply(lambda x: x.split())

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_dist, 
                                   n_iter=10, cv=5, random_state=42, scoring='accuracy')

# Perform random search
random_search.fit(training_data['tokens'])

# Best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Get best model
best_model = random_search.best_estimator_

# Transform the data using the best model
X_train_vectors = transform_word2vec(best_model)

# Split the data into training and testing sets
X_train_W, X_test_W, y_train_W, y_test_W = train_test_split(X_train_vectors, training_data['category_encoded'], test_size=0.2, random_state=42)



# ----------------------------------------------------------------------- Balanced data -----------------------------------------------------------------

# Checking for class imbalance after handling imbalance
# fig = px.bar(x=pd.Series(y_resampled_T).value_counts().index,
             # y=pd.Series(y_resampled_T).value_counts(normalize=True),
             # title='Class Distribution in Training Set after Resampling')
# fig.write_html('plot1.html')

# ---------------------------------------------------------- Train and Evaluate Methode --------------------------------------------------------

def train_and_evaluate_cv(X_train, y_train, classifier):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    reports = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled_fold, y_resampled_fold = smote.fit_resample(X_train_fold, y_train_fold) # type: ignore

        clf = classifier
        clf.fit(X_resampled_fold, y_resampled_fold)
        y_pred = clf.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred)
        report = classification_report(y_val_fold, y_pred)
        accuracies.append(accuracy)
        reports.append(report)
    
    # Average accuracy and report over folds
    avg_accuracy = np.mean(accuracies)
    avg_report = classification_report(y_val_fold, y_pred)
    return avg_accuracy, avg_report



# TF-IDF results with cross-validation
classifier = RandomForestClassifier()

accuracy_T_cv, report_T_cv = train_and_evaluate_cv(X_train_T, y_train_T, classifier)
print("TF-IDF Cross-Validation Accuracy:", accuracy_T_cv)
print("TF-IDF Cross-Validation Classification Report:\n", report_T_cv)

# Word2Vec results with cross-validation
accuracy_W_cv, report_W_cv = train_and_evaluate_cv(X_train_W, y_train_W, classifier)
print("Word2Vec Cross-Validation Accuracy:", accuracy_W_cv)
print("Word2Vec Cross-Validation Classification Report:\n", report_W_cv)