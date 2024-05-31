from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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

# Print the sample sizes
print(f"Number of samples in the training set: {len(train_texts)}")

class ShapePrinterBefore(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"Shape of dataset before SMOTE: {X.shape}")
        return X

# Define a custom transformer to print the shape
class ShapePrinterAfter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"Shape of dataset after SMOTE: {X.shape}")
        return X
    
# Define a custom scorer that logs validation set sizes
def custom_scorer(estimator, X, y):
    print(f"Validation set size: {len(X)}")
    return estimator.score(X, y)

# Define the classifiers to test
classifiers = {
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVC': SVC(),
    'DT': DecisionTreeClassifier()
}


# Define grid search parameters for different classifiers
parameters = {
    'RandomForest': {
        'clf__n_estimators': [100,200,500,700],
        'clf__min_samples_leaf': [5,10,30],
        'clf__max_depth': [None, 20, 30, 40]
    },
    'LogisticRegression': {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1'],
        'clf__solver': ['liblinear','saga']
    },
    'SVC': {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    },
    'DT': {
        'clf__max_depth': [None, 10, 20, 30]
    }
}


# Define the base pipeline
def create_base_pipeline(classifier):
    return ImbPipeline([
        ('count', CountVectorizer()),
        ('tf', TfidfTransformer()),
        # ('shape_printer_before', ShapePrinterBefore()),
        ('smote', SMOTE(random_state=42)),
        # ('shape_printer_after', ShapePrinterAfter()),
        ('clf', classifier)
    ])



# Iterate over classifiers
results = {}
for clf_name, clf in classifiers.items():
    print(f"Training with {clf_name}...")
    
    # Create the base pipeline for the classifier
    base_pipeline = create_base_pipeline(clf)
    
    # Perform grid search
    gs_clf = RandomizedSearchCV(base_pipeline, parameters[clf_name], n_jobs=-1, cv=5) # scoring=custom_scorer
    gs_clf.fit(train_texts, train_labels)
    
    # Output the best score and parameters
    best_score = gs_clf.best_score_
    best_params = gs_clf.best_params_
    
    print(f'Best score: {best_score}')
    print(f'Best parameters: {best_params}')

    # Retrieve the best model from RandomizedSearchCV
    best_model = gs_clf.best_estimator_

    # Create the test pipeline without SMOTE
    test_pipeline = Pipeline([
        ('count', best_model.named_steps['count']), # type: ignore
        ('tf', best_model.named_steps['tf']), # type: ignore
        ('clf', best_model.named_steps['clf']) # type: ignore
    ])

    # Transform the test data and make predictions
    test_predictions = test_pipeline.predict(test_texts)
    
    # Evaluate the model performance
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    report = classification_report(test_labels, test_predictions)
    
    # Store the results
    results[clf_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }
    
    # Print the evaluation metrics
    print(f"Results for {clf_name}:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Classification Report:\n{report}\n')

     # Print the cross-validation results
    print(f"Cross-validation results for {clf_name}:")
    cv_results_df = pd.DataFrame(gs_clf.cv_results_)
    print(cv_results_df[['mean_test_score', 'std_test_score', 'params']])

