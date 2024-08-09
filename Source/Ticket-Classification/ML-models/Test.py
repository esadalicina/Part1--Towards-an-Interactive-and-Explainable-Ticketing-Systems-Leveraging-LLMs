import os
import time
import joblib
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
import nltk
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

print("Hypertunning with TF-idf without Stopwords")

nltk.download('punkt')

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)

# Extract the relevant columns
ticket_data = df_clean['complaint_what_happened_without_stopwords']
label_data = df_clean['category_encoded']
real_category_data = df_clean['category']
subcategory_data = df_clean['product']
complaint = df_clean['complaint_what_happened']

# Split the data into training and testing sets, keeping the original raw text
train_texts, test_texts, train_labels, test_labels, train_raw, test_raw, train_cat, test_cat, train_sub, test_sub = train_test_split(
    ticket_data, label_data, complaint,real_category_data, subcategory_data, test_size=0.3, random_state=42, shuffle=True
)

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
    'SVC': SVC()
}

# Define grid search parameters for different classifiers
parameters = {
    'SVC': {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf'],
        'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    }
}

# Define the base pipeline
def create_base_pipeline(classifier):
    return ImbPipeline([
        ('count', CountVectorizer()),
        ('tf', TfidfTransformer()),
        ('smote', SMOTE(random_state=42)),
        ('clf', classifier)
    ])

# Initialize variables to store the best model and its score
best_overall_model = None
best_overall_score = 0
best_count_vect = None
best_tfidf_transformer = None

# Iterate over classifiers
results = []
for clf_name, clf in classifiers.items():
    print(f"Training with {clf_name}...")
    
    # Create the base pipeline for the classifier
    base_pipeline = create_base_pipeline(clf)
    
    start_train_time = time.time()

    # Perform grid search
    gs_clf = GridSearchCV(base_pipeline, parameters[clf_name], n_jobs=-1, cv=5)
    gs_clf.fit(train_texts, train_labels)
    end_train_time = time.time()
    
    # Output the best score and parameters
    best_score = gs_clf.best_score_
    best_params = gs_clf.best_params_
    
    print(f'Best score: {best_score}')
    print(f'Best parameters: {best_params}')
    print(f'Training Time for {clf_name}: {end_train_time - start_train_time:.2f} seconds')

    # Retrieve the best model from RandomizedSearchCV
    best_model = gs_clf.best_estimator_

    if best_score > best_overall_score:
        best_overall_model = best_model
        best_overall_score = best_score
        best_count_vect = best_model.named_steps['count']  # type: ignore
        best_tfidf_transformer = best_model.named_steps['tf']  # type: ignore

    # Create the test pipeline without SMOTE
    test_pipeline = Pipeline([
        ('count', best_model.named_steps['count']),  # type: ignore
        ('tf', best_model.named_steps['tf']),  # type: ignore
        ('clf', best_model.named_steps['clf'])  # type: ignore
    ])

    # Transform the test data and make predictions
    start_test_time = time.time()
    test_predictions = test_pipeline.predict(test_texts)
    end_test_time = time.time()
    print(f'Test Evaluation Time for {clf_name}: {end_test_time - start_test_time:.2f} seconds')

    # Function to plot confusion matrix
    def plot_confusion_matrix(y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(filename)  # Save the plot to a file
        plt.close()  # Close the plot to prevent it from displaying

    # Identify misclassified tickets
    misclassified_tickets = []
    for i, (text, true_label, pred_label, real_category, subcategory, raw_text) in enumerate(
        zip(test_texts, test_labels, test_predictions, test_cat, test_sub, test_raw)):
        if true_label != pred_label:
            misclassified_tickets.append({
                'Text': text,
                'True Label': true_label,
                'Predicted Label': pred_label,
                'Real Category': real_category,
                'Subcategory': subcategory,
                'Original Raw Text': raw_text
            })

    # Create a DataFrame for the misclassified tickets
    misclassified_df = pd.DataFrame(misclassified_tickets)

    # Save the misclassified tickets to a CSV file
    misclassified_df.to_csv("/home/users/elicina/Master-Thesis/Diagrams/ML-Results/TF/misclassified_tickets_SVM_Real.csv", index=False)

    # Evaluate the model performance
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    report = classification_report(test_labels, test_predictions)

    # Store the results
    results.append({
        'Classifier': clf_name,
        'Best Score': best_score,
        'Best Parameters': best_params,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Training Time (s)': end_train_time - start_train_time,
        'Test Evaluation Time (s)': end_test_time - start_test_time
    })

    # Print the evaluation metrics
    print(f"Results for {clf_name}:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Classification Report:\n{report}\n')
