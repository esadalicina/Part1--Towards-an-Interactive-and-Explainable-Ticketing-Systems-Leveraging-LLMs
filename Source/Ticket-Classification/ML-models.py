import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from Tokenization import *


# ---------------------------------------------------------- Train and Evaluate Classifiers --------------------------------------------------------

def train_and_evaluate_cv(X_train, y_train, classifier):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    reports = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        clf = classifier
        clf.fit(X_train_fold, y_train_fold)
        y_pred = clf.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_pred)
        report = classification_report(y_val_fold, y_pred)
        accuracies.append(accuracy)
        reports.append(report)
    
    # Average accuracy and report over folds
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy, reports

# List of classifiers to evaluate
classifier_list = [
    RandomForestClassifier(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    MultinomialNB(),
    XGBClassifier(),
    SVC(),
    KNeighborsClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
]

for classifier in classifier_list:
    accuracy_T_cv, reports_T_cv = train_and_evaluate_cv(X_train_tf_resampled, train_labels_resampled, classifier)
    print(f"Classifier: {classifier.__class__.__name__}")
    print("TF-IDF Cross-Validation Accuracy:", accuracy_T_cv)
    for i, report in enumerate(reports_T_cv):
        print(f"TF-IDF Cross-Validation Classification Report for fold {i+1}:\n", report)

    # Word2Vec results with cross-validation
    accuracy_W_cv, reports_W_cv = train_and_evaluate_cv(train_embeddings_resampled, train_labels_resampled_w2v, classifier)
    print("Word2Vec Cross-Validation Accuracy:", accuracy_W_cv)
    for i, report in enumerate(reports_W_cv):
        print(f"Word2Vec Cross-Validation Classification Report for fold {i+1}:\n", report)


    # Final evaluation on test set
    final_classifier = classifier
    final_classifier.fit(X_train_tf_resampled, train_labels_resampled)
    final_predictions = final_classifier.predict(X_test_tf)
    final_accuracy = accuracy_score(test_labels, final_predictions)
    final_report = classification_report(test_labels, final_predictions, zero_division=0)
    print("Final TF-IDF Test Accuracy:", final_accuracy)
    print("Final TF-IDF Test Classification Report:\n", final_report)

    # Final evaluation on test set with Word2Vec
    final_classifier_w2v = classifier
    final_classifier_w2v.fit(train_embeddings_resampled, train_labels_resampled_w2v)
    final_predictions_w2v = final_classifier_w2v.predict(test_embeddings)
    final_accuracy_w2v = accuracy_score(test_labels, final_predictions_w2v)
    final_report_w2v = classification_report(test_labels, final_predictions_w2v, zero_division=0)
    print("Final Word2Vec Test Accuracy:", final_accuracy_w2v)
    print("Final Word2Vec Test Classification Report:\n", final_report_w2v)