import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from Tokenization import *

# Define deep learning models
def create_dnn_model(input_dim, output_dim):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_dim, output_dim):
    model = Sequential([
        Embedding(input_dim, 100, input_length=input_dim),
        LSTM(64, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# List of deep learning models to evaluate
model_list = [
    create_dnn_model(X_train_tf_resampled.shape[1], len(np.unique(train_labels_resampled))),
    create_rnn_model(X_train_tf_resampled.shape[1], len(np.unique(train_labels_resampled)))
]

# Function to train and evaluate deep learning models using cross-validation
def train_and_evaluate_cv(X_train, y_train, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    reports = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        y_pred = np.argmax(model.predict(X_val_fold), axis=-1)
        report = classification_report(y_val_fold, y_pred)
        accuracies.append(accuracy)
        reports.append(report)
    
    # Average accuracy and report over folds
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy, reports

# TF-IDF results with cross-validation
for model in model_list:
    accuracy_T_cv, reports_T_cv = train_and_evaluate_cv(X_train_tf_resampled, train_labels_resampled, model)
    print(f"Model: {model.name if hasattr(model, 'name') else model.__class__.__name__}")
    print("TF-IDF Cross-Validation Accuracy:", accuracy_T_cv)
    for i, report in enumerate(reports_T_cv):
        print(f"TF-IDF Cross-Validation Classification Report for fold {i+1}:\n", report)

    # Word2Vec results with cross-validation
    accuracy_W_cv, reports_W_cv = train_and_evaluate_cv(train_embeddings_resampled, train_labels_resampled_w2v, model)
    print("Word2Vec Cross-Validation Accuracy:", accuracy_W_cv)
    for i, report in enumerate(reports_W_cv):
        print(f"Word2Vec Cross-Validation Classification Report for fold {i+1}:\n", report)

    # Final evaluation on test set
    model.fit(X_train_tf_resampled, train_labels_resampled, epochs=10, batch_size=32, verbose=0)
    final_accuracy = model.evaluate(X_test_tf, test_labels, verbose=0)[1]
    y_pred = np.argmax(model.predict(X_test_tf), axis=-1)
    final_report = classification_report(test_labels, y_pred, zero_division=0)
    print("Final TF-IDF Test Accuracy:", final_accuracy)
    print("Final TF-IDF Test Classification Report:\n", final_report)

    # Final evaluation on test set with Word2Vec
    model.fit(train_embeddings_resampled, train_labels_resampled_w2v, epochs=10, batch_size=32, verbose=0)
    final_accuracy_w2v = model.evaluate(test_embeddings, test_labels, verbose=0)[1]
    y_pred_w2v = np.argmax(model.predict(test_embeddings), axis=-1)
    final_report_w2v = classification_report(test_labels, y_pred_w2v, zero_division=0)
    print("Final Word2Vec Test Accuracy:", final_accuracy_w2v)
    print("Final Word2Vec Test Classification Report:\n", final_report_w2v)
