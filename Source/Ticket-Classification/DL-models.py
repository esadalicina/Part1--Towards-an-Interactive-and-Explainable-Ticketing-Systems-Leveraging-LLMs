import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D, Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from Tokenization import *


text_lengths = [len(text.split()) for text in train_texts] 
max_text_length = max(text_lengths)
mean_text_length = np.mean(text_lengths)
median_text_length = np.median(text_lengths)

print("Maximum text length:", max_text_length)
print("Mean text length:", mean_text_length)
print("Median text length:", median_text_length)

def create_dnn_model(input_dim, output_dim):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(vocab_size, output_dim, input_length):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=input_length),
        LSTM(64, return_sequences=True),
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def to_dense(data):
    if hasattr(data, "toarray"):
        return data.toarray()
    elif isinstance(data, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(data).numpy()
    return data

model_list = [
    create_dnn_model(train_embeddings.shape[1], len(np.unique(train_labels))),
    #create_rnn_model(vocab_size=train_embeddings.shape[1], output_dim=len(np.unique(train_labels)), input_length=max_text_length)
]

def train_and_evaluate_cv(X_train, y_train, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    reports = []
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        X_train_fold, X_val_fold = to_dense(X_train_fold), to_dense(X_val_fold)

        model.fit(X_train_fold, y_train_fold, epochs=200, batch_size=128, verbose=0)
        _, accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        y_pred = np.argmax(model.predict(X_val_fold), axis=-1)
        report = classification_report(y_val_fold, y_pred)
        accuracies.append(accuracy)
        reports.append(report)
    
    avg_accuracy = np.mean(accuracies)
    return avg_accuracy, reports, model

for model in model_list:

    accuracy_W_cv, reports_W_cv, model1 = train_and_evaluate_cv(to_dense(train_embeddings), train_labels, model)
    print("Word2Vec Cross-Validation Accuracy:", accuracy_W_cv)
    for i, report in enumerate(reports_W_cv):
        print(f"Word2Vec Cross-Validation Classification Report for fold {i+1}:\n", report)

    # model.fit(train_embeddings_resampled, train_labels_resampled_w2v, epochs=50, batch_size=64, class_weight=class_weights_dict, verbose=0) # type: ignore
    final_accuracy_w2v = model1.evaluate(to_dense(test_embeddings), test_labels, verbose=0)[1]
    y_pred_w2v = np.argmax(model.predict(to_dense(test_embeddings)), axis=-1)
    final_report_w2v = classification_report(test_labels, y_pred_w2v, zero_division=0)
    print("Final Word2Vec Test Accuracy:", final_accuracy_w2v)
    print("Final Word2Vec Test Classification Report:\n", final_report_w2v)

