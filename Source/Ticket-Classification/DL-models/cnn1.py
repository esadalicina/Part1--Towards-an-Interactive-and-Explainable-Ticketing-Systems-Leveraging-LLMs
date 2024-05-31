import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category_encoded']

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.2, random_state=42, shuffle=True)

X_train, X_val, Y_train, Y_val = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42, shuffle=True)

# Define vocabulary size and embedding dimension
vocab_size = 10000
embedding_dim = 50

# Calculate the maximum sequence length
max_seq_length = max(len(sequence.split()) for sequence in train_texts)

# Convert text data to sequences
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_texts)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Pad sequences to ensure uniform length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_seq_length, padding='post')

# Create the model
model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length))

# First Conv1D layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Second Conv1D layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Third Conv1D layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# Fourth Conv1D layer
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(GlobalMaxPooling1D())

# Flatten the output
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(5, activation='softmax'))

# Compile the model with sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
model.fit(X_train_pad, Y_train, epochs=100, batch_size=32, 
          validation_data=(X_val_pad, Y_val), callbacks=[early_stopping, reduce_lr]) 
