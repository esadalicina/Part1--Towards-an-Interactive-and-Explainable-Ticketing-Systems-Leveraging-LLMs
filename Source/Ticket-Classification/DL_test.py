import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from Tokenization import *

# Analyze text length distribution
text_lengths = [len(text.split()) for text in train_texts] 
max_text_length = max(text_lengths)
mean_text_length = np.mean(text_lengths)
median_text_length = np.median(text_lengths)

print("Maximum text length:", max_text_length)
print("Mean text length:", mean_text_length)
print("Median text length:", median_text_length)


# pad dataset to a maximum review length in words
X_train = sequence.pad_sequences(train_embeddings_resampled, maxlen=max_text_length)
X_test = sequence.pad_sequences(test_embeddings, maxlen=max_text_length)
print(X_train.shape)
print(X_test.shape)

VOCAB_SIZE = X_train.shape[1]
EMBED_SIZE = 100
EPOCHS=2
BATCH_SIZE=128
MAX_SEQUENCE_LENGTH = max_text_length


# create the model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, train_labels_resampled_w2v, 
          validation_split=0.1,
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          verbose=1) # type: ignore

# Final evaluation of the model
scores = model.evaluate(X_test, test_labels, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

predictions = model.predict_classes(X_test).ravel()
predictions[:10]

from sklearn.metrics import confusion_matrix, classification_report

labels = ['Bank Account or Service', 'Loans', 'Credit Cards and Prepaid Cards', 'Credit Reporting and Debt Collection', 'Money Transfers and Financial Services']
print(classification_report(test_labels, predictions))
pd.DataFrame(confusion_matrix(test_labels, predictions), index=labels, columns=labels)