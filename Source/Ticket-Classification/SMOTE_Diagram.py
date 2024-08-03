from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

# ---------------------------------------------------------------- Choose right columns ----------------------------------------------------

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data = df_clean['complaint_what_happened_without_stopwords']
label_data = df_clean['category_encoded']

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.2, random_state=42, shuffle=True)


# ----------------------------------------------------------------- Tokenization with Tfidf ----------------------------------------------------------

def Tfidf_method(training_data, count_vect=None, tfidf_transformer=None):
    if count_vect is None:
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(training_data)
        tfidf_transformer = TfidfTransformer()
        X_train_tf = tfidf_transformer.fit_transform(X_train_counts)
    else:
        X_train_counts = count_vect.transform(training_data)
        X_train_tf = tfidf_transformer.transform(X_train_counts)
    return X_train_tf, count_vect, tfidf_transformer

# Apply TF-IDF method to train and test data
X_train_tf, count_vect, tfidf_transformer = Tfidf_method(train_texts)
X_test_tf, _, _ = Tfidf_method(test_texts, count_vect, tfidf_transformer)


import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns


# Handle class imbalance using SMOTE on TF-IDF features
smote = SMOTE(random_state=42)
X_train_tf_resampled, train_labels_resampled = smote.fit_resample(X_train_tf, train_labels) # type: ignore

class_counts = Counter(train_labels_resampled)

# Sort classes by their count
classes, counts = zip(*sorted(class_counts.items()))
colors = sns.color_palette('viridis', len(class_counts))
# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color=colors)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Number of Samples per Class After SMOTE')
plt.xticks(classes) 
plt.savefig("/home/users/elicina/Master-Thesis/smote.png") # Ensure x-ticks are labeled with class names
plt.show()