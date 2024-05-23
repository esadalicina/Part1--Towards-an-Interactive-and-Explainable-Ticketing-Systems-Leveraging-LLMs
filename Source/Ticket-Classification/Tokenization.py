from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------- Choose right columns ----------------------------------------------------

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
training_data = df_clean[['complaint_what_happened_lemmatized', 'category_encoded']]

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


def Word2vec_methode(training_data):
    pass

# Apply TF-IDF method
X_train_w2v = Word2vec_methode(training_data)

# Split the data into training and testing sets
X_train_W, X_test_W, y_train_W, y_test_W = train_test_split(X_train_tf, training_data['category_encoded'], test_size=0.2, random_state=42)



# ----------------------------------------------------------------------- Balanced data -----------------------------------------------------------------

# Before resampling
# print(f"Original shape of X_train: {X_train_T.shape}")
# print(f"Original shape of y_train: {y_train_T.shape}")

# Handle class imbalance using SMOTE
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train_T, y_train_T) # type: ignore

# Checking for class imbalance after handling imbalance
# fig = px.bar(x=pd.Series(y_resampled).value_counts().index,
             # y=pd.Series(y_resampled).value_counts(normalize=True),
             # title='Class Distribution in Training Set after Resampling')
# fig.write_html('plot1.html')

# Print the shape of the resampled data
# print(f'Shape of X_resampled: {X_resampled.shape}')
# print(f'Shape of y_resampled: {y_resampled.shape}')