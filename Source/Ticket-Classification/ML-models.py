from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"
# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
training_data = df_clean[['complaint_what_happened_lemmatized', 'category_encoded']]

# Display the first few rows of the DataFrame
print(training_data.head(20))

unique_categories = df_clean['category'].unique() # type: ignore

count_vect = CountVectorizer()

# Write your code to get the Vector count
X_train_counts = count_vect.fit_transform(training_data['complaint_what_happened_lemmatized'])

# Write your code here to transform the word vector to tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tf = tfidf_transformer.fit_transform(X_train_counts)

# Checking for class imbalance
px.bar(x=training_data['category_encoded'].value_counts().index, y=training_data['category_encoded'].value_counts().values/max(training_data['category_encoded'].value_counts().values), title='Class Imbalance').show # type: ignore

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X_train_tf, training_data['category_encoded']) # type: ignore


# Prepare the training and test data
#train_X, test_X, train_y, test_y = train_test_split(X_res, y_res, test_size=0.2, random_state=40)


# Function to evaluate the model and display the results
#def eval_model(y_test, y_pred, y_pred_proba, type='Training'):
#    print(type, 'results')
#    print('Accuracy: ', accuracy_score(y_test, y_pred).round(2)) # type: ignore
#    print('Precision: ', precision_score(y_test, y_pred, average='weighted').round(2)) # type: ignore # type: ignore
#    print('Recall: ', recall_score(y_test, y_pred, average='weighted').round(2)) # type: ignore
#    print('F1 Score: ', f1_score(y_test, y_pred, average='weighted').round(2)) # type: ignore
#    print('ROC AUC Score: ', roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr').round(2)) # type: ignore
#    print('Classification Report: ', classification_report(y_test, y_pred))
#    cm = confusion_matrix(y_test, y_pred)
#    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=training_data['category_encoded'].unique()) # type: ignore
#    disp.plot()


# Function to grid search the best parameters for the model
#def run_model(model):
#    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
#    grid = GridSearchCV(model, param_grid={}, cv=cv, scoring='f1_weighted', verbose=1, n_jobs=-1)
#    grid.fit(train_X, train_y)
#    return grid.best_estimator_


# print("------------------------------------------# 1. Logistic Regression #-------------------------------------------")

#model1 = run_model(LogisticRegression())
#eval_model(train_y, model1.predict(train_X), model1.predict_proba(train_X), type='Training') # type: ignore
#eval_model(test_y, model1.predict(test_X), model1.predict_proba(test_X), type='Test') # type: ignore



# print("-----------------------------------------------# 2. Decision Tree #--------------------------------------------")

#model2 = run_model(DecisionTreeClassifier())
#eval_model(train_y, model2.predict(train_X), model2.predict_proba(train_X), type='Training') # type: ignore
#eval_model(test_y, model2.predict(test_X), model2.predict_proba(test_X), type='Test') # type: ignore
#
#
# print("---------------------------------------------# 3. Random Forest #--------------------------------------------")

#model3 = run_model(RandomForestClassifier())
#eval_model(train_y, model3.predict(train_X), model3.predict_proba(train_X), type='Training') # type: ignore
#eval_model(test_y, model3.predict(test_X), model3.predict_proba(test_X), type='Test') # type: ignore
#
#
# print("-----------------------------------------------# 4. Naive Bayes #----------------------------------------------")

#model4 = run_model(MultinomialNB())
#eval_model(train_y, model4.predict(train_X), np.exp(model4.predict_log_proba(train_X)), type='Training') # type: ignore
#eval_model(test_y, model4.predict(test_X), np.exp(model4.predict_log_proba(test_X)), type='Test') # type: ignore
#
#
# print("-------------------------------------------------# 5. XGBoost #------------------------------------------------")

#model5 = run_model(XGBClassifier(use_label_encoder=False))
#eval_model(train_y, model5.predict(train_X), model5.predict_proba(train_X), type='Training') # type: ignore
#eval_model(test_y, model5.predict(test_X), model5.predict_proba(test_X), type='Test') # type: ignore
#
#
# print("----------------------------------------------------# 6. SVM #-------------------------------------------------")

#model6 = run_model(SVC(probability=True))
#eval_model(train_y, model6.predict(train_X), model6.predict_proba(train_X), type='Training') # type: ignore
#eval_model(test_y, model6.predict(test_X), model6.predict_proba(test_X), type='Test') # type: ignore
#
#
# print("-------------------------------------------------# Conclusion #------------------------------------------------")
#
# # Applying the best model on the Custom Text
# # We will use the XGBoost model as it has the best performance
# df_complaints = pd.DataFrame({'complaints': [
#     "I can not get from chase who services my mortgage, who owns it and who has original loan docs",
#     "The bill amount of my credit card was debited twice. Please look into the matter and resolve at the earliest.",
#     "I want to open a salary account at your downtown branch. Please provide me the procedure.",
#     "Yesterday, I received a fraudulent email regarding renewal of my services.",
#     "What is the procedure to know my CIBIL score?",
#     "I need to know the number of bank branches and their locations in the city of Dubai"
# ]})
#
#
# def predict_lr(text):
#     Topic_names = {0: 'Credit Reporting and Debt Collection', 1: 'Credit Cards and Prepaid Cards',
#                    2: 'Bank Account or Service', 3: 'Loans', 4: 'Money Transfers and Financial Services'}
#     X_new_counts = count_vect.transform(text)
#     X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#     predicted = model5.predict(X_new_tfidf)
#     return Topic_names[predicted[0]]
#
#
# df_complaints['tag'] = df_complaints['complaints'].apply(lambda x: predict_lr([x]))
# print(df_complaints)


# ---------------------------------------------------- Save Model ------------------------------------------------------

# # Save the model
# joblib.dump(model, '/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Model/xgb_model.pkl')
#
# # Saving the objects
# joblib.dump(count_vect, '/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Model/count_vect.pkl')
# joblib.dump(tfidf_transformer, '/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Model/tfidf_transformer.pkl')

