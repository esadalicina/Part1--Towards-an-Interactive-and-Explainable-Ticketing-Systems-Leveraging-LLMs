import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Load the model
loaded_model = joblib.load('/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Model/xgb_model.pkl')
# Load the objects
vect = joblib.load('/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Model/count_vect.pkl')
transformer = joblib.load('/Users/esada/Documents/UNI.lu/MICS/Master-Thesis/Model/tfidf_transformer.pkl')


def predict_lr(text):
    Topic_names = {0:'Bank Account services', 1:'Others', 2:'Mortgage/Loan', 3:'Credit card or prepaid card', 4:'Theft/Dispute Reporting'}
    X_new_counts = vect.transform(text)
    X_new_tfidf = transformer.transform(X_new_counts)
    predicted = loaded_model.predict(X_new_tfidf)
    return Topic_names[predicted[0]]


