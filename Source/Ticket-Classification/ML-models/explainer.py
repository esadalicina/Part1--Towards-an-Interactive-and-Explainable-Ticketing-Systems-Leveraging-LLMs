import joblib
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt


# Load the model and required objects with error handling
try:
    loaded_model = joblib.load('/home/users/elicina/Master-Thesis/Models/MLmodel/Shap/modelML.pkl')
    tfidf_vectorizer = joblib.load('/home/users/elicina/Master-Thesis/Models/MLmodel/Shap/tfidf_transformer.pkl')
    explainer = joblib.load('/home/users/elicina/Master-Thesis/Models/MLmodel/Shap/explainer.pkl')
except Exception as e:
    print(f"Error loading model or objects: {e}")
    raise

# Classify the new tickets
def predict_lr(text):
    Topic_names = {
        0: 'Credit Reporting and Debt Collection',
        1: 'Credit Cards and Prepaid Cards',
        2: 'Bank Account or Service',
        3: 'Loans',
        4: 'Money Transfers and Financial Services'
    }
    X_new_tfidf = tfidf_vectorizer.transform(text)
    predicted = loaded_model.predict(X_new_tfidf)
    predicted_proba = loaded_model.predict_proba(X_new_tfidf)
    return Topic_names[predicted[0]], predicted[0], predicted_proba

# Function to explain predictions using KernelExplainer
def explain_texts(texts):
     X_tfidf = tfidf_vectorizer.transform(texts)
     shap_values = explainer.shap_values(X_tfidf)
     return shap_values



# Simplified SHAP explanation plot
def plot_shap_values(shap_values, feature_names, class_index=0):
    # Extract shap_values for the specified class and reshape to 2D
    shap_values_class = shap_values[0][:, class_index].reshape(1, -1)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_class, feature_names=feature_names)
    plt.show()

# Example usage
new_texts = ["I am having trouble logging into my bank account. I have tried multiple times and I have received no response. I tried resetting my password, but that did not seem to work either. I am concerned that there may be an issue with my account security or that someone has accessed my account without my permission."]

prediction, label, prob = predict_lr(new_texts)
print(f"Prediction: {prediction}")
print(f"Predicted class index: {label}")
print(f"Prediction probabilities: {prob}")

# Explain the prediction for the single text
shap_values = explain_texts(new_texts)

# Verify the structure of shap_values
print(f"Shape of shap_values: {np.shape(shap_values)}")

# Plot SHAP values for the predicted class
plot_shap_values(shap_values, tfidf_vectorizer.get_feature_names_out(), class_index=label)
