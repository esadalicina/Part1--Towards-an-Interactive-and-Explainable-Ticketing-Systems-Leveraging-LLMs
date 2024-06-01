import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report

# Initialize the pipeline for zero-shot classification using the RoBERTa model
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category_encoded']

# Define the candidate labels
candidate_labels = label_data.unique()  # type: ignore # Use unique category encoded values as candidate labels

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []

print("1")

# Perform classification for each sequence in the dataset
for sequence, true_label in zip(ticket_data, label_data):
    print("2")
    result = classifier(sequence, candidate_labels)
    predicted_label = result['labels'][0] # type: ignore
    print("3")
    true_labels.append(true_label)
    print("4")
    predicted_labels.append(predicted_label)
    print("5")

print("6")
# Generate and print the classification report
report = classification_report(true_labels, predicted_labels, target_names=candidate_labels)
print(report)
