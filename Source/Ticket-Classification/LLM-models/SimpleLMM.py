import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report
import os

# Configurations for distributed setup
pconfig = {
    "master_addr": os.getenv("MASTER_ADDR", "localhost"),
    "master_port": int(os.getenv("MASTER_PORT", 9994)),
    "rank": int(os.getenv("RANK", "0")),
    "local_rank": int(os.getenv("LOCAL_RANK", "0")),
    "world_size": int(os.getenv("WORLD_SIZE", "1"))
}
print(pconfig)

# Initialize the pipeline for zero-shot classification using the RoBERTa model
classifier = pipeline('zero-shot-classification', model='roberta-large-mnli')

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Drop rows with missing values in 'complaint_what_happened_lemmatized'
df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])


# Keep the columns "complaint_what_happened_lemmatized" & "category" only in the new dataframe
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category']

# Define the candidate labels
candidate_labels = label_data.unique() # type: ignore

# Lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Perform classification for each sequence in the dataset
for sequence, true_label in zip(ticket_data, label_data):
    try:
        result = classifier(sequence, candidate_labels)
        predicted_label = result['labels'][0] # type: ignore
        true_labels.append(true_label)  # Ensure true labels are strings
        predicted_labels.append(predicted_label)
    except Exception as e:
        print(f"Error processing sequence: {sequence}\nError: {e}")

# Generate and print the classification report
report = classification_report(true_labels, predicted_labels, target_names=candidate_labels)
print(report)
