import torch
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# test 
# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Drop rows with missing values in 'complaint_what_happened_lemmatized'
df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category_encoded']

# Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize and encode the training and test data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)

# Convert the encoded inputs to PyTorch tensors
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_labels.values)

test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_labels.values)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Define the training parameters
batch_size = 32
epochs = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(epochs):
    model.train()
    for i in range(0, len(train_inputs), batch_size):
        optimizer.zero_grad()
        outputs = model(input_ids=train_inputs[i:i+batch_size], attention_mask=train_masks[i:i+batch_size], labels=train_labels[i:i+batch_size])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(input_ids=test_inputs, attention_mask=test_masks)
    predictions = torch.argmax(outputs.logits, dim=1)
    accuracy = (predictions == test_labels).float().mean()

print("Accuracy:", accuracy.item())


# Save the trained model
# model.save_pretrained("/path/to/save/your/model")
# tokenizer.save_pretrained("/path/to/save/your/model")

# Load the model for inference
# model = BertForSequenceClassification.from_pretrained("/path/to/save/your/model")
# tokenizer = BertTokenizer.from_pretrained("/path/to/save/your/model")
