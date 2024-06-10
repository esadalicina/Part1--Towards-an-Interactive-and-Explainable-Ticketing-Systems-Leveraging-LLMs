# Import the necessary libraries
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset

# Preprocess the data
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)
df = df_clean.sample(n=15000, random_state=42)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

ticket_data = df['complaint_what_happened_lemmatized']
label_data = df['category_encoded']

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.3, random_state=42)

# Encode the training data
train_encoded = tokenizer.batch_encode_plus(
    train_texts.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length', 
    max_length=256, 
    truncation=True,
    return_tensors='pt'
)

# Encode the testing data
test_encoded = tokenizer.batch_encode_plus(
    test_texts.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length', 
    max_length=256, 
    truncation=True,
    return_tensors='pt'
)

train_input_ids = train_encoded['input_ids']
train_attention_masks = train_encoded['attention_mask']
train_labels = torch.tensor(train_labels.astype(int).values, dtype=torch.long)

test_input_ids = test_encoded['input_ids']
test_attention_masks = test_encoded['attention_mask']
test_labels = torch.tensor(test_labels.astype(int).values, dtype=torch.long)

# Define KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, dim=1).flatten()
    labels_flat = labels.flatten()
    return torch.sum(pred_flat == labels_flat) / len(labels_flat)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, output_attentions=False, output_hidden_states=False)

# Define the training parameters
batch_size = 32
epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Cross-validation loop
for fold, (train_index, val_index) in enumerate(kf.split(train_input_ids)):
    print(f"Fold {fold + 1}/{kf.n_splits}") # type: ignore

    # Split data into train and validation for this fold
    fold_train_inputs, fold_val_inputs = train_input_ids[train_index], train_input_ids[val_index]
    fold_train_masks, fold_val_masks = train_attention_masks[train_index], train_attention_masks[val_index]
    fold_train_labels, fold_val_labels = train_labels[train_index], train_labels[val_index]

    fold_train_dataset = TensorDataset(fold_train_inputs, fold_train_masks, fold_train_labels)
    fold_val_dataset = TensorDataset(fold_val_inputs, fold_val_masks, fold_val_labels)

    fold_train_dataloader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True)
    fold_val_dataloader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_accuracy = 0

        for batch in fold_train_dataloader:
            b_input_ids, b_attention_masks, b_labels = batch
            optimizer.zero_grad()
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_masks,
                labels=b_labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            train_accuracy += calculate_accuracy(outputs.logits, b_labels).item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(fold_train_dataloader)
        avg_train_accuracy = train_accuracy / len(fold_train_dataloader)

        # Evaluate the model on the validation set
        model.eval()
        val_accuracy = 0
        with torch.no_grad():
            for batch in fold_val_dataloader:
                b_input_ids, b_attention_masks, b_labels = batch
                outputs = model(
                    input_ids=b_input_ids,
                    attention_mask=b_attention_masks
                )
                val_accuracy += calculate_accuracy(outputs.logits, b_labels).item()

        avg_val_accuracy = val_accuracy / len(fold_val_dataloader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}")

# Prepare the test dataset and dataloader
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set
model.eval()
test_accuracy = 0
with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids, b_attention_masks, b_labels = batch
        outputs = model(
            input_ids=b_input_ids,
            attention_mask=b_attention_masks
        )
        test_accuracy += calculate_accuracy(outputs.logits, b_labels).item()

avg_test_accuracy = test_accuracy / len(test_dataloader)
print("Test Accuracy:", avg_test_accuracy)
