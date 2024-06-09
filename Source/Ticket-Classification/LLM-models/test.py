# Import the necessary libraries
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# Preprocess the data
df_clean = pd.read_csv("/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv", header=None, names=["complaint_what_happened_lemmatized", "category_encoded"])
df = df_clean.sample(n=1000, random_state=42)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Split the dataset into training, validation, and testing sets
t_texts, test_texts, t_labels, test_labels = train_test_split(df["complaint_what_happened_lemmatized"], df["category_encoded"], test_size=0.3, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(t_texts, t_labels, test_size=0.1, random_state=42)

# Encode the data
train_encoded = tokenizer.batch_encode_plus(
    train_texts.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length', 
    max_length=256, 
    truncation=True,
    return_tensors='pt'
)
val_encoded = tokenizer.batch_encode_plus(
    val_texts.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length', 
    max_length=256, 
    truncation=True,
    return_tensors='pt'
)
test_encoded = tokenizer.batch_encode_plus(
    test_texts.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length', 
    max_length=256, 
    truncation=True,
    return_tensors='pt'
)

# Prepare training, validation, and testing data
train_input_ids = train_encoded['input_ids']
train_attention_masks = train_encoded['attention_mask']
train_labels = torch.tensor(train_labels.astype(int).values, dtype=torch.long)

val_input_ids = val_encoded['input_ids']
val_attention_masks = val_encoded['attention_mask']
val_labels = torch.tensor(val_labels.astype(int).values, dtype=torch.long)

test_input_ids = test_encoded['input_ids']
test_attention_masks = test_encoded['attention_mask']
test_labels = torch.tensor(test_labels.astype(int).values, dtype=torch.long)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, output_attentions=False, output_hidden_states=False)

# Define the training parameters
batch_size = 32
epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Create DataLoader for batch processing
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, dim=1).flatten()
    labels_flat = labels.flatten()
    return torch.sum(pred_flat == labels_flat) / len(labels_flat)

# Train the model
for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_accuracy = 0
    for batch in train_dataloader:
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
    avg_loss = total_loss / len(train_dataloader)
    avg_train_accuracy = train_accuracy / len(train_dataloader)
    
    # Evaluate the model on the validation set
    model.eval()
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids, b_attention_masks, b_labels = batch
            outputs = model(
                input_ids=b_input_ids,
                attention_mask=b_attention_masks
            )
            val_accuracy += calculate_accuracy(outputs.logits, b_labels).item()
    avg_val_accuracy = val_accuracy / len(val_dataloader)
    
    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Validation Accuracy: {avg_val_accuracy:.4f}")

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
