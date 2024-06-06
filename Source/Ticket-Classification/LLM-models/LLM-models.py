import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE


# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df_clean = pd.read_csv(file_path)


# Extract the relevant columns for training
ticket_data = df_clean['complaint_what_happened_lemmatized'].tolist()  # type: ignore
label_data = df_clean['category_encoded'].tolist()  # type: ignore

# Split the data into train, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.3, random_state=42)
t_texts, val_texts, t_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)



# Define the maximum length for BERT input (maximum length BERT can handle is 512 tokens)
max_length = 512

# Preprocess the data with padding and truncation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def encode_texts(texts):
    return tokenizer.batch_encode_plus(
        texts, 
        add_special_tokens=True, 
        return_attention_mask=True, 
        padding='max_length', 
        max_length=max_length,  # Use a fixed maximum length
        truncation=True,
        return_tensors='pt'
    )

train_encodings = encode_texts(t_texts)
val_encodings = encode_texts(val_texts)
test_encodings = encode_texts(test_texts)

train_labels = torch.tensor(t_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

# Create the DataLoaders
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels) # type: ignore
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, output_attentions=False, output_hidden_states=False)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Train the model with early stopping
epochs = 30
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for step, batch in enumerate(train_dataloader):
        batch_input_ids, batch_attention_masks, batch_labels = batch
        
        optimizer.zero_grad()  # Reset gradients
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()  # Update parameters

        predictions = torch.argmax(outputs.logits, dim=-1)
        correct_predictions += (predictions == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)
        
        if step % 10 == 0 and step > 0:
            print(f"  Batch {step} of {len(train_dataloader)}. Loss: {loss.item()}")
    
    avg_train_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_predictions / total_predictions
    print(f"Average training loss: {avg_train_loss}")
    print(f"Training accuracy: {train_accuracy}")
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    correct_val_predictions = 0
    total_val_predictions = 0
    for batch in val_dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch
        
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct_val_predictions += (predictions == batch_labels).sum().item()
            total_val_predictions += batch_labels.size(0)
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_accuracy = correct_val_predictions / total_val_predictions
    print(f"Average validation loss: {avg_val_loss}")
    print(f"Validation accuracy: {val_accuracy}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Load the best model for testing
model.load_state_dict(torch.load('best_model.pt'))

# Test the model
model.eval()
correct_test_predictions = 0
total_test_predictions = 0
with torch.no_grad():
    all_predictions = []
    for batch in test_dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch
        
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        predictions = torch.argmax(outputs.logits, dim=-1).flatten()
        correct_test_predictions += (predictions == batch_labels).sum().item()
        total_test_predictions += batch_labels.size(0)
        all_predictions.extend(predictions.cpu().numpy())
    
    test_accuracy = correct_test_predictions / total_test_predictions
    print("Test Accuracy:", test_accuracy)



# Save the trained model
# model.save_pretrained("/path/to/save/your/model")
# tokenizer.save_pretrained("/path/to/save/your/model")

# Load the model for inference
# model = BertForSequenceClassification.from_pretrained("/path/to/save/your/model")
# tokenizer = BertTokenizer.from_pretrained("/path/to/save/your/model")
