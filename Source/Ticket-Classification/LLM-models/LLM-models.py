# Import necessary libraries
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Clean the dataset by removing rows with NaN values in 'complaint_what_happened_lemmatized'
df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Extract the relevant columns for training
ticket_data = df_clean['complaint_what_happened_lemmatized'].tolist()  # type: ignore
label_data = df_clean['category_encoded'].tolist()  # type: ignore

print("Number of ticket data:", len(ticket_data))
print("Number of label data:", len(label_data))
print("Sample label data:", label_data[:10])  # Print first few labels for debugging
print("Unique label values:", set(label_data))

# Define the maximum length for BERT input (maximum length BERT can handle is 512 tokens)
max_length = 512

# Preprocess the data with padding and truncation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encoded_data = tokenizer.batch_encode_plus(
    ticket_data, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length', 
    max_length=max_length,  # Use a fixed maximum length
    truncation=True,
    return_tensors='pt'
)

input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(label_data)

# Create the DataLoader
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=16)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, output_attentions=False, output_hidden_states=False)


if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device('cuda')
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device('cpu')

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the model
epochs = 100

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        batch_input_ids, batch_attention_masks, batch_labels = tuple(t.to(device) for t in batch)
        
        optimizer.zero_grad()  # Reset gradients
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()  # Update parameters
        
        if step % 10 == 0 and step > 0:
            print(f"  Batch {step} of {len(dataloader)}. Loss: {loss.item()}")
    
    avg_train_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_train_loss}")

# Evaluate the model
model.eval()
with torch.no_grad():
    all_predictions = []
    for batch in DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32):
        batch_input_ids, batch_attention_masks, batch_labels = tuple(t.to(device) for t in batch)
        
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
        predictions = torch.argmax(outputs.logits, dim=-1).flatten()
        all_predictions.extend(predictions.cpu().numpy())
    
    accuracy = (torch.tensor(all_predictions) == labels).sum().item() / len(labels)
    print("Accuracy:", accuracy)

# Save the trained model
# model.save_pretrained("/path/to/save/your/model")
# tokenizer.save_pretrained("/path/to/save/your/model")

# Load the model for inference
# model = BertForSequenceClassification.from_pretrained("/path/to/save/your/model")
# tokenizer = BertTokenizer.from_pretrained("/path/to/save/your/model")
