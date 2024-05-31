# Import the necessary libraries
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification


# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category_encoded']

print("Number of ticket data:", len(ticket_data))
print("Number of label data:", len(label_data))
print("Sample label data:", label_data[:10])  # Print first few labels for debugging
print("Unique label values:", label_data.unique()) # type: ignore


# Preprocess the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encoded_data = tokenizer.batch_encode_plus(ticket_data, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=256, return_tensors='pt')
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = torch.tensor(label_data.values)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, output_attentions=False, output_hidden_states=False)

# Define the training parameters
batch_size = 16
epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(epochs):
    model.train()
    for i in range(0, input_ids.size(0), batch_size):
        optimizer.zero_grad()
        outputs = model(input_ids[i:i+batch_size], attention_mask=attention_masks[i:i+batch_size], labels=labels[i:i+batch_size])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_masks)
    predictions = torch.argmax(outputs[0], dim=1).flatten()
    accuracy = torch.sum(predictions == labels) / len(labels)

print("Accuracy:", accuracy.item())
