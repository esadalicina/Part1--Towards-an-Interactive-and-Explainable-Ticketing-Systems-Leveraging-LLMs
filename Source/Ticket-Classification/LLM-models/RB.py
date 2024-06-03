import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import torch
from imblearn.over_sampling import SMOTE

#test

# Specify the file path of your CSV file
file_path = "/home/users/elicina/Master-Thesis/Dataset/Cleaned_Dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

df_clean = df.dropna(subset=['complaint_what_happened_lemmatized'])

# Keep the columns "complaint_what_happened" & "category_encoded" only in the new dataframe --> training_data
ticket_data = df_clean['complaint_what_happened_lemmatized']
label_data = df_clean['category_encoded']

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(ticket_data, label_data, test_size=0.2, random_state=42, shuffle=True)


# Load RoBERTa tokenizer and model with the correct number of labels
num_labels = df_clean['category_encoded'].nunique()
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

# Load BERT tokenizer and model with the correct number of labels
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)


# Tokenize and encode text data
def tokenize_data(tokenizer, texts, max_length):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

# Train and evaluate RoBERTa and BERT models
def train_and_evaluate_model(model, tokenizer,train_texts, train_labels, test_texts, test_labels):

    print("1")
    # Tokenize and encode training data
    train_input_ids, train_attention_masks = tokenize_data(tokenizer, train_texts, max_length=300)
    train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)

    # Tokenize and encode test data
    test_input_ids, test_attention_masks = tokenize_data(tokenizer, test_texts, max_length=300)
    test_labels_tensor = torch.tensor(test_labels.values, dtype=torch.long)

    # Create DataLoader for training and testing data
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels_tensor)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

    test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels_tensor)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Move model to the appropriate device
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(20):  # You can adjust the number of epochs
        print("2")
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    test_predictions = []
    test_true_labels = []
    for batch in test_dataloader:
        print("3")
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        test_predictions.extend(predictions.cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and classification report
    accuracy = accuracy_score(test_true_labels, test_predictions)
    report = classification_report(test_true_labels, test_predictions)
    return accuracy, report


# Train and evaluate RoBERTa model
roberta_accuracy, roberta_report = train_and_evaluate_model(roberta_model, roberta_tokenizer, train_texts, train_labels, test_texts, test_labels)
print("RoBERTa Test Accuracy:", roberta_accuracy)
print("RoBERTa Test Classification Report:\n", roberta_report)

# Train and evaluate BERT model
#bert_accuracy, bert_report = train_and_evaluate_model(bert_model, bert_tokenizer, train_texts, train_labels, test_texts, test_labels)
#print("BERT Test Accuracy:", bert_accuracy)
#print("BERT Test Classification Report:\n", bert_report)



