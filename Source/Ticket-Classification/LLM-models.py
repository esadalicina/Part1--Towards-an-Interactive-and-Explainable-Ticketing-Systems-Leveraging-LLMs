from sklearn.base import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load RoBERTa tokenizer and model
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize and encode text data
def tokenize_data(tokenizer, texts, max_length):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                      
                            add_special_tokens = True, 
                            max_length = max_length,           
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',     
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

# Train and evaluate RoBERTa and BERT models
def train_and_evaluate_model(model, tokenizer, train_texts, train_labels, test_texts, test_labels):
    # Tokenize and encode training data
    train_input_ids, train_attention_masks = tokenize_data(tokenizer, train_texts, max_length=128)
    train_labels_tensor = torch.tensor(train_labels)

    # Tokenize and encode test data
    test_input_ids, test_attention_masks = tokenize_data(tokenizer, test_texts, max_length=128)
    test_labels_tensor = torch.tensor(test_labels)

    # Create DataLoader for training and testing data
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels_tensor)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

    test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels_tensor)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(3):  # You can adjust the number of epochs
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    test_predictions = []
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        test_predictions.extend(predictions.tolist())

    # Calculate accuracy and classification report
    accuracy = accuracy_score(test_labels, test_predictions)
    report = classification_report(test_labels, test_predictions)
    return accuracy, report

# Train and evaluate RoBERTa model
roberta_accuracy, roberta_report = train_and_evaluate_model(roberta_model, roberta_tokenizer, train_test_split, train_labels, test_texts, test_labels)
print("RoBERTa Test Accuracy:", roberta_accuracy)
print("RoBERTa Test Classification Report:\n", roberta_report)

# Train and evaluate BERT model
bert_accuracy, bert_report = train_and_evaluate_model(bert_model, bert_tokenizer, train_texts, train_labels, test_texts, test_labels)
print("BERT Test Accuracy:", bert_accuracy)
print("BERT Test Classification Report:\n", bert_report)
