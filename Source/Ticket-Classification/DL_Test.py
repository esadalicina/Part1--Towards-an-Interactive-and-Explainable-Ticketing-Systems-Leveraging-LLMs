import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from Tokenization import *

# Hyperparameters
embedding_dimension = 100
ngram_lengths = [2, 3, 4, 5]
num_filters = 200
min_length = 300  # Replace with the minimum document length in your dataset
num_words = len(train_embeddings_resampled)  # Assuming this is your vocabulary size
num_classes = 5  # Replace with the number of classes in your dataset

# Sample data (replace with your dataset)
X = torch.randint(0, num_words, (1000, min_length)) # type: ignore
y = torch.randint(0, num_classes, (1000,)) # type: ignore

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, ngram_lengths, num_filters, num_classes, min_length):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=ngram, padding='same'),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.AdaptiveMaxPool1d(1)
            ) for ngram in ngram_lengths
        ])
        
        self.fc = nn.Linear(num_filters * len(ngram_lengths), num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        convs = [conv(x).squeeze(2) for conv in self.convs]
        x = torch.cat(convs, 1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

model = TextCNN(vocab_size=num_words, embedding_dim=embedding_dimension, ngram_lengths=ngram_lengths, num_filters=num_filters, num_classes=num_classes, min_length=min_length)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    best_accuracy = 0.0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader.dataset)}, Accuracy: {accuracy}")

    print(f"Best Validation Accuracy: {best_accuracy}")
    model.load_state_dict(best_model)
    return model

# Train the model
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)
