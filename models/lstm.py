from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def extract_word2vec_features(list_of_tokens):
    word2vec_model = Word2Vec(sentences=list_of_tokens, vector_size=100, window=5, min_count=1, workers=4)

    # Function to get the average word vector for a document
    def get_average_word_vector(tokens, model, vector_size):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        if not vectors:
            return [0] * vector_size
        return sum(vectors) / len(vectors)
        
    # Apply the function to each document
    return [get_average_word_vector(tokens, word2vec_model, 100) for tokens in list_of_tokens]

def train_word2vec_model(list_of_tokens, vector_size=100, window=5, min_count=1, workers=4):
    """
    Train a Word2Vec model on a list of tokenized sentences.

    Parameters:
    - list_of_tokens: List of tokenized sentences.
    - vector_size: Dimensionality of the word vectors.
    - window: Maximum distance between the current and predicted word within a sentence.
    - min_count: Ignores all words with a total frequency lower than this.
    - workers: Number of CPU cores to use when training the model.

    Returns:
    - Trained Word2Vec model.
    """
    # Create and train the Word2Vec model
    word2vec_model = Word2Vec(
        sentences=list_of_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )

    return word2vec_model

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_lstm_model(train_loader, val_loader, input_size, hidden_size, output_size, num_layers, lr, epochs):
    model = SimpleLSTM(input_size, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")

    return model

def lstm_pipeline(train_df, val_df, test_df, input_size=100, hidden_size=128, output_size=3, num_layers=2, lr=0.001, epochs=10, batch_size=16):
    # Define DataLoader for each set
    train_dataset = TensorDataset(torch.tensor(train_df['word2vec_features'].tolist()), torch.tensor(train_df['bias'].tolist()))
    val_dataset = TensorDataset(torch.tensor(val_df['word2vec_features'].tolist()), torch.tensor(val_df['bias'].tolist()))
    test_dataset = TensorDataset(torch.tensor(test_df['word2vec_features'].tolist()), torch.tensor(test_df['bias'].tolist()))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the LSTM model
    lstm_model = train_lstm_model(train_loader, val_loader, input_size, hidden_size, output_size, num_layers, lr, epochs)

    # Evaluate the model on the test set
    test_preds = predict_lstm_model(lstm_model, test_loader)

    return lstm_model, test_preds

def predict_lstm_model(model, dataloader):
    model.eval()
    all_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds