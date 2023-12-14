import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnsembleModel(torch.nn.Module):
    def __init__(self, models, device):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(models)
        self.device = device

    def forward(self, input_ids, attention_mask, labels):
        # Forward pass for each model in the ensemble
        logits_list = [model(input_ids, attention_mask).logits for model in self.models]
        
        # Take the average of logits as the final output
        logits_avg = torch.mean(torch.stack(logits_list), dim=0)

        outputs = {'logits': logits_avg, 'loss': None}
        return outputs

class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

from gensim.models import Word2Vec

# Method to tokenize and preprocess the input text
def tokenize_data(texts, labels, max_length=128):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    
    # Tokenize input texts and pad sequences
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt',
                      )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Find the maximum length within the batch
    max_len = max(len(ids[0]) for ids in input_ids)

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    labels = torch.tensor(labels, dtype=torch.long)

    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    labels = labels.to(device)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

# Method to split the dataset into training and validation sets
def split_data(input_ids, attention_masks, labels, validation_split=0.2, test_split=0.1):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    num_samples = len(dataset)
    
    # Calculate sizes for training, validation, and test sets
    val_size = int(validation_split * num_samples)
    test_size = int(test_split * num_samples)
    train_size = num_samples - val_size - test_size

    # Split the dataset
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    return train_dataset, val_dataset, test_dataset

# Method to train the BERT model
def train_model(model, train_dataloader, val_dataloader, epochs=10, lr=2e-5, weight_decay=1e-3):

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    early_stopping = EarlyStopping(patience=5, path='bert_model.pt')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            # Move batch to GPU
            inputs, attention_mask, labels = batch

            inputs = inputs.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            # loss = outputs.loss
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss}')

        # Validation
        model.eval()
        val_loss, val_accuracy = evaluate_model(model, val_dataloader)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step()


# Method to evaluate the BERT model
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:

            inputs, attention_mask, labels = batch

            inputs = inputs.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            labels = labels.to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            # loss = outputs.loss
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            # logits = outputs
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

def test_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:

            inputs, attention_mask, labels = batch

            inputs = inputs.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            labels = labels.to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            # loss = outputs.loss
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            # logits = outputs
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels