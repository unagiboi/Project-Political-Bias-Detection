import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report

def fine_tune_bert(X_train, y_train, X_test, y_test):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Assuming binary classification

    # Convert X_train and y_train to tensors
    X_train = torch.tensor(X_train)
    labels_train = torch.tensor(y_train)

    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train, labels_train)

    # Fine-tuning parameters
    batch_size = 8
    epochs = 3
    learning_rate = 2e-5

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Fine-tune the model
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            input_ids, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_bert_model')

    # Load the fine-tuned model
    fine_tuned_model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert_model')

    # Tokenize and encode the test data
    encoded_data_test = tokenizer.batch_encode_plus(
        X_test.tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

    # Create PyTorch datasets for test data
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(y_test.tolist())

    test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)

    # Create DataLoader for test data
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the fine-tuned model on test data
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())

    # Evaluate the model
    print("BERT (Fine-Tuned):")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

# You can add more functions or code as needed
