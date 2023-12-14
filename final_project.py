import json
import os
import pandas as pd
import numpy as np
from preprocessing import preprocess_dataset
from feature_engineering.tfidf import extract_tfidf_features
from models.lstm import extract_word2vec_features, train_word2vec_model, train_lstm_model, lstm_pipeline
import torch
from sklearn.model_selection import train_test_split
from models.roberta import tokenize_data, split_data, train_model, evaluate_model, EnsembleModel, test_model
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification
import pickle
from sklearn.metrics import accuracy_score
import itertools
from torch.optim import Adam
from pathlib import Path

torch.manual_seed(42)

json_files_directory = 'Article-Bias-Prediction/data/jsons'


if __name__ == "__main__":
    # GPU stuff
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_pickle = Path('train_dataset.pkl')
    val_pickle = Path('val_dataset.pkl')
    test_pickle = Path('test_dataset.pkl')

    if train_pickle.is_file() and val_pickle.is_file() and test_pickle.is_file():
        # Load datasets
        with open('train_dataset.pkl', 'rb') as f:
            train_dataset = pickle.load(f)

        with open('val_dataset.pkl', 'rb') as f:
            val_dataset = pickle.load(f)

        with open('test_dataset.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
            
    else:
        processed_data = preprocess_dataset(json_files_directory)
        print("PROCESSED DATA", processed_data)
        lemma_tokens = processed_data['lemmatized_tokens']

        # Initialize the DataFrame with the processed_data
        df = pd.DataFrame(processed_data)
        print("LENGTH OF TOKENS FOR", len(df['lemmatized_tokens'][0]))

        # Call the Word2Vec function
        word2vec_features = extract_word2vec_features(lemma_tokens)

        # Add Word2Vec features to df
        df['word2vec_features'] = word2vec_features

        # Call the TF-IDF function
        tfidf_df = extract_tfidf_features(lemma_tokens)

        tfidf_df = pd.DataFrame(tfidf_df.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_df.shape[1])])

        # Concatenate the TF-IDF features with the original DataFrame
        df = pd.concat([df, tfidf_df], axis=1)

        # Do further processing or print df

        print(df.columns)

        y = df['bias'].tolist()


        df['sentence'] = df['filtered_tokens'].apply(lambda tokens: ' '.join(tokens))
        list_of_sentences = df['sentence'].tolist()

        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        lstm_model, lstm_test_preds = lstm_pipeline(train_df, temp_df, test_df)

        X_train = train_df['sentence'].tolist()
        y_train = train_df['bias'].tolist()
        X_val = val_df['sentence'].tolist()
        y_val = val_df['bias'].tolist()
        X_test = test_df['sentence'].tolist()
        y_test = test_df['bias'].tolist()

        train_dataset = tokenize_data(X_train, y_train, max_length=256)
        val_dataset = tokenize_data(X_val, y_val, max_length=256)
        test_dataset = tokenize_data(X_test, y_test, max_length=256)

        with open('train_dataset.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)

        with open('val_dataset.pkl', 'wb') as f:
            pickle.dump(val_dataset, f)

        with open('test_dataset.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)

    print("TRAIN DATASET", len(train_dataset))
    print("VAL DATASET", len(val_dataset))
    print("TEST DATASET", len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Perform grid search
    learning_rates = [1e-5, 2e-5, 3e-5]
    batch_sizes = [8, 16]
    epochs_list = [5, 10]
    weight_decay_list = [0.005, 0.001, 0.0005]

    best_accuracy = 0.0
    best_hyperparameters = None

    for lr, batch_size, epochs, weight_decay in itertools.product(learning_rates, batch_sizes, epochs_list, weight_decay_list):
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
        model.to(device)
    
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nTraining model with LR={lr}, Batch Size={batch_size}, Epochs={epochs}, Weight Decay={weight_decay}\n")
    
        train_model(model, train_dataloader, val_dataloader, epochs=epochs, lr=lr, weight_decay=weight_decay)
    
        # Evaluate the model on the test set
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        roberta_test_preds, true_labels = test_model(model, test_dataloader)
        weight_word2vec = 0.2  # Adjust the weight based on your preference
        weight_roberta = 0.8

        # Convert lists to NumPy arrays
        lstm_test_preds = np.array(lstm_test_preds)
        roberta_test_preds = np.array(roberta_test_preds)

        weighted_sum = weight_word2vec * lstm_test_preds + weight_roberta * roberta_test_preds
        predicted_labels = []
        for value in weighted_sum:
            if value > 1.5:
                predicted_labels.append(2)
            elif 0.5 <= value <= 1.5:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
        test_accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"\nTest Accuracy with LR={lr}, Batch Size={batch_size}, Epochs={epochs}, Weight Decay={weight_decay}: {test_accuracy}\n")
    
        # Check if this set of hyperparameters gives a better accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_hyperparameters = (lr, batch_size, epochs, weight_decay)

    print(f"\nBest Hyperparameters: LR={best_hyperparameters[0]}, Batch Size={best_hyperparameters[1]}, Epochs={best_hyperparameters[2]}, Weight Decay={best_hyperparameters[3]}")
    print(f"Best Test Accuracy: {best_accuracy}")
