import json
import os
import pandas as pd
import pathlib
from preprocessing import preprocess_dataset
from feature_engineering.tfidf import extract_tfidf_features
from feature_engineering.word2vec import extract_word2vec_features
import torch
from models.bert import fine_tune_bert


# CURRENTLY TESTING W ONE FILE
json_files_directory = 'Article-Bias-Prediction/data/testing'


if __name__ == "__main__":
    processed_data = preprocess_dataset(json_files_directory)
    print("PROCESSED DATA", processed_data)
    lemma_tokens = processed_data['lemmatized_tokens']

    # Initialize the DataFrame with the processed_data
    df = pd.DataFrame(processed_data)
    print("LENGTH OF TOKENS FOR", len(df['lemmatized_tokens'][0]))

    # Call the Word2Vec function
    word2vec_features = extract_word2vec_features(lemma_tokens)
    # print(word2vec_features)

    # Add Word2Vec features to df
    df['word2vec_features'] = word2vec_features

    # Call the TF-IDF function
    tfidf_df = extract_tfidf_features(lemma_tokens)

    tfidf_df = pd.DataFrame(tfidf_df.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_df.shape[1])])

    # Concatenate the TF-IDF features with the original DataFrame
    df = pd.concat([df, tfidf_df], axis=1)

    # Do further processing or print df
    print(df)

    # To split the dataset into train, validation, and test sets
    (train, val, test) = torch.utils.data.random_split(df, [0.8, 0.1, 0.1])

    #X_train should probably be word2vec features
    fine_tune_bert(X_train, y_train, X_test, y_test)

