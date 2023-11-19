import re
import string
import nltk
import json
import os
import pandas as pd
import ssl
import pathlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_and_tokenize(text):
    # Remove HTML tags, URLs, and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)

    return tokens


def remove_stopwords(tokens):
    # Remove common stop words from the tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def apply_stemming(tokens):
    # Apply stemming using NLTK's Porter Stemmer
    porter_stemmer = PorterStemmer()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def apply_lemmatization(tokens):
    # Apply lemmatization using NLTK's WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def load_dataset(jsons_path):
    # Load dataset from JSON files
    with open(jsons_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def preprocess_dataset(json_files_directory):
    
    processed_data_list = []

    # Iterate through each JSON file in the directory
    for filename in os.listdir(json_files_directory):
        # Load JSON data from the file
        with open(os.path.join(json_files_directory, filename), 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        # Apply text processing to the 'content_original' field
        content_original = json_data.get('content_original', '')
        tokens = clean_and_tokenize(content_original)
        filtered_tokens = remove_stopwords(tokens)
        stemmed_tokens = apply_stemming(filtered_tokens)
        lemmatized_tokens = apply_lemmatization(filtered_tokens)

        # Append the processed tokens to the DataFrame
        processed_tokens_dict = {
            'original_tokens': tokens,
            'filtered_tokens': filtered_tokens,
            'stemmed_tokens': stemmed_tokens,
            'lemmatized_tokens': lemmatized_tokens
        }

        # Append the dictionary to the list
        processed_data_list.append(processed_tokens_dict)

    processed_data = pd.DataFrame(processed_data_list)
    return processed_data

# import json
# import os

# # Function definitions from the previous code snippet

# # Specify the directory where your JSON files are located
# # Specify the directory where you want to save the processed JSON files
# output_directory = '/Article-Bias-Prediction/data/processed_json'

# # Iterate through each JSON file in the directory
# for filename in os.listdir(json_files_directory):
#     # Load JSON data from the file
#     with open(os.path.join(json_files_directory, filename), 'r', encoding='utf-8') as file:
#         json_data = json.load(file)

#     # Apply text processing to the 'content_original' field
#     content_original = json_data.get('content_original', '')
#     tokens = clean_and_tokenize(content_original)
#     filtered_tokens = remove_stopwords(tokens)
#     stemmed_tokens = apply_stemming(filtered_tokens)
#     lemmatized_tokens = apply_lemmatization(filtered_tokens)

#     # Update the JSON data with the processed tokens
#     json_data['processed_tokens'] = {
#         'original_tokens': tokens,
#         'filtered_tokens': filtered_tokens,
#         'stemmed_tokens': stemmed_tokens,
#         'lemmatized_tokens': lemmatized_tokens
#     }

#     # Save the processed JSON data to a new file in the output directory
#     output_filename = f"processed_{filename}"
#     output_path = os.path.join(output_directory, output_filename)
#     with open(output_path, 'w', encoding='utf-8') as output_file:
#         json.dump(json_data, output_file, ensure_ascii=False, indent=2)

#     print(f"Processed and saved: {output_path}")