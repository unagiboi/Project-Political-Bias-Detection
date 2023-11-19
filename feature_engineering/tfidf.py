from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_tfidf_features(list_of_tokens):
    # Join the tokens for each document to form a list of strings
    corpus = [' '.join(tokens) for tokens in list_of_tokens]

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features parameter

    # Fit and transform the corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    # Convert the TF-IDF matrix to a DataFrame
    # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    return tfidf_matrix

# Example usage of the function
# if __name__ == "__main__":
#     # Example JSON data
#     example_tokens = ["mr.", "obama", "should", "draw", "the", "circle", "of", "inclusion", "..."]
    
#     # Call the function
#     tfidf_result = extract_tfidf_features(example_tokens)
    
#     # Print or do further processing with tfidf_result
#     print(tfidf_result)