from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

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
