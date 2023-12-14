from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

def extract_bert_features(list_of_tokens):
    from torch.utils.data import Dataset, DataLoader

    class CustomDataset(Dataset):
        def __init__(self, tokenized_texts, labels):
            self.tokenized_texts = tokenized_texts
            self.labels = labels

        def __len__(self):
            return len(self.tokenized_texts)

        def __getitem__(self, idx):
            return {'input_ids': self.tokenized_texts[idx], 'labels': self.labels[idx]}

    # Example usage:
    dataset = CustomDataset(tokenized_texts, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    inputs_ids = []

    for tokens in list_of_tokens:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        #tokens_backtranslated = tokenizer.convert_ids_to_tokens(token_ids)
        inputs_ids.append(token_ids)

    print("Length of features")
    print(len(inputs_ids[0]))
    print(len(inputs_ids[1]))

    return inputs_ids