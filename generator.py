import torch
from torch.utils.data import DataLoader, Dataset

class NGramDataset(Dataset):
    def __init__(self, ngram_model, tokenizer, n):
        self.ngram_model = ngram_model
        self.tokenizer = tokenizer
        self.n = n
        self.vocab_size = len(tokenizer)
        self.sequences = self._generate_sequences()

    def _generate_sequences(self):
        sequences = []
        for ngram in self.ngram_model.ngrams(self.n):
            tokens = [self.tokenizer.word_to_index[word] for word in ngram]
            sequences.append(tokens)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[-1], dtype=torch.long)
        return x, y