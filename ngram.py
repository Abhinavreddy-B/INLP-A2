from tokenizer import Tokeniser
from tqdm import tqdm
from typing import List

from collections import defaultdict
from tqdm import tqdm
from typing import List, Tuple, Dict, Generator

import torch
from torch.utils.data import DataLoader, Dataset

class NGramModel:
    def __init__(self, n: int):
        """
        Initializes the n-gram model.
        
        Args:
            n (int): The size of the n-grams (e.g., 2 for bigrams, 3 for trigrams).
        """
        self.__n = n
        self.__n_gram_counts = defaultdict(int)  # Stores counts of n-grams
        self.__total_n_grams = 0  # Total number of n-grams in the corpus

    def train(self, corpus: List[List[str]]):
        """
        Trains the n-gram model on the given corpus.
        
        Args:
            corpus (List[List[str]]): A list of tokenized sentences.
        """
        for sentence in tqdm(corpus, desc=f"Training {self.__n}-gram model"):
            for i in range(len(sentence) - self.__n + 1):
                n_gram = tuple(sentence[i:i + self.__n])  # Convert list to tuple for hashability
                self.__n_gram_counts[n_gram] += 1
                self.__total_n_grams += 1

    @property
    def ngrams(self) -> Generator[Tuple[str], None, None]:
        for n_gram in self.__n_gram_counts:
            yield n_gram

    def get_count(self, n_gram: Tuple[str]) -> int:
        """
        Returns the count of a specific n-gram.
        
        Args:
            n_gram (Tuple[str]): The n-gram to query.
        
        Returns:
            int: The count of the n-gram.
        """
        return self.__n_gram_counts.get(n_gram, 0)

    def get_probability(self, n_gram: Tuple[str]) -> float:
        """
        Returns the probability of a specific n-gram.
        
        Args:
            n_gram (Tuple[str]): The n-gram to query.
        
        Returns:
            float: The probability of the n-gram.
        """
        if self.__total_n_grams == 0:
            return 0.0
        return self.__n_gram_counts.get(n_gram, 0) / self.__total_n_grams

    def get_all_n_grams(self) -> Dict[Tuple[str], int]:
        """
        Returns all n-grams and their counts.
        
        Returns:
            Dict[Tuple[str], int]: A dictionary of n-grams and their counts.
        """
        return self.__n_gram_counts

    def get_total_n_grams(self) -> int:
        """
        Returns the total number of n-grams in the model.
        
        Returns:
            int: The total number of n-grams.
        """
        return self.__total_n_grams

    def __contains__(self, n_gram: Tuple[str]) -> bool:
        """
        Checks if an n-gram exists in the model.
        
        Args:
            n_gram (Tuple[str]): The n-gram to check.
        
        Returns:
            bool: True if the n-gram exists, False otherwise.
        """
        return n_gram in self.__n_gram_counts

    def __getitem__(self, n_gram: Tuple[str]) -> int:
        """
        Returns the count of a specific n-gram (allows for dictionary-like access).
        
        Args:
            n_gram (Tuple[str]): The n-gram to query.
        
        Returns:
            int: The count of the n-gram.
        """
        return self.get_count(n_gram)

    def state_dict(self):
        return {
            'n_gram_counts': self.__n_gram_counts,
            'total_n_grams': self.__total_n_grams
        }

    def load_state_dict(self, state_dict):
        self.__n_gram_counts = state_dict['n_gram_counts']
        self.__total_n_grams = state_dict['total_n_grams']
            
class NGramDataset(Dataset):
    def __init__(self, ngram_model, tokenizer, n):
        self.ngram_model = ngram_model
        self.tokenizer = tokenizer
        self.n = n
        self.vocab_size = len(tokenizer)
        self.sequences = self._generate_sequences()

    def _generate_sequences(self):
        sequences = []
        for ngram in self.ngram_model.ngrams:
            tokens = [self.tokenizer.word_to_index(word) for word in ngram]
            sequences.append(tokens)
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[-1], dtype=torch.long)
        return x, y

def generate_n_gram_model(n, corpus: List[List[str]]):
    model = NGramModel(n)
    model.train(corpus)
    return model