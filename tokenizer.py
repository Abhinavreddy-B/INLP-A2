import sys
from typing import List

import re

import nltk
# nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer

import re
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from typing import List, Dict, Set

class Tokeniser:
    __frequency_treshold = 2

    __tokenizer = None
    __substitutions = None
    __vocab: Set[str] = set()
    __vocab_freq: Dict[str, int] = {}
    __word_to_index: Dict[str, int] = {}
    __index_to_word: Dict[int, str] = {}
    __vocab_size: int = 0

    def __init__(self):
        ## https://uibakery.io/regex-library/url-regex-python
        url_regex = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\\+.~#?&\/=]*)?'
        hashtag_regex = r'#\w+'
        mentions_regex = r'@\w+'
        percentage_regex = r'\d+\s*\%'
        range_regex = r'\d+\s*[-–]\s*\d+'

        self.__substitutions = [
            ["<URL>", url_regex],
            ["<HASHTAG>", hashtag_regex],
            ["<MENTION>", mentions_regex],
            ["<PERCENTAGE>", percentage_regex],
            ["<RANGE>", range_regex]
        ]

        self.__tokenizer = MWETokenizer([('<','URL','>'), ('<','HASHTAG','>'), ('<','MENTION','>'), ('<','PERCENTAGE','>'), ('<','RANGE','>')], separator='')

    def __scrap_unnecessary_corpus(self, corpus: str):
        ## Pride and Prejudice
        corpus = re.sub(r'CHAPTER *[A-Z]+\.', ' ', corpus)
        corpus = re.sub(r'END OF VOL\. *[A-Z]+\.', ' ', corpus)
        corpus = re.sub(r'VOL\. *[A-Z]+\.', ' ', corpus)
        corpus = re.sub(r'Section *\d+\.', ' ', corpus)
        corpus = re.sub(r'Mr\.', 'Mr', corpus)
        corpus = re.sub(r'Mrs\.', 'Mrs', corpus)

        ## Ulysses
        corpus = re.sub(r'e\.g\.', 'eg', corpus, flags=re.IGNORECASE)
        corpus = re.sub(r'_?\(([^\)]*)\)_?', lambda match: f"{match.group(1)}", corpus)
        corpus = re.sub(r'—(\w*)', lambda match: f"{match.group(1)}", corpus)
        corpus = re.sub(r'\.+', '.', corpus)
        corpus = re.sub(r'\?+', '?', corpus)

        corpus = corpus.lower()

        return corpus

    def __split_into_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)
    
    def __tokenise_into_words(self, text: str) -> List[List[str]]:
        text = self.__scrap_unnecessary_corpus(text)

        for [substitution, pattern] in self.__substitutions:
            text = re.sub(pattern, substitution, text)

        sentences = self.__split_into_sentences(text)

        return [self.__tokenizer.tokenize(word_tokenize(sentence)) for sentence in sentences]
    
    def tokenise_into_words(self, text: str) -> List[str]:
        text = self.__tokenise_into_words(text)
        transformed_text = []

        for sentence in text:
            transformed_sentence = []
            for word in sentence:
                if word in self.__vocab:
                    transformed_sentence.append(word)
                else:
                    transformed_sentence.append('<UNK>')
            transformed_text.append(transformed_sentence)
        
        return transformed_text

    def tokenise_into_sentence(self, text: str) -> List[str]:
        text = self.__scrap_unnecessary_corpus(text)
        return self.__split_into_sentences(text)

    def build_vocabulary(self, corpus: str):
        """
        Builds the vocabulary from the given corpus.
        """
        tokenized_sentences = self.__tokenise_into_words(corpus)

        for sentence in tokenized_sentences:
            self.__vocab.update(sentence)
            self.__vocab_freq.update({word: self.__vocab_freq.get(word, 0) + 1 for word in sentence})

        replaced_tokenized_sentences = [
            [
                word if self.__vocab_freq[word] >= self.__frequency_treshold else '<UNK>'
                for word in sentence
            ]
            for sentence in tokenized_sentences
        ]

        # Update the tokenized_sentences in place
        tokenized_sentences = replaced_tokenized_sentences

        # Filter the vocabulary in one pass
        self.__vocab = set([word for word in self.__vocab if self.__vocab_freq[word] > self.__frequency_treshold])

        self.__vocab.update(['<s>', '</s>', '<UNK>', '<PAD>'])

        self.__word_to_index = {word: idx for idx, word in enumerate(self.__vocab)}
        self.__index_to_word = {idx: word for word, idx in self.__word_to_index.items()}
        self.__vocab_size = len(self.__vocab)

    def word_to_index(self, word: str) -> int:
        """
        Converts a word to its corresponding index.
        """
        return self.__word_to_index.get(word, self.__word_to_index.get("<UNK>", -1))

    def index_to_word(self, index: int) -> str:
        """
        Converts an index to its corresponding word.
        """
        return self.__index_to_word.get(index, "<UNK>")

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        """
        return self.__vocab_size

    def encode(self, sentence: List[str]) -> List[int]:
        """
        Encodes a sentence (list of words) into a list of indices.
        """
        return [self.word_to_index(word) for word in sentence]

    def decode(self, indices: List[int]) -> List[str]:
        """
        Decodes a list of indices into a sentence (list of words).
        """
        return [self.index_to_word(index) for index in indices]

    def __len__(self):
        return self.vocab_size

    def state_dict(self):
        return {
            'word_to_index': self.__word_to_index,
            'index_to_word': self.__index_to_word,
            'vocab': self.__vocab,
            'vocab_freq': self.__vocab_freq,
            'vocab_size': self.__vocab_size
        }

    def load_state_dict(self, state_dict):
        self.__word_to_index = state_dict['word_to_index']
        self.__index_to_word = state_dict['index_to_word']
        self.__vocab = state_dict['vocab']
        self.__vocab_freq = state_dict['vocab_freq']
        self.__vocab_size = state_dict['vocab_size']

if __name__ == '__main__':
    with open('/dev/tty', 'w') as tty:
        print("Enter your text (press Ctrl+D or Ctrl+Z on Windows to finish):", file=tty)

    text = sys.stdin.read()

    tokeniser = Tokeniser()
    tokeniser.build_vocabulary(text)
    print('Vocab size:', tokeniser.vocab_size)

    tokens = tokeniser.tokenise_into_words(text)

    # print(tokens)
