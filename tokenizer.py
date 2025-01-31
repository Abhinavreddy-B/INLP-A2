import sys
from typing import List

import re

import nltk
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer, MWETokenizer

class Tokeniser:
    __tokenizer = None

    __substitutions = None

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
    
    def tokenise_into_words(self, text: str) -> List[List[str]]:
        text = self.__scrap_unnecessary_corpus(text)

        for [substitution, pattern] in self.__substitutions:
            text = re.sub(pattern, substitution, text)

        sentences = self.__split_into_sentences(text)

        return [self.__tokenizer.tokenize(word_tokenize(sentence)) for sentence in sentences]
    
    def tokenise_into_sentence(self, text: str) -> List[str]:
        text = self.__scrap_unnecessary_corpus(text)
        return self.__split_into_sentences(text)

if __name__ == '__main__':
    with open('/dev/tty', 'w') as tty:
        print("Enter your text (press Ctrl+D or Ctrl+Z on Windows to finish):", file=tty)

    text = sys.stdin.read()

    tokeniser = Tokeniser()
    tokens = tokeniser.tokenise_into_words(text)

    print(tokens)
