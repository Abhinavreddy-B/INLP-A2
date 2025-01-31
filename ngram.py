from tokenizer import Tokeniser
from tqdm import tqdm
from typing import List

class N_Gram_Model:
    __n = None
    __p_n = None

    def __init__(self, n):
        self.__n = n
        self.__p_n = {}

    def train(self, corpus: List[List[str]]):
        for sentence in tqdm(corpus, desc=f'Generating {self.__n} gram'):
            for i in range(len(sentence) - self.__n + 1):
                n_gram = tuple(sentence[i:i + self.__n])

                if n_gram in self.__p_n:
                    self.__p_n[n_gram] += 1
                else:
                    self.__p_n[n_gram] = 1

    def __getitem__(self, n_gram):
        if n_gram not in self.__p_n:
            return 0
        return self.__p_n[n_gram]
    
    def __contains__(self, n_gram):
        return n_gram in self.__p_n

    def get_all_n_grams(self):
        return self.__p_n
    
def generate_n_gram_model(n, corpus: List[List[str]]):
    model = N_Gram_Model(n)
    model.train(corpus)
    return model