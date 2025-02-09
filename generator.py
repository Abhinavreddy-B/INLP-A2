import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from tokenizer import Tokeniser
from ngram import NGramModel, NGramDataset
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import random

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

EMBEDDING_DIM = 50
HIDDEN_DIM = 128
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32

RNN_PAD_TOKEN = '<PAD>'

class RNNDataset(Dataset):
    def __init__(self, tokenized_sentences):
        self.input_target_pairs = self._create_sequences(tokenized_sentences)
    
    def _create_sequences(self, tokenized_sentences):
        input_target_pairs = []
        
        for sentence in tokenized_sentences:
            input_seq = sentence[:-1]
            target_seq = sentence[1:]
            input_target_pairs.append((torch.tensor(input_seq), torch.tensor(target_seq)))
        
        input_target_pairs.sort(key=lambda x: len(x[0]), reverse=True)

        return input_target_pairs
    
    def __len__(self):
        return len(self.input_target_pairs)
    
    def __getitem__(self, idx):
        input_seq, target = self.input_target_pairs[idx]
        
        # Convert to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return input_tensor, target_tensor

class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n):
        super(FFNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear((n-1) * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        hidden = self.activation(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden):
        # x: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        # RNN forward pass
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, sequence_length, hidden_dim)
        
        # Predict the next word using the last hidden state
        output = self.fc(output[:, -1, :])  # (batch_size, output_dim)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(1, batch_size, self.rnn.hidden_size)

class NWP_Base(ABC):
    _k: int

    _raw_corpus: str = None
    _corpus: List[List[str]] = None
    _vocabulary: set = None

    _tokenizer: Tokeniser = None
    _model: nn.Module = None

    _dataset: Dataset = None
    _dataloader: DataLoader = None

    _optimizer: optim.Optimizer = None
    _criterion: nn.Module = None

    def __init__(self, tokenizer: Tokeniser, k: int):
        super().__init__()
        self._tokenizer = tokenizer
        self._k = k

    @abstractmethod
    def _train_init(self, tokenized_corpus: List[List[str]], checkpoint):
        pass
    
    @abstractmethod
    def _train_step(self):
        pass
    
    @abstractmethod
    def _get_proba_next_word(self, input_sequence):
        pass

    def predict_next_word(self, input_sequence):
        self._model.eval()
        with torch.no_grad():
            input_sentences = self._tokenizer.tokenise_into_words(input_sequence)
            last_sentence = input_sentences[-1]

            output = self._get_proba_next_word(last_sentence)

            probabilities = torch.softmax(output, dim=1)
            top_k_prob, top_k_indices = torch.topk(probabilities, self._k)
            top_k_words = [self._tokenizer.index_to_word(index.item()) for index in top_k_indices[0]]
            return top_k_words, top_k_prob.flatten()
    
    def train(self, tokenized_corpus: List[List[str]], checkpoint = None, till_epoch = EPOCHS, save_checkpoint_path = None):
        if checkpoint is not None:
            self._current_epoch = checkpoint['epoch']
        else:
            self._current_epoch = -1

        self._train_init(tokenized_corpus, checkpoint)

        self._optimizer = optim.Adam(self._model.parameters(), lr=LEARNING_RATE)
        if checkpoint is not None:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if (self._current_epoch >= till_epoch):
            return

        start_from_epoch = self._current_epoch + 1
        end_at_epoch = till_epoch
        epoch = start_from_epoch

        t = trange(start_from_epoch, end_at_epoch, desc="Epoch Progress")
        t.set_postfix(loss=f'-', progress=f'{epoch}/{end_at_epoch}')

        for epoch in t:
            total_loss = self._train_step()

            self._current_epoch = epoch
            self.save_checkpoint(save_checkpoint_path)

            avg_loss = total_loss / len(self._dataloader)
            t.set_postfix(loss=f'{avg_loss:.4f}', progress=f'{epoch+1}/{end_at_epoch}')
            tqdm.write(f'Trained Epoch [{epoch + 1}/{end_at_epoch}], Average Loss {avg_loss:.4f}')

    def benchmark(self, tokenized_corpus: List[List[str]]):
        perp_list = []

        self._model.eval()
        with torch.no_grad():
            for sentence in tqdm(tokenized_corpus, desc='Running banchmark'):
                perplexity = 0.0
                count = 0
                for i in range(1, len(sentence)):
                    output = self._get_proba_next_word(sentence[:i])
                    probabilities = torch.softmax(output, dim=1)[0]
                    proba = probabilities[self._tokenizer.word_to_index(sentence[i])]
                    perplexity += np.log(proba)
                    count += 1

                perplexity = np.exp(-perplexity / count)
                perp_list.append(perplexity)

        return perp_list

    @abstractmethod
    def _type_specific_checkpoint_attr(self, checkpoint):
        pass

    def save_checkpoint(self, save_checkpoint_path):
        checkpoint = {
            'epoch': self._current_epoch,
            'tokenizer': self._tokenizer.state_dict(),
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }
        checkpoint = self._type_specific_checkpoint_attr(checkpoint)

        torch.save(checkpoint, save_checkpoint_path)

class NWP_FFNN(NWP_Base):
    __n :int= None
    __ngram :NGramModel = None

    def __init__(self, tokenizer, k, n):
        super().__init__(tokenizer, k)
        self.__n = n
        self.__ngram = NGramModel(self.__n)
    
    def _type_specific_checkpoint_attr(self, checkpoint):
        checkpoint['ngram'] = self.__ngram.state_dict()
        return checkpoint

    def _train_init(self, tokenized_corpus: List[List[str]], checkpoint):
        self._model = FFNN(
                        self._tokenizer.vocab_size,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        self._tokenizer.vocab_size,
                        self.__n)

        tokenized_corpus = [['<s>'] * (self.__n - 1) + sentence + ['</s>'] for sentence in tokenized_corpus]
        if checkpoint is not None:
            self.__ngram.load_state_dict(checkpoint['ngram'])
        else:
            self.__ngram.train(tokenized_corpus)

        if checkpoint is not None:
            self._model.load_state_dict(checkpoint['model_state_dict'])
        
        self._dataset = NGramDataset(self.__ngram, self._tokenizer, self.__n)
        self._dataloader = DataLoader(self._dataset, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)

        self._criterion = nn.CrossEntropyLoss()

    def _train_step(self):
        total_loss = 0
        for i, (x, y) in tqdm(enumerate(self._dataloader),desc='Batch', total=len(self._dataloader), leave=False):
            self._optimizer.zero_grad()
            outputs = self._model(x)
            loss = self._criterion(outputs, y)
            total_loss = total_loss + loss
            loss.backward()
            self._optimizer.step()
        return total_loss

    def _get_proba_next_word(self, input_sequence):
        last_sentence = ['<s>'] * (self.__n - 1) + input_sequence

        input_sequence = self._tokenizer.encode(last_sentence)
        input_sequence = input_sequence[-(self.__n-1):]

        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)
        output = self._model(input_tensor)
        return output

class NWP_RNN(NWP_Base):
    def __init__(self, tokenizer, k):
        super().__init__(tokenizer, k)

    def _type_specific_checkpoint_attr(self, checkpoint):
        return checkpoint

    def _train_init(self, tokenized_corpus: List[List[str]], checkpoint):
        self._model = RNN(
                        self._tokenizer.vocab_size,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        self._tokenizer.vocab_size)

        tokenized_corpus = [['<s>'] + sentence + ['</s>'] for sentence in tokenized_corpus]
        tokenized_corpus = [[self._tokenizer.word_to_index(word) for word in sentence] for sentence in tokenized_corpus]

        RNN_PAD_TOKEN_INT = self._tokenizer.word_to_index(RNN_PAD_TOKEN)

        if checkpoint is not None:
            self._model.load_state_dict(checkpoint['model_state_dict'])
        
        def collate_fn(batch):
            inputs, targets = zip(*batch)
            inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=RNN_PAD_TOKEN_INT)
            targets = pad_sequence(targets, batch_first=True, padding_value=RNN_PAD_TOKEN_INT)
            return inputs_padded, targets

        self._dataset = RNNDataset(tokenized_corpus)
        self._dataloader = DataLoader(self._dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, persistent_workers=True)

        self._criterion = nn.CrossEntropyLoss(ignore_index=self._tokenizer.word_to_index(RNN_PAD_TOKEN))

    def _train_step(self):
        total_loss = 0

        for i, (x, y) in tqdm(enumerate(self._dataloader),desc='Batch', total=len(self._dataloader), leave=False):
            self._optimizer.zero_grad()

            hidden = self._model.init_hidden(x.size(0))

            sequence_length = x.size(1)
            sequence_loss = 0

            for index in range(sequence_length):
                outputs, hidden = self._model(x[:, index].unsqueeze(1), hidden)
                loss = self._criterion(outputs, y[:, index])
                sequence_loss = sequence_loss + loss

            total_loss += sequence_loss.item() / sequence_length

            sequence_loss.backward()
            self._optimizer.step()

            hidden = hidden.detach()

        return total_loss

    def _get_proba_next_word(self, input_sequence):
        last_sentence = ['<s>'] + input_sequence
        input_sequence = self._tokenizer.encode(last_sentence)
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)  # Add batch dimension

        hidden = self._model.init_hidden(1)
        output, hidden = self._model(input_tensor, hidden)
        return output

class NWP_Wrapper:

    __lm_type = None
    __corpus_path = None
    __k = None
    __n = None

    __corpus: str = None

    __train_corpus_raw: List[str] = None
    __train_corpus: List[List[str]] = None

    __test_corpus_raw: List[str] = None
    __test_corpus: List[List[str]] = None

    __tokenizer: Tokeniser = None
    __model: NWP_Base = None

    def __init__(self, lm_type, corpus_path, k, n = 3):
        self.__lm_type = lm_type
        self.__corpus_path = corpus_path
        self.__k = k
        self.__n = n

        self.__tokenizer = Tokeniser()

        if self.__lm_type == 'f':
            self.__model = NWP_FFNN(self.__tokenizer, self.__k, self.__n)
        elif self.__lm_type == 'r':
            self.__model = NWP_RNN(self.__tokenizer, self.__k)
        else:
            raise ValueError(f'Invalid model type: {self.__lm_type}')

    def _learn(self, local_corpus, save_checkpoint_path, till_epoch = EPOCHS, load_checkpoint_path = None):
        checkpoint = None
        if load_checkpoint_path is not None:
            checkpoint = self.__load_checkpoint(load_checkpoint_path)
        if checkpoint is not None:
            self.__tokenizer.load_state_dict(checkpoint['tokenizer'])
        else:
            self.__tokenizer.build_vocabulary(local_corpus)
        
        tokenized_local_corpus = self.__tokenizer.tokenise_into_words(local_corpus)

        self.__model.train(tokenized_local_corpus, checkpoint, till_epoch, save_checkpoint_path)
    
    def train(self, save_checkpoint_path, till_epoch = EPOCHS, load_checkpoint_path = None, benchmark_file = None):

        with open(self.__corpus_path, 'r') as f:
            self.__corpus = f.read()

        corpus_sentences = self.__tokenizer.tokenise_into_sentence(self.__corpus)

        random.seed(RANDOM_SEED)
        random.shuffle(corpus_sentences)

        self.__train_corpus_sent = corpus_sentences[:-1000]
        self.__test_corpus_sent = corpus_sentences[-1000:]

        self.__train_corpus_raw = ' '.join(self.__train_corpus_sent)
        self.__test_corpus_raw = ' '.join(self.__test_corpus_sent)

        self._learn(self.__train_corpus_raw, save_checkpoint_path, till_epoch, load_checkpoint_path)

        self.__train_corpus = self.__tokenizer.tokenise_into_words(self.__train_corpus_raw)
        self.__test_corpus = self.__tokenizer.tokenise_into_words(self.__test_corpus_raw)

        if benchmark_file != None:
            train_benchmark_file = f'{benchmark_file}-train.txt'
            train_perplexity_list = self.__model.benchmark(self.__train_corpus)
            train_avg_perplexity = sum(train_perplexity_list) / len(train_perplexity_list)
            with open(train_benchmark_file, 'w') as f:
                print(f'{train_avg_perplexity}', file=f)

                for sentence, perplexity in zip(self.__train_corpus_sent, train_perplexity_list):
                    print(f'{sentence}\t{perplexity}', file=f)

            test_benchmark_file = f'{benchmark_file}-test.txt'
            test_perplexity_list = self.__model.benchmark(self.__test_corpus)
            test_avg_perplexity = sum(test_perplexity_list) / len(test_perplexity_list)
            with open(test_benchmark_file, 'w') as f:
                print(f'{test_avg_perplexity}', file=f)

                for sentence, perplexity in zip(self.__test_corpus_sent, test_perplexity_list):
                    print(f'{sentence}\t{perplexity}', file=f)

    def __load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(f'{checkpoint_path}')
        return checkpoint

    def predict_next_word(self, input_sequence):
        return self.__model.predict_next_word(input_sequence)

corpus_map = {
    'Pride and Prejudice - Jane Austen.txt': 'p',
    'Ulysses - James Joyce.txt': 'u'
}

def main():
    parser = argparse.ArgumentParser(description="Generate text using a specified language model.")

    parser.add_argument('lm_type', type=str, choices=['f', 'r', 'l'],
                        help="Type of language model: -f for FFNN, -r for RNN, -l for LSTM")
    parser.add_argument('corpus_path', type=str,
                        help="File path of the dataset to use for the language model")
    parser.add_argument('k', type=int,
                        help="Number of candidates for the next word to be printed")
    
    parser.add_argument('--load_checkpoint', type=str, default=None,
                    help="Path to load a pre-trained model checkpoint (optional)")
    parser.add_argument('--save_checkpoint', type=str, default=None,
                    help="Path to save the trained model checkpoint (optional)")
    parser.add_argument('--n', type=int, default=3,
                    help="Value of n for n-gram models (optional)")
    parser.add_argument('--epochs', type=int, default=10,
                    help="No of epochs to run for. (this includes the epochs of already trained model)")
    parser.add_argument('--benchmark_file', type=str, default=None,
                    help="Benchmarking runs only if this is provided")

    args = parser.parse_args()

    if args.save_checkpoint is None:
        args.save_checkpoint = f'checkpoints/checkpoint_{corpus_map[args.corpus_path.split('/')[-1]]}_{args.lm_type}_{args.n}.pt'

    model = NWP_Wrapper(args.lm_type, args.corpus_path, args.k, args.n)
    model.train(args.save_checkpoint, load_checkpoint_path=args.load_checkpoint, till_epoch=args.epochs, benchmark_file=args.benchmark_file)

    if args.benchmark_file is not None:
        return

    while True:
        input_sequence = input("Enter the input sequence (space-separated): ")
        if input_sequence == 'QUIT':
            break

        next_word, probas = model.predict_next_word(input_sequence)

        for word, proba in zip(next_word, probas):
            print(f"Word: {word} \t| Probability: {proba:.4f}")

if __name__ == "__main__":
    main()