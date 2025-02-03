import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset
from tokenizer import Tokeniser
from ngram import NGramModel, NGramDataset

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 128

RNN_SEQUENCE_LENGTH = 10
RNN_PAD_TOKEN_INT = -1

class RNNDataset(Dataset):
    def __init__(self, tokenized_sentences, sequence_length, pad_token):
        self.sequence_length = sequence_length
        self.pad_token = pad_token
        self.input_target_pairs = self._create_sequences(tokenized_sentences)
    
    def _create_sequences(self, tokenized_sentences):
        input_target_pairs = []
        
        for sentence in tokenized_sentences:
            # Split sentence into sequences of length `sequence_length`
            for i in range(len(sentence) - self.sequence_length):
                input_seq = sentence[i:i + self.sequence_length]
                target = sentence[i + self.sequence_length]
                input_target_pairs.append((input_seq, target))
        
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

class NWP_Wrapper:

    __lm_type = None
    __corpus_path = None
    __k = None
    __n = None

    __corpus = None
    __tokeniser = None
    __tokenized_corpus = None
    __ngram = None
    __dataset = None
    __dataloader = None
    __model = None
    __criterion = None
    __optimizer = None

    def __init__(self, lm_type, corpus_path, k, n = 3):
        self.__lm_type = lm_type
        self.__corpus_path = corpus_path
        self.__k = k
        self.__n = n

    def __FFNN_init(self, checkpoint = None):
        self.__tokenized_corpus = [['<s>'] * (self.__n - 1) + sentence + ['</s>'] for sentence in self.__tokenized_corpus]
        self.__ngram = NGramModel(self.__n)
        if checkpoint is not None:
            self.__ngram.load_state_dict(checkpoint['ngram'])
        else:
            self.__ngram.train(self.__tokenized_corpus)

        self.__model = FFNN(
                        self.__tokeniser.vocab_size,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        self.__tokeniser.vocab_size,
                        self.__n)

        if checkpoint is not None:
            self.__model.load_state_dict(checkpoint['model_state_dict'])
        
        self.__dataset = NGramDataset(self.__ngram, self.__tokeniser, self.__n)
        self.__dataloader = DataLoader(self.__dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    def __FFNN_pred(self, input_sequence :str):
        self.__model.eval()
        with torch.no_grad():
            input_sentences = self.__tokeniser.tokenise_into_words(input_sequence)
            last_sentence = input_sentences[-1]
            last_sentence = ['<s>'] * (self.__n - 1) + last_sentence

            input_sequence = [self.__tokeniser.word_to_index(word) for word in last_sentence]
            input_sequence = input_sequence[-(self.__n-1):]

            input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)
            output = self.__model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_k_prob, top_k_indices = torch.topk(probabilities, self.__k)
            top_k_words = [self.__tokeniser.index_to_word(index.item()) for index in top_k_indices[0]]
            return top_k_words, top_k_prob.flatten()

    def __FFNN_train(self):
        total_loss = 0
        for i, (x, y) in tqdm(enumerate(self.__dataloader),desc='Batch', total=len(self.__dataloader), leave=False):
            self.__optimizer.zero_grad()
            outputs = self.__model(x)
            loss = self.__criterion(outputs, y)
            total_loss = total_loss + loss
            loss.backward()
            self.__optimizer.step()
        return total_loss

    def __RNN_init(self, checkpoint = None):
        self.__tokenized_corpus = [['<s>'] + sentence + ['</s>'] for sentence in self.__tokenized_corpus]
        self.__tokenized_corpus = [[self.__tokeniser.word_to_index(word) for word in sentence] for sentence in self.__tokenized_corpus]

        self.__model = RNN(
                        self.__tokeniser.vocab_size,
                        EMBEDDING_DIM,
                        HIDDEN_DIM,
                        self.__tokeniser.vocab_size)

        if checkpoint is not None:
            self.__model.load_state_dict(checkpoint['model_state_dict'])
        
        def collate_fn(batch):
            inputs, targets = zip(*batch)
            
            # Pad input sequences to the same length
            inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=RNN_PAD_TOKEN_INT)
            
            # Convert targets to a tensor
            targets = torch.tensor(targets, dtype=torch.long)
            
            return inputs_padded, targets

        self.__dataset = RNNDataset(self.__tokenized_corpus, RNN_SEQUENCE_LENGTH, RNN_PAD_TOKEN_INT)
        self.__dataloader = DataLoader(self.__dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    def __RNN_train(self):
        total_loss = 0
        hidden = self.__model.init_hidden(BATCH_SIZE)

        for i, (x, y) in tqdm(enumerate(self.__dataloader),desc='Batch', total=len(self.__dataloader), leave=False):
            self.__optimizer.zero_grad()

            if hidden.size(1) != x.size(0):  
                hidden = hidden[:, :x.size(0), :].detach()

            # print('Forward pass', x.shape, hidden.shape)
            outputs, hidden = self.__model(x, hidden)
            loss = self.__criterion(outputs, y)
            total_loss = total_loss + loss
            loss.backward()
            self.__optimizer.step()

            hidden = hidden.detach()

        return total_loss
    
    def __RNN_pred(self, input_sequence :str):
        self.__model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            # Tokenize the input sequence into words
            input_sentences = self.__tokeniser.tokenise_into_words(input_sequence)
            last_sentence = input_sentences[-1]  # Get the last sentence
            
            # Pad the last sentence with <s> tokens to match the context length (n-1)
            last_sentence = ['<s>'] * (self.__n - 1) + last_sentence
            
            # Convert words to indices
            input_sequence = [self.__tokeniser.word_to_index(word) for word in last_sentence]
            input_sequence = input_sequence[-(self.__n - 1):]  # Truncate to the last (n-1) words
            
            # Convert input sequence to a tensor
            input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            
            # Initialize hidden state for the RNN
            hidden = self.__model.init_hidden(1)  # Batch size is 1 for inference
            
            # Forward pass through the RNN
            output, hidden = self.__model(input_tensor, hidden)
            
            # Compute probabilities using softmax
            probabilities = torch.softmax(output, dim=1)
            
            # Get the top-k predicted words and their probabilities
            top_k_prob, top_k_indices = torch.topk(probabilities, self.__k)
            top_k_words = [self.__tokeniser.index_to_word(index.item()) for index in top_k_indices[0]]
            
            return top_k_words, top_k_prob.flatten()

    def train(self, save_checkpoint_path, till_epoch = EPOCHS, load_checkpoint_path = None):
        checkpoint = None
        if load_checkpoint_path is not None:
            checkpoint = self.__load_checkpoint(load_checkpoint_path)

        with open(self.__corpus_path, 'r') as f:
            self.__corpus = f.read()


        if checkpoint is not None:
            self.__current_epoch = checkpoint['epoch']
        else:
            self.__current_epoch = -1


        self.__tokeniser = Tokeniser()
        if checkpoint is not None:
            self.__tokeniser.load_state_dict(checkpoint['tokenizer'])
        else:
            self.__tokeniser.build_vocabulary(self.__corpus)
        self.__tokenized_corpus = self.__tokeniser.tokenise_into_words(self.__corpus)

        if self.__lm_type == 'f':
            self.__FFNN_init(checkpoint)
        elif self.__lm_type == 'r':
            self.__RNN_init(checkpoint)
        else:
            raise ValueError(f"Invalid language model type: {self.__lm_type}")

        self.__criterion = nn.CrossEntropyLoss()

        self.__optimizer = optim.Adam(self.__model.parameters(), lr=LEARNING_RATE)
        if checkpoint is not None:
            self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if (self.__current_epoch >= till_epoch):
            return

        start_from_epoch = self.__current_epoch + 1
        end_at_epoch = till_epoch
        epoch = start_from_epoch

        t = trange(start_from_epoch, end_at_epoch, desc="Epoch Progress")
        t.set_postfix(loss=f'-', progress=f'{epoch + 1}/{end_at_epoch}')

        for epoch in t:
            if self.__lm_type == 'f':
                total_loss = self.__FFNN_train()
            elif self.__lm_type == 'r':
                total_loss = self.__RNN_train()
            else:
                raise ValueError(f"Invalid language model type: {self.__lm_type}")

            self.__current_epoch = epoch
            self.__save_checkpoint(save_checkpoint_path)

            avg_loss = total_loss / len(self.__dataloader)
            t.set_postfix(loss=f'{avg_loss:.4f}', progress=f'{epoch+1}/{end_at_epoch}')
            tqdm.write(f'Trained Epoch [{epoch + 1}/{EPOCHS}], Average Loss {avg_loss:.4f}')

    def __save_checkpoint(self, save_checkpoint_path):
        checkpoint = {
            'epoch': self.__current_epoch,
            'tokenizer': self.__tokeniser.state_dict(),
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }
        if(self.__lm_type == 'f'):
            checkpoint['ngram'] = self.__ngram.state_dict()

        torch.save(checkpoint, save_checkpoint_path)

    def __load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(f'{checkpoint_path}')
        return checkpoint

    def predict_next_word(self, input_sequence):
        if self.__lm_type == 'f':
            return self.__FFNN_pred(input_sequence)
        elif self.__lm_type == 'r':
            return self.__RNN_pred(input_sequence)

corpus_map = {
    'Pride and Prejudice - Jane Austen.txt': 'p',
    'Ulysses - James Joyce.txt': 'u'
}

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate text using a specified language model.")
    
    # Add arguments
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
                    help="No of epochs to run from. (this includes the epochs of already trained model)")

    # Parse the arguments
    args = parser.parse_args()

    if args.save_checkpoint is None:
        args.save_checkpoint = f'checkpoints/checkpoint_{corpus_map[args.corpus_path.split('/')[-1]]}_{args.lm_type}_{args.n}.pt'

    model = NWP_Wrapper(args.lm_type, args.corpus_path, args.k, args.n)
    model.train(args.save_checkpoint, load_checkpoint_path=args.load_checkpoint)
    # model.train(checkpoint_path='checkpoints/checkpoint_f_3.pt')

    while True:
        input_sequence = input("Enter the input sequence (space-separated): ")
        next_word, probas = model.predict_next_word(input_sequence)

        for word, proba in zip(next_word, probas):
            print(f"Word: {word} \t| Probability: {proba:.4f}")

if __name__ == "__main__":
    main()