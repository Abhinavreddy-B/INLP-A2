import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from tokenizer import Tokeniser
from ngram import NGramModel, NGramDataset

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 128

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

    def train(self, till_epoch = EPOCHS, checkpoint_path = None):
        checkpoint = None
        if checkpoint_path is not None:
            checkpoint = self.__load_checkpoint(checkpoint_path)

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
            avg_loss = 0
            for i, (x, y) in tqdm(enumerate(self.__dataloader),desc='Batch', total=len(self.__dataloader), leave=False):
                self.__optimizer.zero_grad()
                outputs = self.__model(x)
                loss = self.__criterion(outputs, y)
                avg_loss = avg_loss + loss
                loss.backward()
                self.__optimizer.step()

                # if (i+1) % 100 == 0:
                #     print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(self.__dataloader)}], Loss: {loss.item():.4f}')

            self.__current_epoch = epoch
            self.__save_checkpoint()

            avg_loss /= len(self.__dataloader)
            t.set_postfix(loss=f'{avg_loss:.4f}', progress=f'{epoch+1}/{end_at_epoch}')
            tqdm.write(f'Trained Epoch [{epoch + 1}/{EPOCHS}], Average Loss {avg_loss:.4f}')

    def __save_checkpoint(self):
        checkpoint = {
            'epoch': self.__current_epoch,
            'tokenizer': self.__tokeniser.state_dict(),
            'ngram': self.__ngram.state_dict(),
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }

        torch.save(checkpoint, f'checkpoints/checkpoint_{self.__lm_type}_{self.__n}.pt')

    def __load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(f'{checkpoint_path}')
        return checkpoint

    def predict_next_word(self, input_sequence):
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
    
    # Parse the arguments
    args = parser.parse_args()

    model = NWP_Wrapper(args.lm_type, args.corpus_path, args.k)
    model.train()
    # model.train(checkpoint_path='checkpoints/checkpoint_f_3.pt')

    while True:
        input_sequence = input("Enter the input sequence (space-separated): ")
        next_word, probas = model.predict_next_word(input_sequence)

        for word, proba in zip(next_word, probas):
            print(f"Word: {word} \t| Probability: {proba:.4f}")

    # Here you would add the logic to handle the language model and text generation
    # based on the arguments provided.

if __name__ == "__main__":
    main()