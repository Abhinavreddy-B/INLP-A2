# Usage

```sh
usage: generator.py [-h] [--load_checkpoint LOAD_CHECKPOINT] [--save_checkpoint SAVE_CHECKPOINT] [--n N]  [--epochs EPOCHS] [--benchmark_file BENCHMARK_FILE] {f,r,l} corpus_path k

Generate text using a specified language model.

positional arguments:
  {f,r,l}               Type of language model: f for FFNN, r for RNN, l for LSTM
  corpus_path           File path of the dataset to use for the language model
  k                     Number of candidates for the next word to be printed

options:
  -h, --help            show this help message and exit
  --load_checkpoint LOAD_CHECKPOINT
                        Path to load a pre-trained model checkpoint (optional)
  --save_checkpoint SAVE_CHECKPOINT
                        Path to save the trained model checkpoint (optional)
  --n N                 Value of n for n-gram models (Only for FFNN) (optional)
  --epochs EPOCHS       No of epochs to run for. (this includes the epochs of already trained model)
  --benchmark_file BENCHMARK_FILE
                        Benchmarking runs only if this is provided
```

Example:

To run generator

```sh
python generator.py f ./corpus/Pride\ and\ Prejudice\ -\ Jane\ Austen.txt 5 --n 5 --load_checkpoint ./checkpoints/f_p_5n_20ep.pt
```

To run benchmarking:
```sh
python generator.py f ./corpus/Pride\ and\ Prejudice\ -\ Jane\ Austen.txt 5 --n 5 --load_checkpoint ./checkpoints/f_p_5n_20ep.pt --benchmark_file ./<some name>
```

Files will be created as `<some-name>-train.txt` and `<some-name>-test.txt`

## All checkpoints are available at [https://github.com/Abhinavreddy-B/INLP-A2/releases/tag/checkpoints](https://github.com/Abhinavreddy-B/INLP-A2/releases/tag/checkpoints)

## Note:

Also if possible use:

```sh
ulimit -v 4000000 && python .....
```

to avoid memory hogging if any of the models use to much memory

## Note:

Due to memory constrains:
- used hidden layer size 128 for `Pride and Prejudice` corpus. 
- used hidden layer size 64 for `Ulysses` corpus.

So make the changes accordingly in `HIDDEN_DIM` variable in the code before running the model.