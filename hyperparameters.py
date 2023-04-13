import torch

# Hyperparameters
batch_size = 32                                             # Number of block_size length blocks to train on for each line in a file
block_size = 32                                             # Number of tokens to use as context when generating
num_batches = 1000                                          # Number of batches to take from each training file
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Use GPU if available
est_iters = 200                                             # Number of iterations to average for estimating loss
min_line_length = 256                                       # Minimum number of characters for the line to be included in the dataset
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2