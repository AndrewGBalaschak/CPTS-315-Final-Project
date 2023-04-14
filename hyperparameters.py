import torch

# GPT Hyperparameters
batch_size = 32                                             # Number of block_size length blocks to train on for each line in a file
block_size = 64                                             # Number of tokens to use as context, affects VRAM usage, training time
num_batches = 1000                                          # Number of batches to sample from each training file, serves as a way to limit training time
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Use GPU if available
n_embd = 384
n_head = 8                                                  # Number of self-attention heads, affects training time
n_layer = 16                                                # Number of layers in the network, affects training time
dropout = 0.2
learning_rate = 0.0001
est_iters = 100                                             # Number of iterations to average over for estimating loss

# Data Cleanup Hyperparameters
min_line_length = 256                                       # Minimum number of characters for the line to be included in the dataset