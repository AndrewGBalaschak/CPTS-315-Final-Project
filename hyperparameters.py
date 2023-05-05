import torch

# GPT Hyperparameters
batch_size = 16                                             # Number of block_size length blocks to train on for each line in a file
block_size = 128                                            # Number of tokens to use as context, affects VRAM usage, training time
num_batches = 1024                                          # Number of batches to sample from each training file, serves as a way to limit training time
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Use GPU if available
n_embd = 512                                                # Number of embedding features
n_head = 8                                                  # Number of self-attention heads, affects training time
n_layer = 12                                                # Number of layers in the network, affects training time
dropout = 0.1
learning_rate = 0.001

est_iters = 400                                             # Number of iterations to average over for estimating loss

# Data Cleanup Hyperparameters
min_line_length = 256                                       # Minimum number of characters for the line to be included in the dataset