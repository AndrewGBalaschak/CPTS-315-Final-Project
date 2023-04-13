import tiktoken
import os
import time
import math
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
batch_size = 10000

# Read Wikipedia dataset
with open(os.path.join(ROOT_DIR, 'data', 'wikisent2.txt'), 'r') as file:
    dataset = file.read().splitlines()

# Tiktoken encoding
enc = tiktoken.get_encoding("cl100k_base")

# Batch encode the data - this uses too much RAM for large datasets
# encoded_data = enc.encode_batch(dataset)

# Open output file
output = open(os.path.join(ROOT_DIR, 'data', 'pre-encoded-data.txt'), 'w')

# Encode the data in chunks of batch_size in order to save RAM
i = 0
batch_counter = 1
total_batches = math.ceil(len(dataset) / batch_size)
start_time = time.time()

while(i < len(dataset)):
    batch_start_time = time.time()

    # Encode batch of batch_size entries
    encoded_batch = enc.encode_batch(dataset[i:i+batch_size])

    # Write to output file
    for data in encoded_batch:
        output.write(str(data[0]))

        for token in data[1:]:
            output.write(",")
            output.write(str(token))
        output.write("\n")

    print("Batch {}/{} time: {}\t total time elapsed: {}".format(batch_counter, total_batches, time.time() - batch_start_time, time.time() - start_time))
    i = i + batch_size
    batch_counter = batch_counter + 1

output.close()