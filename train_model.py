import torch
import tiktoken
import os
import random
import GPT_model as g
import hyperparameters as h
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
data_path = 'data/enwiki20201020-tokenized'
enc = tiktoken.get_encoding("cl100k_base")

# Open pre-encoded dataset directory
directory = os.fsencode(os.path.join(ROOT_DIR, data_path))

# Tiktoken encoding
enc = tiktoken.get_encoding("cl100k_base")

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - h.block_size, (h.batch_size,))
    x = torch.stack([data[i:i+h.block_size] for i in ix])
    y = torch.stack([data[i+1:i+h.block_size+1] for i in ix])
    x, y = x.to(h.device), y.to(h.device)
    return x, y

def my_get_batch(split):
    data = train_data if split == 'train' else test_data
    row = data[random.randint(0, len(data)-1)]                  # Pick random row

    x_ary = []
    y_ary = []

    # Grab up to the max of batch_size chunks of inputs and targets
    for i in range(h.batch_size):
        x_temp = []
        y_temp = []

        random_offset = random.randint(0, max(len(row) - h.block_size, 0))
        #random_offset = 0

        # Grab a block_size chunk of inputs and targets
        for j in range(h.block_size):
            if((j + random_offset) > len(row) - 2):
                break
            x_temp.append(row[(j + random_offset)])
            y_temp.append(row[(j + random_offset + 1)])


        # Pad the end of the chunk if we happen to get a row that is too short to grab block_size
        while(len(x_temp) < h.block_size):
            x_temp.append(0)
        while(len(y_temp) < h.block_size):
            y_temp.append(0)

        # Add the array of inputs and targets to the matrix
        x_ary.append(torch.tensor(x_temp))
        y_ary.append(torch.tensor(y_temp))
        
    # Pad the end of the input if we happen to get a row that is too short to grab batch_size number of chunks from it
    while(len(x_ary) < h.batch_size):
        x_ary.append(torch.zeros(len(x_ary[0])))
    while(len(y_ary) < h.batch_size):
        x_ary.append(torch.zeros(len(x_ary[0])))

    # Convert matrix into a tensor, send to GPU and return
    x = torch.stack(x_ary)
    y = torch.stack(y_ary)
    x, y = x.to(h.device), y.to(h.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(h.est_iters)
        for k in range(h.est_iters):
            X, Y = my_get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Create the model and load it onto the GPU
model = g.GPTLanguageModel()
m = model.to(h.device)

# Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=h.learning_rate)

count = 1
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    data_file = open(os.path.join(ROOT_DIR, data_path, filename), 'r', encoding='utf-8').read().splitlines()

    # Read the data into a python array
    data_array = []
    for line in data_file:
        # Add new row to file
        data_array.append([])
        for i in line.split(','):
            data_array[-1].append(int(i))

    for iter in tqdm(range(h.num_batches)):
        #data_array = random.shuffle(data_array)
        
        # Train-Test split
        #data_tensor = torch.tensor(data_array, dtype=torch.long)
        n = int(0.9 * len(data_array))
        train_data = data_array[:n]
        test_data = data_array[n:]

        # sample a batch of data
        # xb, yb = get_batch('train')
        xb, yb = my_get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Evaluate loss after training on a given file has finished
    losses = estimate_loss()
    print(f"\nstep {count}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Save the weights
torch.save(m.state_dict(), os.path.join(ROOT_DIR, 'trained models', 'weights.pt'))
torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'trained models', 'weights2.pt'))

# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save(os.path.join(ROOT_DIR, 'trained models', 'model.pt')) # Save

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=h.device)
print(enc.decode(m.generate(context, max_new_tokens=50)[0].tolist()))