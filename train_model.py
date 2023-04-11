import torch
import tiktoken
import os
import time
import GPT_model as g
import hyperparameters as h

ROOT_DIR = os.path.dirname(__file__)

start_time = time.time()
#print("%s seconds" % (time.time() - start_time))

# Read Wikipedia dataset
with open(os.path.join(ROOT_DIR, 'data', 'wikisent2.txt'), 'r') as file:
    dataset = file.read().splitlines()

# Read pre-encoded dataset
with open(os.path.join(ROOT_DIR, 'data', 'pre-encoded-data.txt'), 'r') as file:
    encoded_data = file.read().splitlines()


# Tiktoken encoding
enc = tiktoken.get_encoding("cl100k_base")

# Train-Test split
# data = torch.tensor(enc.encode(dataset), dtype=torch.long)
# encoded_data = enc.encode_batch(dataset)
data = torch.tensor(encoded_data, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - h.block_size, (h.batch_size,))
    x = torch.stack([data[i:i+h.block_size] for i in ix])
    y = torch.stack([data[i+1:i+h.block_size+1] for i in ix])
    x, y = x.to(h.device), y.to(h.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(h.eval_iters)
        for k in range(h.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = g.GPTLanguageModel()

m = model.to(h.device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=h.learning_rate)

for iter in range(h.max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % h.eval_interval == 0 or iter == h.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the weights
torch.save(m.state_dict(), os.path.join(ROOT_DIR, 'trained models', 'weights.pt'))
torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'trained models', 'weights.pt'))

# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save(os.path.join(ROOT_DIR, 'trained models', 'model.pt')) # Save

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=h.device)
print(enc.decode(m.generate(context, max_new_tokens=50)[0].tolist()))