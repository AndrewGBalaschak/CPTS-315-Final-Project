import torch
import tiktoken
import os
import GPT_model as g
import hyperparameters as h

ROOT_DIR = os.path.dirname(__file__)
enc = tiktoken.get_encoding("cl100k_base")

# m = torch.jit.load(os.path.join(ROOT_DIR, 'trained models', 'model.pt'))

# Load the model
m = g.GPTLanguageModel()
m.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'trained models', 'weights.pt')))
m.to(h.device)
m.eval()

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Test model generation
context = torch.zeros((1, 1), dtype=torch.long, device=h.device)
print(enc.decode(m.generate(context, max_new_tokens=50)[0].tolist()))

# Prompt user for input
user_input = input("Input: ")

# Main loop
while(user_input != "exit"):
    # Encode the input into tokens
    encoded_input = enc.encode(user_input)
    
    print(f"Encoded input: {encoded_input}")

    # Generate from the model
    #context = torch.zeros((1, 1), dtype=torch.long, device=h.device)
    user_context = torch.tensor(encoded_input, dtype=torch.long, device=h.device).unsqueeze(0)
    #context = torch.cat((context, user_context), dim=1)

    print(enc.decode(m.generate(user_context, max_new_tokens=50)[0].tolist()))

    # Prompt user for input
    user_input = input("Input: ")