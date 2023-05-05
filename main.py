import torch
import tiktoken
import os
import GPT_model as g
import hyperparameters as h

ROOT_DIR = os.path.dirname(__file__)
enc = tiktoken.get_encoding("cl100k_base")

# Load the model
m = g.GPTLanguageModel()
m.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'trained models', 'weights-28.pt')))
m.to(h.device)
m.eval()

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Prompt user for input
user_input = input("Input: ")

# Main loop
while(user_input != "exit"):
    # Encode the input into tokens
    encoded_input = enc.encode(user_input)
    print(f"Encoded input: {encoded_input}")

    # Put tokens in tensor
    user_context = torch.tensor([enc.encode(user_input)], dtype=torch.long, device=h.device)

    # Generate
    print("Completion: ", enc.decode(m.generate(user_context, 32)[0].tolist()))

    print()
    # Prompt user for input
    user_input = input("Input (type \"exit\" to quit): ")