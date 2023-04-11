import os
import random

ROOT_DIR = os.path.dirname(__file__)
probability = 0.05

with open(os.path.join(ROOT_DIR, 'data', 'wikisent2.txt'), 'r') as file:
    dataset = file.read().splitlines()

output = open(os.path.join(ROOT_DIR, 'data', 'wikisent2-small.txt'), 'w')

for line in dataset:
    if(random.random() <= probability):
        output.write(line)
        output.write("\n")

output.close()