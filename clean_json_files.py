import json
import os
import tiktoken
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
data_path = 'data/enwiki20201020'

directory = os.fsencode(os.path.join(ROOT_DIR, data_path))

count = 1
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    outfilename = "cleaned-file-" + str(count) + ".txt"

    # Parse the data
    if(filename.endswith(".json")):
        jsonfile = open(os.path.join(ROOT_DIR, data_path, filename), 'r', encoding='utf-8')
        outfile = open(os.path.join(ROOT_DIR, data_path + '-cleaned', outfilename), 'w', encoding='utf-8')

        data = json.load(jsonfile)
        
        # Write the data
        for entry in data:
            line = entry['text']

            # TODO - remove non-natural language content
                # TODO - remove content in between ==
                # TODO - remove context in between curly brackets
            
            outfile.write(line)
            outfile.write("\n")

        jsonfile.close()
        outfile.close()
    count = count + 1