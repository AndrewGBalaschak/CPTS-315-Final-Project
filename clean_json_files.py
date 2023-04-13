import json
import os
import re
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
data_path = 'data/enwiki20201020'

directory = os.fsencode(os.path.join(ROOT_DIR, data_path))

def remove_brackets(input_string):
    output = ""
    in_brackets = 0
    in_curly = 0
    in_equals = 0

    for i in range(len(input_string)-1):
        # Detect start
        if input_string[i] == '[':
            in_brackets += 1
        elif input_string[i] == '{':
            in_curly += 1
        elif input_string[i] == '=' and input_string[i+1] == '=' and in_equals == 0:
            in_equals += 1
            i += 1
        
        # Detect end
        elif input_string[i] == ']' and in_brackets > 0:
            in_brackets -= 1
        elif input_string[i] == '}' and in_curly > 0:
            in_curly -= 1
        elif input_string[i] == '=' and input_string[i-1] == '=' and in_equals > 0:
            in_equals -= 1
        
        # Not in any restricted areas
        elif in_brackets == 0 and in_equals == 0:
            output += input_string[i]
    
    return output

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


            # This block removes non-natural language content
            # Remove text between brackets
            #line = remove_brackets(line)

            re.sub("[\(\[].*?[\)\]]", "", x)

            # Try to remove anything after "Category:"
            try:
                line = line[:line.index("Category:")]
            except:
                pass
            
            outfile.write(line)
            outfile.write("\n")

        jsonfile.close()
        outfile.close()
    count = count + 1