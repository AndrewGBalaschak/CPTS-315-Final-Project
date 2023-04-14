import json
import os
import re
import hyperparameters as h
from tqdm import tqdm

ROOT_DIR = os.path.dirname(__file__)
data_path = 'data/enwiki20201020'
directory = os.fsencode(os.path.join(ROOT_DIR, data_path))

# These are deprecated, super duper slow since they aren't compiled!
'''
# Removes text between start_char and end_char, inclusive
def remove_text_between(text, start_char, end_char):
    while start_char in text:
        text = re.subn(f'[{start_char}][^{start_char}{end_char}]*[{end_char}]', '', text)[0]
    return text

# Removes text between start_str and end_str, inclusive
def remove_text_between_str(text, start_str, end_str):
    while start_str in text:
        text = re.subn(f'{re.escape(start_str)}[^{re.escape(start_str)}{re.escape(end_str)}]*{re.escape(end_str)}', '', text)[0]
    return text

filter_pairs = [['(',')'], ['{','}'], ['==','==']]
'''

# Compiled regex for text cleanup
regex_parentheses = re.compile(r'[(][^()]*[)]')         # Remove text between parentheses
regex_braces = re.compile(r'\{[^{}]*\}')                # Remove text between curly braces
regex_equals = re.compile(r'==[^=]*==')                 # Remove text between ==
regex_english = re.compile(r'[^\x00-\x7F]+')            # Remove non-ASCII characaters
regex_punctuation = re.compile(r'\s+([,.!?])')          # Remove any leftover spaces before punctuation
regex_double_spaces = re.compile(r' +')                 # Remove any duplicate spaces

# Remove non-natural language content using compiled regex
def clean_string(input_string):
    output = input_string
    # Remove text in parenthesis, brackets, etc
    output = regex_parentheses.sub('', output)
    output = regex_braces.sub('', output)
    output = regex_equals.sub('', output)
    output = regex_english.sub('', output)
    output = output.replace('*', '')
    output = regex_punctuation.sub('', output)
    
    # Remove duplicate spaces that may have occured from previous step
    output = regex_double_spaces.sub(' ', output)

    # Try to remove anything after "Category:", usually this is at the end
    try:
        output = output[:output.index("Category:")]
    except:
        pass
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
            line = clean_string(line)

            # Line has to be above certain length after filtering to be valuable
            if len(line) > h.min_line_length:
                outfile.write(line)
                outfile.write("\n")

        jsonfile.close()
        outfile.close()
    count = count + 1