# CPTS-315-Final-Project

## Project Goal
To develop a conversational language model using PyTorch and the transformer architecture that inspires users to learn more about the world around them.

## Project Implementation
For this project I am using PyTorch and the transformer architecture detailed in the paper "Attention Is All You Need" by Vaswani et al.

This model is to be trained on a ~26GB text dump from Wikipedia that has been formatted and cleaned by David Shapiro.

The Wikipedia data is tokenized using OpenAI's Tiktoken library.

## Project Limitations
Here is a list of limitations of this model in order of easiest-to-address to hardest-to-address

- Context Size: This language model by default uses a context size of 16, as set in `hyperparameters.py`, meaning that only the past 16 tokens are considered when the model is generating text. For context, the last-generation GPT-3 model uses 2048 tokens for context, meaning my model uses 0.8% the maximum context of GPT-3. While this is not important for shorter queries and responses, it is integral to have a large context size for longer, more coherent responses.

- Dataset: This language model is trained on a ~26GB dump from Wikipedia. While this is a lot of text, it is very small compared to the amount of text that the last-generation GPT-3 model was trained on. GPT-3 was also trained on a dataset from Wikipedia, but this dataset makes up only 0.6% of the tokens that GPT-3 was trained on. To even come close to the amount of data GPT-3 was trained on, much more storage space would be required.

- Parameters: This language model uses 87.76 million parameters, which might seem like a lot, but for reference the last-generation GPT-3 model uses 175 billion parameters, meaning my model uses 0.05% (about one half of one half of one half of one half of one percent!) the parameters of GPT-3. As such, its output can not be expected to be on par with GPT-3.

- Training Hardware: This language model was trained on high-end consumer hardware, which presents limitations in the amount of RAM and VRAM available for the model. Additionally, this limits the size of the model through limitations on available storage

- Training time: This language model was trained on my personal computer, which prevents me from being able to use my machine for other purposes while training is ongoing. As such, this limits the amount of training time that is possible.

## Running The Model
To run this model for yourself, all you need to do is run `main.py`

## Training The Model
To train the model yourself, you will need to download the Wikipedia dataset from the Kaggle link in the Citations section.

Once you have downloaded and extracted the data to the `data` directory in the project's directory, you will need to run `clean_json_files.py` to extract the raw text from the json files, this will export the data to a new folder suffixed with `-cleaned`.

After this, the data will have to be tokenized, which is done with the `encode_data.py` file. Similarly, these tokenized files will be exported to a new folder suffixed with `-tokenized`.

Once this data preparation has been completed, you may now run `train_model.py`, which trains a transformer model on the data in the `-tokenized` directory. The `-cleaned` directory may be deleted now, if you wish.

Upon completion, the model will print a test generation to the console, and the weights will be exported as `weights.pt` under the `trained models` directory.
These weights can now be loaded in `main.py`

If you want to adjust the model's hyperparameters, they are accessible in `hyperparameters.py` and synchronize to all relevant files in this repo.

## Future Work
Training the model on the WET files from Common Crawl would be a very interesting expansion of this project, as the informal language used on sites other than Wikipedia may help the generated responses to be more conversational.

## Citations
- David Shapiro's Plain Text Wikipedia Dataset: https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011
  - David Shapiro's Plain Text Wikipedia Repo: https://github.com/daveshap/PlainTextWikipedia
- Andrej Karpathy's PyTorch Transformer Implementation Tutorial: https://www.youtube.com/watch?v=kCc8FmEb1nY
- Information on GPT-3: https://en.wikipedia.org/wiki/GPT-3