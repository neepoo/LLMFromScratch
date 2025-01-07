import os
import re
import urllib.request

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt"
       )

file_path = "the-verdict.txt"
# if the file is not present, download it, otherwise, load it
if not os.path.exists(file_path):
    urllib.request.urlretrieve(url, file_path)

# load the the-verdict.txt file
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# print("Total number of characters in the file:", len(raw_text))
# print(raw_text[:99])

# tokenizer the text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))

# converting tokens into token IDs
#  let’s convert these tokens from a Python string to an integer representation to
# produce the token IDs. This conversion is an intermediate step before converting the
# token IDs into embedding vectors.

# create a vocabulary token <-> number
all_words = sorted(set(preprocessed))
# V2 tokenizer add special token "unk" and "endoftext"
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>", ])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
# vocab = {token:integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break
