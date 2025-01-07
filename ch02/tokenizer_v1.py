import re

from get_the_verdict import vocab


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocesses = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocesses = [item.strip() for item in preprocesses if item.strip()]
        ids = [self.str_to_int[token] for token in preprocesses]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab=vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride. """
ids = tokenizer.encode(text)
print(ids)
# turn these token IDs back into text using the decode method
print(tokenizer.decode(ids))
