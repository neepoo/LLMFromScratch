import re

from ch02.get_the_verdict import vocab


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocesses = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocesses = [item.strip() for item in preprocesses if item.strip()]
        ids = [self.str_to_int.get(token, self.str_to_int["<|unk|>"]) for token in preprocesses]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str.get(i) for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


if __name__ == "__main__":
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)

    tokenizer = SimpleTokenizerV2(vocab=vocab)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

