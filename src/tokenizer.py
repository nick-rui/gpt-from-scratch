class CharacterTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.character_to_token = {ch:t for t, ch in enumerate(self.vocab)}
        self.token_to_character = {t:ch for t, ch in enumerate(self.vocab)}

    def encode(self, string):
        return [self.character_to_token[ch] for ch in string]


    def decode(self, tokens):
        return ''.join([self.token_to_character[t] for t in tokens])


if __name__ == "__main__":
    vocab = set(list("the quick brown fox jumps over the lazy dog"))
    tokenizer = CharacterTokenizer(vocab)

    string = "hello my name is nick"
    print(f"string = {string}")
    print(f"tokens = {tokenizer.encode(string)}")
    print(f"decoded tokens = {tokenizer.decode(tokenizer.encode(string))}")
