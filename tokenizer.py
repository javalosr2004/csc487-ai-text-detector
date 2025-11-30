import json
import os


class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]"]

    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)

        self.char_to_idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        for i, char in enumerate(sorted(chars)):
            self.char_to_idx[char] = len(self.special_tokens) + i

        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

    def encode(self, text, max_len=512):
        tokens = [self.char_to_idx["[CLS]"]]
        for char in text[:max_len - 1]:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx["[UNK]"]))

        # Pad to max_len
        while len(tokens) < max_len:
            tokens.append(self.char_to_idx["[PAD]"])

        return tokens[:max_len]

    def decode(self, token_ids):
        chars = []
        for idx in token_ids:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char not in self.special_tokens:
                    chars.append(char)
        return "".join(chars)

    @property
    def vocab_size(self):
        return len(self.char_to_idx)

    @property
    def pad_token_id(self):
        return self.char_to_idx["[PAD]"]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.char_to_idx, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.char_to_idx = json.load(f)
            self.idx_to_char = {int(i): c for c, i in self.char_to_idx.items()}

