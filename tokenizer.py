import json
import os

from transformers import BertTokenizer


class CharTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]

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

    @property
    def mask_token_id(self):
        return self.char_to_idx["[MASK]"]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.char_to_idx, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "r") as f:
                self.char_to_idx = json.load(f)
            self.idx_to_char = {int(i): c for c, i in self.char_to_idx.items()}


class BertTokenizerWrapper:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def build_vocab(self, texts):
        pass

    def encode(self, text, max_len=512):
        encoded = self.tokenizer.encode(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True
        )
        return encoded

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    def save(self, path):
        pass

    def load(self, path):
        pass


def make_tokenizer(cfg):
    tokenizer_type = cfg.get("training", {}).get("tokenizer", "char")
    if tokenizer_type == "bert":
        return BertTokenizerWrapper()
    return CharTokenizer()

