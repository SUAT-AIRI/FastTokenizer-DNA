from collections import Counter
import json
import os

class UnigramTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.token2id = {}
        self.id2token = {}

    def train(self, corpus_iter):
        print("[Unigram] Training started...")
        # Step 1: collect all substrings with frequency
        freq = Counter()
        for line in corpus_iter():
            line = line.strip()
            for i in range(len(line)):
                for j in range(i + 1, min(len(line), i + 10) + 1):
                    substr = line[i:j]
                    freq[substr] += 1

        # Step 2: keep top substrings by frequency
        most_common = freq.most_common(self.vocab_size)
        self.vocab = set(sub for sub, _ in most_common)

        self.token2id = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        print(f"[Unigram] Training finished. Vocab size: {len(self.vocab)}")

    def encode(self, text):
        i = 0
        tokens = []
        while i < len(text):
            matched = False
            for j in range(min(10, len(text) - i), 0, -1):
                sub = text[i:i + j]
                if sub in self.vocab:
                    tokens.append(self.token2id[sub])
                    i += j
                    matched = True
                    break
            if not matched:
                i += 1  # skip unknown
        return tokens

    def decode(self, ids):
        return ''.join([self.id2token[i] for i in ids if i in self.id2token])

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "unigram_vocab.json"), "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "vocab": list(self.vocab),
                "token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()}
            }, f, indent=2)
        print(f"[Unigram] Tokenizer saved to {path}")

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "unigram_vocab.json"), "r") as f:
            data = json.load(f)
        tok = cls(data["vocab_size"])
        tok.vocab = set(data["vocab"])
        tok.token2id = data["token2id"]
        tok.id2token = {int(k): v for k, v in data["id2token"].items()}
        return tok

# Example corpus generator
def corpus_generator():
    for line in [
        "ACGTACGTGATTACAGGCT",
        "TATAAGCTAGACGT",
        "GATTACAGATTACAGATTACA",
    ]:
        yield line
if __name__ == '__main__':
    # Train and save unigram tokenizer
    unigram_tokenizer = UnigramTokenizer(vocab_size=100)
    unigram_tokenizer.train(corpus_generator)
    unigram_tokenizer.save("unigram_tokenizer")
    tokenizer_loaded = unigram_tokenizer.load("./unigram_tokenizer")
    test = "ACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCT"
    ids = tokenizer_loaded.encode(test)
    print("Encoded:", ids)
    print("Decoded:", tokenizer_loaded.decode(ids))

