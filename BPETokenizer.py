import json
import os

class BPEEncoder:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.token2id = {}
        self.id2token = {}

    def train(self, corpus_iter):
        print("[BPE] Training started...")
        from collections import Counter
        pair_counts = Counter()
        words = []

        for line in corpus_iter():
            line = line.strip()
            word = tuple(line)
            words.append(word)
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i+1])] += 1

        vocab = set(ch for word in words for ch in word)

        while len(vocab) < self.vocab_size and pair_counts:
            most_common = pair_counts.most_common(1)[0][0]
            new_token = ''.join(most_common)
            vocab.add(new_token)

            new_words = []
            for word in words:
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == most_common:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(tuple(new_word))
            words = new_words

            pair_counts.clear()
            for word in words:
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i+1])] += 1

        self.vocab = vocab
        self.token2id = {tok: i for i, tok in enumerate(sorted(vocab))}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
        print(f"[BPE] Training finished. Vocab size: {len(self.vocab)}")

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for l in range(min(10, len(text)-i), 0, -1):
                sub = text[i:i+l]
                if sub in self.token2id:
                    tokens.append(self.token2id[sub])
                    i += l
                    matched = True
                    break
            if not matched:
                tokens.append(-1)
                i += 1
        return tokens

    def decode(self, ids):
        return ''.join([self.id2token[i] for i in ids if i in self.id2token])

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "bpe_vocab.json"), "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "vocab": list(self.vocab),
                "token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()}
            }, f, ensure_ascii=False, indent=2)
        print(f"[BPE] Tokenizer saved to {path}")

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "bpe_vocab.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(vocab_size=data["vocab_size"])
        tokenizer.vocab = set(data["vocab"])
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {int(k): v for k, v in data["id2token"].items()}
        return tokenizer

# 保存模型
# Recreate and train BPE tokenizer
bpe_tokenizer = BPEEncoder(vocab_size=100)

# Example corpus generator
def corpus_generator():
    for line in [
        "ACGTACGTGATTACAGGCT",
        "TATAAGCTAGACGT",
        "GATTACAGATTACAGATTACA",
    ]:
        yield line

bpe_tokenizer.train(corpus_generator)
bpe_tokenizer.save("bpe_tokenizer")
tokenizer_loaded = bpe_tokenizer.load("./bpe_tokenizer")
test = "ACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCT"
ids = tokenizer_loaded.encode(test)
print("Encoded:", ids)
print("Decoded:", tokenizer_loaded.decode(ids))
