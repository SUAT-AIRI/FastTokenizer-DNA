# custom_tokenizer/tokenizer.py

from collections import defaultdict, Counter
import re
import json
import os

class FastTokenizer:
    def __init__(self, long_tokens=None, vocab_size=1000):
        self.long_tokens = sorted(long_tokens or [], key=len, reverse=True)  # 长串优先匹配
        self.vocab_size = vocab_size
        self.vocab = set(self.long_tokens)
        self.vocab.update(list(set(char for token in self.long_tokens for char in token)))  # 保证字符也在
        self.token2id = {}
        self.id2token = {}

    def train(self, corpus_iter):
        print("[INFO] Starting tokenizer training...")
        pair_counts = Counter()
        visited_tokens = set()

        # 第一次统计
        for line in corpus_iter():
            cleaned_line = self.replace_long_tokens(line.strip())
            pair_counts.update(self.get_pairs_from_line(cleaned_line))

        # 开始迭代合并
        iters = 0
        max_iters = 5000
        while len(self.vocab) < self.vocab_size and pair_counts and iters < max_iters:
            best_pair, _ = pair_counts.most_common(1)[0]
            new_token = ''.join(best_pair)
            if new_token in self.vocab or new_token in visited_tokens:
                del pair_counts[best_pair]
                continue
            self.vocab.add(new_token)
            visited_tokens.add(new_token)

            # 重新统计
            pair_counts.clear()
            for line in corpus_iter():
                cleaned_line = self.replace_long_tokens(line.strip())
                pair_counts.update(self.get_pairs_from_line(cleaned_line))

            iters += 1

        print(f"[INFO] Final vocab size: {len(self.vocab)}")
        self.build_token_dict()

    def replace_long_tokens(self, text):
        for token in self.long_tokens:
            text = text.replace(token, ' ')
        return text

    def get_pairs_from_line(self, line):
        pair_counts = Counter()
        for word in line.split():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += 1
        return pair_counts

    def build_token_dict(self):
        self.token2id = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def encode(self, text):
        # Step 1: 长串优先切分
        tokens = []
        remainder = text
        for token in self.long_tokens:
            parts = remainder.split(token)
            tmp = []
            for part in parts[:-1]:
                tmp.append(part)
                tmp.append(token)
            tmp.append(parts[-1])
            remainder = ''.join(tmp)

        # Step 2: 对剩余字符做 BPE
        output = []
        i = 0
        while i < len(remainder):
            matched = False
            for l in range(min(10, len(remainder) - i), 0, -1):  # 最多看10长度
                sub = remainder[i:i + l]
                if sub in self.vocab:
                    output.append(sub)
                    i += l
                    matched = True
                    break
            if not matched:
                output.append(remainder[i])  # fallback to char
                i += 1
        return [self.token2id.get(tok, -1) for tok in output if tok.strip()]

    def decode(self, ids):
        return ''.join([self.id2token[i] for i in ids if i in self.id2token])

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({
                "long_tokens": self.long_tokens,
                "vocab_size": self.vocab_size,
                "vocab": list(self.vocab),
                "token2id": self.token2id,
                "id2token": {str(k): v for k, v in self.id2token.items()}
            }, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Tokenizer saved to {path}")

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "vocab.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(long_tokens=data["long_tokens"], vocab_size=data["vocab_size"])
        tokenizer.vocab = set(data["vocab"])
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {int(k): v for k, v in data["id2token"].items()}
        return tokenizer

# 用法示例
if __name__ == '__main__':
    # 假设已有解析器给出长串词表
    long_tokens = ['ACGTACGT', 'GATTACA', 'TATA','GATT']

    def corpus_generator():
        for line in [
            "ACGTACGTGATTACAGGCT",
            "TATAAGCTAGACGT",
            "GATTACAGATTACAGATTACA",
        ]:
            yield line

    tokenizer = FastTokenizer(long_tokens=long_tokens, vocab_size=100)
    tokenizer.train(corpus_generator)

    tokenizer.save("./fast_tokenizer")
    tokenizer_loaded = FastTokenizer.load("./fast_tokenizer")

    test = "ACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCT"
    ids = tokenizer_loaded.encode(test)
    print("Encoded:", ids)
    print("Decoded:", tokenizer_loaded.decode(ids))
