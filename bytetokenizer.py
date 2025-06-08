# byte_tokenizer.py

import os
import json

class ByteTokenizer:
    def __init__(self):
        self.token2id = {bytes([i]).decode("latin1"): i for i in range(256)}
        self.id2token = {i: bytes([i]).decode("latin1") for i in range(256)}

    def encode(self, text):
        byte_seq = text.encode("utf-8")
        return [b for b in byte_seq]

    def decode(self, ids):
        byte_seq = bytes(ids)
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "byte_tokenizer.json"), "w") as f:
            json.dump({"token2id": self.token2id, "id2token": {str(k): v for k, v in self.id2token.items()}}, f)
        print(f"[INFO] ByteTokenizer saved to {path}")

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "byte_tokenizer.json"), "r") as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {int(k): v for k, v in data["id2token"].items()}
        return tokenizer

# 用法示例
if __name__ == '__main__':
    tokenizer = ByteTokenizer()
    text = "ACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCTACGTACGTGATTACATATAAGCT"
    ids = tokenizer.encode(text)
    print("Encoded:", ids)
    print("Decoded:", tokenizer.decode(ids))
    tokenizer.save("./byte_tokenizer")
    tokenizer_loaded = ByteTokenizer.load("./byte_tokenizer")
    print("Reloaded Decode:", tokenizer_loaded.decode(ids))
