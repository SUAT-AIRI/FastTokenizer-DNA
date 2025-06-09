from pathlib import Path
import json
from typing import Dict, List
import time
import pandas as pd

from Bytetokenizer import ByteTokenizer
# Load the three tokenizers
from FastTokenizer import FastTokenizer
from BPETokenizer import BPETokenizer
from UnigramTokenizer import UnigramTokenizer

# Paths to the saved tokenizers
fast_path = "fast_tokenizer"
bpe_path = "bpe_tokenizer"
unigram_path = "unigram_tokenizer"
byte_path = "byte_tokenizer"

# Load tokenizers
fast_tokenizer = FastTokenizer.load(fast_path)
bpe_tokenizer = BPETokenizer.load(bpe_path)
unigram_tokenizer = UnigramTokenizer.load(unigram_path)
byte_tokenizer = ByteTokenizer.load(byte_path)
# Test dataset
# test_set = [
#     "ACGTACGTGATTACAGGCT",
#     "TATAAGCTAGACGT",
#     "GATTACAGATTACAGATTACA",
#     "GGGAAACCCGGGTTTAAA",
#     "ACGTAGCTAGCTAGTTAGC",
#     "TATATATAGCGCGCGCTATA",
#     "GATTACATATAGATTACAGG",
#     "ACGTACGTACGTACGTACGT",
#     "TATACGTAGCTAGCATATA",
#     "GCGTACGTTAGCTAGCTGGA"
# ]
with open("test_dna_corpus.txt",encoding='utf-8',mode='r') as fr:
    test_set = [line.strip() for line in fr.readlines()]

# Evaluator class
class TokenizerEvaluator:
    def __init__(self, tokenizers: Dict[str, any], test_set: List[str]):
        self.tokenizers = tokenizers
        self.test_set = test_set

    def evaluate(self):
        results = []
        for name, tokenizer in self.tokenizers.items():
            total_chars, total_tokens, vocab_hits, accurate_decodes = 0, 0, 0, 0
            for text in self.test_set:
                token_ids = tokenizer.encode(text)
                decoded = tokenizer.decode(token_ids)

                total_chars += len(text)
                total_tokens += len(token_ids)
                vocab_hits += sum(1 for tid in token_ids if tid >= 0)
                accurate_decodes += int(decoded == text)

            coverage = vocab_hits / total_tokens if total_tokens else 0
            compression = total_chars / total_tokens if total_tokens else 0
            accuracy = accurate_decodes / len(self.test_set)
            results.append({
                "Tokenizer": name,
                "CompressionRate": round(compression, 3),
                "Coverage": round(coverage, 3),
                "Accuracy": round(accuracy, 3),
                "AvgTokenPerSeq": round(total_tokens / len(self.test_set), 2)
            })
        return pd.DataFrame(results)

if __name__ == '__main__':
    # Run the evaluation
    evaluator = TokenizerEvaluator({
        "Fast": fast_tokenizer,
        "BPE": bpe_tokenizer,
        "Unigram": unigram_tokenizer,
        "Byte": byte_tokenizer
    }, test_set)
    df_results = evaluator.evaluate()
    print(df_results)
