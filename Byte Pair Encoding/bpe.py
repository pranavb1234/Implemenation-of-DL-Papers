# BPE Implemented from Scratch 

import collections

class BytePairTokenizer:
    def __init__(self, ranks):
        """
        ranks = dictionary of {token_bytes: token_id}
        This is learned during training
        """
        self.ranks = ranks
        self.decoder = {id: token for token, id in ranks.items()}

    # ─────────────────────────────────────────
    # TRAINING  (class method, returns new object)
    # ─────────────────────────────────────────
    @classmethod
    def train(cls, text, vocab_size):
        """Train a tokenizer on text and return a BytePairTokenizer"""

        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256!")

        
        ranks = {}
        for i in range(256):
            ranks[bytes([i])] = i

        
        words = text.split()
        all_parts = []
        for word in words:
            parts = [bytes([b]) for b in word.encode("utf-8")]
            all_parts.append(parts)

        
        while len(ranks) < vocab_size:
            # Count pairs
            stats = collections.Counter()
            for parts in all_parts:
                for pair in zip(parts[:-1], parts[1:]):
                    stats[pair] += 1

            if not stats:
                break

            
            best_pair = max(stats, key=lambda x: stats[x])
            new_token = best_pair[0] + best_pair[1]
            ranks[new_token] = len(ranks)

            print(f"Merge {len(ranks)-256}: {best_pair[0]} + {best_pair[1]} → {new_token}")

            
            all_parts = [
                cls._merge_pair(parts, best_pair)
                for parts in all_parts
            ]

        return cls(ranks) 

    # ─────────────────────────────────────────
    # ENCODING
    # ─────────────────────────────────────────
    def encode(self, text):
        """Convert text → list of token IDs"""
        parts = [bytes([b]) for b in text.encode("utf-8")]

        while True:
            best_idx = None
            best_rank = None

            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = self.ranks.get(pair[0] + pair[1])
                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i

            if best_rank is None:
                break

            pair_to_merge = (parts[best_idx], parts[best_idx + 1])
            parts = self._merge_pair(parts, pair_to_merge)

        return [self.ranks[part] for part in parts]

    # ─────────────────────────────────────────
    # DECODING
    # ─────────────────────────────────────────
    def decode(self, token_ids):
        """Convert token IDs → text"""
        byte_string = b"".join(self.decoder[id] for id in token_ids)
        return byte_string.decode("utf-8", errors="replace")

    # ─────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────
    @staticmethod
    def _merge_pair(parts, pair):
        """Merge all occurrences of pair in parts"""
        new_parts = []
        i = 0
        while i < len(parts) - 1:
            if (parts[i], parts[i+1]) == pair:
                new_parts.append(parts[i] + parts[i+1])
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1
        if i == len(parts) - 1:
            new_parts.append(parts[i])
        return new_parts

    def vocab_size(self):
        return len(self.ranks)

    def __repr__(self):
        return f"BytePairTokenizer(vocab_size={self.vocab_size()})"
        
        
# RESULTS --         

tokenizer = BytePairTokenizer.train(
    text="hello world hello hello world python is great",
    vocab_size=270
)

print("\n", tokenizer)  # BytePairTokenizer(vocab_size=270)

ids = tokenizer.encode("hello world")
print("Encoded:", ids)

text = tokenizer.decode(ids)
print("Decoded:", text)

assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
assert tokenizer.decode(tokenizer.encode("world")) == "world"
print("All checks passed!")