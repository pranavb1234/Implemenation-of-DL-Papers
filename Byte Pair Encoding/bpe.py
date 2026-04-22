import collections

def bpe_train(text, vocab_size):
    # start with individual bytes as tokens
    ranks = {}
    for i in range(256):
        ranks[bytes([i])] = i

    # split text into list of byte-lists
    words = text.split()
    all_parts = []
    for word in words:
        parts = [bytes([b]) for b in word.encode("utf-8")]
        all_parts.append(parts)

    # Keep merging until we reach vocab_size
    while len(ranks) < vocab_size:

        # Count all pairs
        stats = collections.Counter()
        for parts in all_parts:
            for pair in zip(parts[:-1], parts[1:]):
                stats[pair] += 1

        if not stats:
            break  

        # Find most common pair
        best_pair = max(stats, key=lambda x: stats[x])
        
        # Create new token
        new_token = best_pair[0] + best_pair[1]
        ranks[new_token] = len(ranks)  # give it the next ID
        
        print(f"Merged {best_pair[0]} + {best_pair[1]} → {new_token}  (token ID: {ranks[new_token]})")

        # Merge this pair in ALL words
        new_all_parts = []
        for parts in all_parts:
            new_all_parts.append(merge_pair(parts, best_pair))
        all_parts = new_all_parts

    return ranks