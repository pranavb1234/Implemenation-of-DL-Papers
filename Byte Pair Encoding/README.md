# Byte Pair Encoding (BPE) Tokenizer

## Results

```
Merge 1: b'h' + b'e' → b'he'
Merge 2: b'he' + b'l' → b'hel'
Merge 3: b'hel' + b'l' → b'hell'
Merge 4: b'hell' + b'o' → b'hello'
Merge 5: b'w' + b'o' → b'wo'
Merge 6: b'wo' + b'r' → b'wor'
Merge 7: b'wor' + b'l' → b'worl'
Merge 8: b'worl' + b'd' → b'world'
Merge 9: b'p' + b'y' → b'py'
Merge 10: b'py' + b't' → b'pyt'
Merge 11: b'pyt' + b'h' → b'pyth'
Merge 12: b'pyth' + b'o' → b'pytho'
Merge 13: b'pytho' + b'n' → b'python'
Merge 14: b'i' + b's' → b'is'

BytePairTokenizer(vocab_size=270)
Encoded: [259, 32, 263]
Decoded: hello world
All checks passed! 
```

---

## What is BPE?

Byte Pair Encoding (BPE) is a tokenization algorithm used in modern AI models like GPT and Claude. It converts text into numbers (tokens) by finding and merging the most common pairs of bytes repeatedly.

```
"hello world"  →  [259, 32, 263]
```

---

## How It Works

### Step 1 — Start with bytes
Every character is broken into individual bytes:
```
"hello" → [b'h', b'e', b'l', b'l', b'o']
```

### Step 2 — Count pairs
Count how often each pair appears across all words in training data:
```
(b'h', b'e') → 1 time
(b'l', b'l') → 1 time
...
```

### Step 3 — Merge most common pair
Glue the most common pair into one new token:
```
[b'h', b'e', b'l', b'l', b'o']
→ [b'he', b'l', b'l', b'o']    ← merged h+e
```

### Step 4 — Repeat until vocab size is reached
Keep merging until you have the desired number of tokens.

---

## Project Structure

```
byte pair encoding/
│
├── bpe.py       
├── README.md          # This file
└── test.py            # Test script
```
