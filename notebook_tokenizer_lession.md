# Tokenizers: a beginner lesson and the current implementation

Project: Write a tokenizer and Transformer from scratch, then train a roughly 0.1B parameter language model.

Implementation snapshot: October 31, 2026 (first written September 30, 2026; updated after GPT-2-style pre-splitting, the fast trainer and encoder, and the FineWeb-Edu tokenizer).

Audience: a learner who knows basic Python and is beginning machine learning.

This is a standalone learning source. The explanations and worked examples below can be understood without opening the project files. File names identify the implementation being described. Examples with learned IDs refer to specific training corpora; those IDs are not universal.

## 1. What we have built so far

We have built a small, functioning **byte-level byte pair encoding tokenizer**, usually called byte-level BPE, using the Python standard library.

It can:

- Split text into GPT-2-style chunks (words, numbers, symbols, whitespace) before learning merges.
- Learn repeated byte sequences from training documents, quickly enough for hundreds of megabytes.
- Encode text into integer token IDs, on one CPU core or across all of them.
- Decode those IDs back into the original text.
- Handle unfamiliar words, emoji, and valid UTF-8 text from different languages.
- Preserve case, punctuation, spaces, tabs, and newlines in strings passed to its API.
- Save its learned rules and split pattern as JSON and reload them.
- Display the chunks and the byte sequence represented by each token.

The small examples in this lesson use a demonstration tokenizer and a tokenizer trained on a short robot-and-cat text file, so every step can be followed by hand. The project also has a full-size tokenizer: 32,768 IDs trained on 304 MB of FineWeb-Edu, which matches GPT-2's compression on held-out text (4.63 vs 4.64 bytes per token) with a smaller vocabulary. 23 tokenizer tests pass.

The tokenizer is one part of a larger project. The Transformer (about 124 million parameters, `MODEL_LESSON.md`) and the training pipeline (`TRAINING_LESSON.md`) now exist too; this lesson covers only the tokenizer.

## 2. Why a language model needs tokenization

Neural networks calculate with numbers. Tokenization provides a consistent way to represent text as a sequence of numbers.

The full pipeline is:

```text
Text
  → tokenizer
  → token IDs
  → model's embedding lookup
  → Transformer
  → scores for possible next tokens
  → choose a next token
  → repeat
  → tokenizer decodes the output IDs into text
```

The tokenizer is responsible for mapping between text and IDs. The language model learns which IDs are likely to follow which other IDs.

For example, our tiny tokenizer maps:

```text
Hello, little robot!
```

to:

```text
[72, 101, 278, 44, 271, 272, 33]
```

The tokenizer has not understood the sentence. It has represented the sentence using reusable pieces of bytes.

## 3. Terms you should understand

| Term | Meaning in this lesson |
|---|---|
| Token | One unit in the encoded sequence. In our tokenizer, an ordinary token represents one or more bytes. |
| Token ID | An integer identifying a token. |
| Vocabulary | The complete set of IDs and their meanings, including special IDs. |
| Vocabulary size | The number of distinct IDs available. |
| Corpus | The text collection used to train the tokenizer or the language model. |
| Document | One separate text item supplied to tokenizer training. |
| Byte | An 8-bit value, represented by an integer from 0 to 255. |
| UTF-8 | The encoding used to convert our text strings into bytes. |
| BPE | Byte pair encoding: repeatedly combining frequent adjacent units. |
| Merge | A learned rule replacing two adjacent token IDs with one new ID. |
| Merge rank | A rule's position in the learned merge list. Lower ranks were learned earlier. |
| Encoding | Converting text or bytes into token IDs using the saved rules. |
| Decoding | Converting token IDs back into bytes or text. |
| Special token | An ID used as a marker, such as beginning, end, or padding. |
| Round trip | Encoding a value and decoding it back, then checking for exact equality. |
| OOV | Out of vocabulary: a unit missing from a tokenizer's vocabulary. Our full byte base avoids this for ordinary input bytes. |
| Normalization | Changing text into a chosen standard form before tokenization. |
| Pre-tokenization | Splitting input into smaller regions before applying an algorithm such as BPE. |

A token is not necessarily a word or a character. It can represent punctuation, a space, part of a word, a whole word, multiple words, or part of a UTF-8 character. What is possible depends on the tokenizer design and training data.

## 4. Text, Unicode, UTF-8, and bytes

Python's `str` represents Unicode text, while `bytes` represents byte values. Our code explicitly uses `text.encode("utf-8")` and `raw.decode("utf-8")` to move between them. UTF-8 represents a Unicode code point using one to four bytes. See the [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html) for the encoding background.

Try this Python example:

```python
text = "banana"
raw = text.encode("utf-8")

print(raw)
# b'banana'

print(list(raw))
# [98, 97, 110, 97, 110, 97]

print(raw.decode("utf-8"))
# banana
```

The `b` prefix in `b'banana'` means that Python is displaying a bytes value. It is not an additional character in the text.

For this ASCII example:

| Text piece | Byte value |
|---|---:|
| `b` | 98 |
| `a` | 97 |
| `n` | 110 |

Now consider an emoji:

```python
text = "👋"
print(len(text))
# 1

print(list(text.encode("utf-8")))
# [240, 159, 145, 139]

print(len(text.encode("utf-8")))
# 4
```

In this example, the emoji uses four bytes. A byte tokenizer initially needs four tokens to represent it. Later, BPE could learn a token containing some or all of those bytes if they occur often in the training corpus.

For general Unicode text, a visually perceived character can contain multiple code points. Do not assume that the screen's character count, Python's string length, UTF-8 byte count, and token count are identical.

Our public text API accepts strings that can be encoded as valid UTF-8. Python can also hold unusual strings containing lone surrogate code points; strict UTF-8 encoding rejects those. The separate byte API can round-trip arbitrary bytes, even when they do not form valid UTF-8 text.

## 5. Other tokenization approaches

These approaches provide background for understanding our choice:

| Approach | Basic unit or mechanism | Typical tradeoff |
|---|---|---|
| Word | Whole words | Compact for known words; unfamiliar words require a policy. |
| Character | Characters | Broad character coverage can need a large alphabet; sequences are often long. |
| Byte | All 256 byte values | Complete byte coverage; sequences can be long. |
| BPE | Learned adjacent-pair merges | Reuses frequent pieces; details depend on the base units and boundary rules. |
| WordPiece | Learned subword vocabulary | Often uses greedy longest-matching pieces during encoding. |
| Unigram | A probabilistic subword vocabulary | Can score alternative segmentations. |

SentencePiece is a tokenizer library that supports algorithms including BPE and Unigram. Algorithm names and library names describe different things. These high-level comparisons are supported by the [Hugging Face tokenization algorithms guide](https://huggingface.co/docs/transformers/tokenizer_summary).

Our code implements byte-level BPE directly. It has no dependency on Hugging Face Tokenizers, SentencePiece, or a pretrained tokenizer.

## 6. The exact ID layout in this project

The initial layout is:

| IDs | Meaning |
|---|---|
| 0 through 255 | Each ID represents the byte of the same numeric value. |
| 256 | `<bos>`: beginning-of-sequence marker. |
| 257 | `<eos>`: end marker, intended for use at document boundaries. |
| 258 | `<pad>`: padding marker. |
| 259 onward | Learned BPE tokens. |

The initial vocabulary size is therefore:

```text
256 byte IDs + 3 special IDs = 259 total IDs
```

If a tokenizer has 21 learned merges:

```text
259 + 21 = 280 total IDs
```

Its valid IDs are 0 through 279. Vocabulary size counts IDs; it is not the highest valid ID.

The implementation's `vocab` dictionary stores byte sequences for ordinary IDs. Special IDs are stored separately in `SPECIAL_TOKENS` because they do not represent text bytes. The `vocab_size` property counts both categories.

The sizes and IDs in this layout are our project's design choices. Other tokenizers can assign completely different IDs.

## 7. Training BPE step by step

Tokenizer training learns a sequence of merge rules from text frequency. It does not use gradients, a neural network, or an optimizer.

The procedure is:

1. Receive a collection of separate text documents.
2. Split each document into chunks with the GPT-2-style pattern (section 7.5).
3. Encode each chunk as UTF-8 bytes.
4. Treat each byte as an initial token ID.
5. Count all adjacent ID pairs within each chunk.
6. Choose the pair with the highest count.
7. If several pairs have the same highest count, choose the smallest pair of IDs in lexicographic order.
8. Create a new ID representing the concatenated bytes of that pair.
9. Replace its occurrences from left to right, without overlapping replacements.
10. Update the pair counts for the changed chunks.
11. Repeat until the requested vocabulary size is reached or no eligible pair remains.

The default minimum pair frequency is 2. A pair seen only once is therefore not merged under the default setting.

Two implementations follow this procedure. `train_reference` recounts every pair after every merge: slow, but easy to check. `train` gives exactly the same merges much faster (section 7.6). The tests require the two to agree.

### 7.1 Counting pairs in `banana`

The initial IDs are:

```text
[98, 97, 110, 97, 110, 97]
```

The adjacent pairs are:

```text
(98, 97)
(97, 110)
(110, 97)
(97, 110)
(110, 97)
```

Their counts are:

| Pair | Bytes | Count |
|---|---|---:|
| `(98, 97)` | `b` + `a` | 1 |
| `(97, 110)` | `a` + `n` | 2 |
| `(110, 97)` | `n` + `a` | 2 |

`(97, 110)` wins the tie because 97 is smaller than 110. It becomes ID 259, representing `b'an'`.

After this merge:

```text
[98, 259, 259, 97]
```

### 7.2 The demonstration training corpus

The demo supplies these three separate documents:

```python
documents = [
    "banana banana banana",
    "bandana banana",
    "Hello hello hello!",
]
```

Across this corpus, the first eight learned rules are:

| New ID | Left bytes | Right bytes | Resulting bytes | Pair count at that step |
|---:|---|---|---|---:|
| 259 | `a` | `n` | `an` | 10 |
| 260 | `b` | `an` | `ban` | 5 |
| 261 | `an` | `a` | `ana` | 5 |
| 262 | `ban` | `ana` | `banana` | 4 |
| 263 | space | `banana` | ` banana` | 3 |
| 264 | `e` | `l` | `el` | 3 |
| 265 | `l` | `o` | `lo` | 3 |
| 266 | `el` | `lo` | `ello` | 3 |

Counts are measured in the current token sequences at each training step. They are not all calculated from the original bytes once and reused forever.

For a `banana` occurrence, the successive states are:

```text
[98, 97, 110, 97, 110, 97]
→ [98, 259, 259, 97]
→ [260, 259, 97]
→ [260, 261]
→ [262]
```

The demo requests a vocabulary size of 275 but produces 269 total IDs. It learns ten merges and stops because no remaining pair reaches the default minimum frequency. The requested size is a ceiling, not a promise that enough eligible merges exist.

### 7.3 Overlapping counts and replacements

In `aaaa`, there are three adjacent `aa` pairs:

```text
positions 0–1
positions 1–2
positions 2–3
```

If we merge `aa` into ID 259, the result is:

```text
[259, 259]
```

There are two replacements because replacements cannot reuse a byte already consumed by a previous replacement. Pair frequency and replacement count can therefore differ.

### 7.4 Document boundaries

Training on `["a", "b"]` provides two one-byte documents. There is no adjacent pair to merge. Training on `["ab"]` provides one two-byte document containing the pair `(97, 98)`.

The command-line trainer treats each plain input file as one document, and each line of a `.jsonl` file (its `"text"` field) as one document. Lines inside a plain file are not automatically separate documents. Merges never cross a document boundary, and with pre-splitting they also never cross a chunk boundary.

The file reader uses Python text mode, which can normalize platform line endings when loading files. Exact round-trip guarantees refer to the string supplied to `encode`, or the bytes supplied to `encode_bytes`, rather than every possible original file representation.

### 7.5 Pre-splitting into chunks

Without splitting, BPE learns whatever byte runs are frequent, including ones that join a word to its neighbor. On the robot-and-cat corpus, a tokenizer trained with pre-splitting turned off learns pieces such as `t sa`, `.\n` and `e `.

GPT-2 avoided this by cutting text into chunks first and merging only inside a chunk. Our `GPT2_SPLIT_PATTERN` tries these branches in order:

| Branch | Matches | Example |
|---|---|---|
| Contraction | `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d` (any case) | `it's` → `it` + `'s` |
| Letters | an optional space, then letters | ` world` |
| Digits | an optional space, then digits | ` 2026` |
| Other | an optional space, then symbols, including `_` | `,` or ` !?` |
| Whitespace | spaces, leaving one for the next word | `a   b` → `a`, `  `, ` b` |

```python
from tokenizer.bpe import GPT2_SPLIT_PATTERN, pre_split
print(pre_split("Hello world, it's 2026!", GPT2_SPLIT_PATTERN))
# ['Hello', ' world', ',', ' it', "'s", ' 2026', '!']
```

Every character matches some branch, so the chunks always join back into the original text. One known limitation, shared with GPT-2: combining marks, such as Devanagari vowel signs, count as symbols rather than letters, so those words split into more chunks.

### 7.6 Making training fast without changing the answer

Recounting every pair before each merge takes about 60 seconds for 1,000 merges on 0.5 MB. The fast trainer uses three ideas:

1. **Count each distinct chunk once**, with a weight for how often it occurs. 36 MB of Python source has only about 100,000 distinct chunks.
2. **Update counts instead of recounting.** A merge changes only the chunks containing that pair; the trainer remembers which ones they are.
3. **Find the best pair with a heap** of `(-count, pair)` entries. Entries whose count has since changed are skipped when popped.

Training the 32,768-ID vocabulary on 304 MB of FineWeb-Edu takes about 46 seconds and 1.3 GB of memory.

## 8. Encoding uses learned merge ranks

Encoding new text starts from its byte IDs and applies rules from the frozen merge list. It does not recount frequencies to train a new vocabulary.

The encoding procedure is:

1. Split the text into chunks with the same pattern used in training.
2. Convert each chunk into byte IDs.
3. Find adjacent pairs that appear in the learned merge list.
4. Choose the available pair with the earliest learned rank.
5. Replace non-overlapping occurrences of that pair.
6. Repeat until no learned pair applies, then join the chunks' IDs.

Because the same chunks (` the`, `,`, ` of`) repeat constantly, `encode` remembers each chunk's IDs in a cache, which raised speed from about 1.5 to 10 MB/s on one core. `encode_many` spreads documents across processes, reaching about 50 MB/s on 10 cores for large inputs. tiktoken, which is written in Rust, encodes about 3× faster per core.

`encode_bytes(raw)` treats its input as a single chunk with no splitting, so it works for any bytes, including invalid UTF-8.

For example, suppose the learned rules are:

```text
rank 0: a + b → ID 259
rank 1: b + c → ID 260
```

When encoding `abc`, both pairs are initially available. Rank 0 wins:

```text
[97, 98, 99] → [259, 99]
```

This illustrates why merge order is part of the saved tokenizer definition. Our encoder follows BPE ranks, rather than choosing any convenient longest piece.

## 9. Decoding and exact reconstruction

Every ordinary token ID maps to a byte sequence. Decoding performs two operations:

```text
token IDs → concatenate their byte sequences → decode the full bytes as UTF-8
```

We join bytes before decoding because a token may contain only part of a multibyte character. Decoding each token independently can fail even when the full combined sequence is valid.

The fundamental check is:

```python
assert tokenizer.decode(tokenizer.encode(text)) == text
```

For the byte API:

```python
assert tokenizer.decode_bytes(tokenizer.encode_bytes(raw)) == raw
```

Valid text passed through encode and decode is preserved exactly. Arbitrary IDs sampled by the model can form invalid UTF-8, including a sequence ending halfway through an emoji. Strict decoding then raises `UnicodeDecodeError`. The text API supports `errors="replace"` for displaying such output, but replacement is not lossless.

## 10. Special tokens and their current behavior

Special IDs provide markers for the training pipeline. Reserving an ID does not itself teach a model to respond to it.

The Python API can explicitly add beginning and end markers:

```python
from tokenizer import ByteBPETokenizer

tokenizer = ByteBPETokenizer()
ids = tokenizer.encode("Hi", add_bos=True, add_eos=True)
print(ids)
# [256, 72, 105, 257]

print(tokenizer.decode(ids))
# Hi
```

The CLI `encode` command currently adds neither marker. BOS and EOS flags are available in the Python API. Padding is reserved but no batching or padding utility has been implemented.

Typing the literal string `<eos>` is different from explicitly inserting special ID 257:

```python
print(ByteBPETokenizer().encode("<eos>"))
# [60, 101, 111, 115, 62]
```

Default decoding skips BOS, EOS, and PAD IDs. Setting `skip_special_tokens=False` raises an error when a special ID appears, because special markers have no text bytes in this design. That option does not render their names into the decoded string.

When documents are packed together for model training, `scripts/prepare_data.py` inserts an EOS ID after each one, so the model sees where texts end. BPE training neither inserts EOS nor learns merges involving special tokens.

## 11. The tiny robot-and-cat tokenizer

The training file `examples/tiny_corpus.txt` contains:

```text
The little robot opened the door.
The little robot saw a little cat.
The cat sat beside the robot.
The robot said hello. The cat said hello.
They opened the door and walked outside.
Hello, little robot! Hello, little cat!
```

The project file also ends with a newline. With the current file, this is 222 UTF-8 bytes. Training with vocabulary size 280 and the default minimum frequency produces 21 merges.

The command is:

```sh
python3 -m tokenizer train --input examples/tiny_corpus.txt --output models/tiny_tokenizer.json --vocab-size 280
```

The sentence is first split into the chunks `Hello`, `,`, ` little`, ` robot` and `!`. For this particular saved tokenizer:

```text
Hello, little robot!
→ [72, 101, 278, 44, 271, 272, 33]
```

| ID | Byte sequence | Interpretation |
|---:|---|---|
| 72 | `b'H'` | One byte. |
| 101 | `b'e'` | One byte. |
| 278 | `b'llo'` | Learned three-byte piece. |
| 44 | `b','` | One punctuation byte. |
| 271 | `b' little'` | Learned word with its leading space. |
| 272 | `b' robot'` | Learned word with its leading space. |
| 33 | `b'!'` | One punctuation byte. |

The original sentence contains 20 UTF-8 bytes but encodes as seven tokens.

Before pre-splitting existed, the same corpus produced `[72, 278, 44, 279, 270, 116, 33]`: `H`, `ello`, `,`, ` little ` (with spaces on both sides), `robo`, `t`, `!`. Training with `--no-pre-split` still reproduces it. Both use seven tokens, but the split version's pieces line up with words: ` robot` is one token instead of `robo` + `t`.

For the unfamiliar sentence `A spaceship visited Mars.`, this tokenizer produces 23 tokens for 25 UTF-8 bytes. Most of its byte sequences were not learned as larger pieces in the tiny corpus.

Both sentences decode exactly. The difference measures how efficiently the tokenizer represents these strings, not whether a language model understands them.

## 12. Vocabulary size and efficiency

On the same robot-and-cat corpus, with the same default minimum frequency, the verified results for `Hello, little robot!` are:

| Requested and achieved vocabulary size | Learned merges | Tokens for the sentence |
|---:|---:|---:|
| 259 | 0 | 20 |
| 270 | 11 | 12 |
| 280 | 21 | 7 |

A larger vocabulary gives this example more reusable pieces. More generally, corpus selection, boundary rules, and the learned merge list affect the token count. A large vocabulary does not automatically produce a better language model.

Two useful measurements are:

```text
bytes per token = input UTF-8 byte count / token count
token-count reduction = 1 − token count / original byte count
```

For the 20-byte familiar sentence:

```text
bytes per token = 20 / 7 ≈ 2.86
token-count reduction = 1 − 7 / 20 = 65%
```

For the 25-byte unfamiliar sentence:

```text
bytes per token = 25 / 23 ≈ 1.09
```

Avoid dividing by zero for empty input. Compare tokenizers on the same held-out text collection and report totals, rather than using two differently sized sentences as a general quality benchmark.

These are sequence-length measurements. They do not directly establish file compression: storing each ID as a multi-byte integer and storing the vocabulary introduces additional space costs.

The project's real vocabulary size of 32,768 allows up to:

```text
32,768 − 259 = 32,509 learned merges
```

Trained on 304 MB of FineWeb-Edu and measured on 25 MB of held-out documents from a different file:

| Tokenizer | Vocabulary | Bytes per token |
|---|---:|---:|
| Ours | 32,768 | 4.63 |
| GPT-2 (tiktoken) | 50,257 | 4.64 |

Ours was trained on the same kind of text it is tested on, while GPT-2's tokenizer was trained on different web text. The fair summary is "matches GPT-2's compression on FineWeb-Edu with a smaller vocabulary", not "beats GPT-2".

## 13. How vocabulary size affects the model

The language model has a token embedding table. If vocabulary size is `V` and embedding width is `d`, the table has:

```text
V × d parameters
```

At the project's `V = 32,768` and `d = 768`:

```text
32,768 × 768 = 25,165,824 embedding parameters
```

The table is in `lm/model.py`. Its output layer reuses the same matrix (tied weights), so these 25 million parameters are counted once, out of 124,275,456 in total.

A larger vocabulary increases the size of the embedding table and usually the cost of producing scores over the vocabulary. Efficient tokenization can shorten sequences and fit more raw text into a fixed token context. The best vocabulary therefore depends on both the corpus and the training budget.

The `0.1B` in the project name refers to roughly 100 million model parameters; the final model has about 124 million. It is different from vocabulary size, corpus size, and the number of tokens processed during training.

Changing a tokenizer changes the meaning of IDs. Freeze the tokenizer before a full language-model run, and keep its exact saved definition with the model checkpoint. Two tokenizers can assign different bytes to ID 259 even if their vocabulary sizes match.

## 14. Current implementation map

| File | Responsibility |
|---|---|
| `tokenizer/bpe.py` | Pre-splitting, pair counting, pair replacement, reference and fast training, cached and parallel encoding, decoding, JSON save/load. |
| `tokenizer/documents.py` | Reads documents from plain files or `.jsonl` lines. |
| `tokenizer/__init__.py` | Exposes `ByteBPETokenizer` for importing. |
| `tokenizer/__main__.py` | Implements `python3 -m tokenizer` commands. |
| `tests/test_tokenizer.py` | 23 tokenizer correctness tests. |
| `scripts/sample_fineweb.py` | Downloads a reproducible FineWeb-Edu sample. |
| `scripts/bench_tokenizer.py` | Compares a tokenizer with tiktoken on held-out text. |
| `examples/tiny_corpus.txt` | Small robot-and-cat training corpus. |
| `models/demo_tokenizer.json` | Saved demo merge rules. |
| `models/tiny_tokenizer.json` | Saved rules for the 280-ID tiny tokenizer. |
| `models/fineweb_edu_32k.json` | The 32,768-ID FineWeb-Edu tokenizer. |
| `TOKENIZER_LESSON.md` | Shorter walkthrough and exercises. |
| `README.md` | Commands for running the project. |
| `PLAN.md` | Broader learning, model, and training plan. |

The tokenizer uses only the Python standard library: `collections`, `heapq`, `re`, `multiprocessing`, `json`, `pathlib` and a few others. The CLI uses `argparse`; the tests use `unittest`. Only the download and benchmark scripts need extra packages (`pyarrow`, `huggingface_hub`, `tiktoken`).

### 14.1 `count_pairs`

The core operation is:

```python
counts.update(zip(ids, ids[1:]))
```

`ids[1:]` is the sequence starting at the second ID. `zip` aligns it with the original sequence, giving neighbors. `Counter` increments the count for each pair.

### 14.2 `merge_pair`

This function scans from left to right. If the current ID and next ID match the selected pair, it appends the new ID and advances by two positions. Otherwise, it appends the current ID and advances by one position.

It returns a new list. It does not mutate the caller's list in place.

### 14.3 `ByteBPETokenizer.__init__`

Initialization creates all byte tokens. It then reconstructs learned tokens in order. For a learned pair `(left_id, right_id)`:

```python
new_bytes = vocab[left_id] + vocab[right_id]
```

It records the pair's rank and new ID. Merge operands must already exist, and duplicate merge pairs are rejected. Special IDs cannot be merge operands because they have no entry in the ordinary byte vocabulary.

### 14.4 Public API

| Method or property | Purpose |
|---|---|
| `ByteBPETokenizer.train(texts, vocab_size=280, min_frequency=2, pattern=GPT2_SPLIT_PATTERN)` | Train a new tokenizer on an iterable of separate strings (a generator works). `pattern=None` turns pre-splitting off. |
| `ByteBPETokenizer.train_reference(...)` | The simple trainer with the same arguments; kept to check `train`. |
| `ByteBPETokenizer(merges=(), pattern=GPT2_SPLIT_PATTERN)` | Construct a byte tokenizer or reconstruct one from ordered pairs. |
| `encode(text, add_bos=False, add_eos=False)` | Encode a UTF-8 string, optionally adding explicit markers. |
| `encode_many(texts, add_eos=False, processes=None)` | Encode many documents in parallel processes, in order. |
| `encode_bytes(raw)` | Encode a `bytes` value as one chunk, without pre-splitting. |
| `pattern` | The pre-split regular expression, or `None`. |
| `decode(ids, errors="strict", skip_special_tokens=True)` | Decode IDs to text. |
| `decode_bytes(ids, skip_special_tokens=True)` | Decode IDs to bytes. |
| `save(path)` | Save the definition as JSON. |
| `ByteBPETokenizer.load(path)` | Reload a saved definition. |
| `vocab_size` | Count all IDs, including special IDs. |
| `vocab` | Map ordinary IDs to bytes. |
| `merges` | List the ordered merge pairs. |
| `training_history` | In-memory training trace with pair, new ID, and pair count. |

Pass training documents as `[text]` for a single document. Passing just `text` is rejected to prevent accidentally treating each character as a separate document. Decoding expects ordinary Python integer IDs and rejects unknown IDs.

## 15. Saved JSON format

The saved definition contains the format identifier, format version, special IDs, the pre-split pattern, and ordered merge pairs. A short illustrative example, with the pattern shortened, is:

```json
{
  "format": "educational-byte-bpe",
  "version": 2,
  "special_tokens": {
    "<bos>": 256,
    "<eos>": 257,
    "<pad>": 258
  },
  "pattern": "(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?[^\\W\\d_]+| ...",
  "merges": [
    [97, 110],
    [98, 259]
  ]
}
```

The first pair becomes ID 259 and represents `an`. The second becomes ID 260 and represents `ban`.

The loader recreates the 256 byte tokens, then reconstructs each learned byte sequence from earlier operands. It validates the format/version and special-ID definitions through the current code. The pattern is saved because encoding must split text exactly as training did. Older version 1 files, made before pre-splitting, still load, with `pattern` set to `None`.

The JSON does not contain the original corpus, pair frequency history, requested vocabulary size, or minimum-frequency setting. `training_history` is available on the freshly trained object but is not persisted by `save`. Future reproducibility metadata should record the corpus identity and training settings separately.

Saved tokenizer files live under `models/`, which is ignored by Git in the current project.

## 16. Commands you can run now

Run commands from the project folder. Use Python 3.10 or newer. Text examples use single shell quotes so punctuation such as `!` is passed literally in the usual macOS shell.

### Show the demonstration

```sh
python3 -m tokenizer demo
```

This trains the small demo corpus, shows eight merge rules, checks several round trips, and saves `models/demo_tokenizer.json`.

### Run the correctness tests

```sh
python3 -m unittest discover -s tests -v
```

### Train the tiny tokenizer

```sh
python3 -m tokenizer train --input examples/tiny_corpus.txt --output models/tiny_tokenizer.json --vocab-size 280
```

Multiple input paths provide separate documents. The `--min-frequency` option changes the threshold; its default is 2. Add `--no-pre-split` to train without chunking, for comparison. The command also prints how long training took.

### Encode text

```sh
python3 -m tokenizer encode --model models/tiny_tokenizer.json 'Hello, little robot!'
```

Expected output for the current tiny corpus:

```text
[72, 101, 278, 44, 271, 272, 33]
```

### Inspect token pieces

```sh
python3 -m tokenizer inspect --model models/tiny_tokenizer.json 'Hello, little robot!'
```

This shows the pre-split chunks, byte count, token count, ID, piece bytes, whether a piece is a base byte or learned token, and exact reconstruction.

### Decode the complete sentence

```sh
python3 -m tokenizer decode --model models/tiny_tokenizer.json 72 101 278 44 271 272 33
```

Expected output:

```text
Hello, little robot!
```

### Decode five raw byte IDs

```sh
python3 -m tokenizer decode --model models/tiny_tokenizer.json 72 101 108 108 111
```

Expected output:

```text
Hello
```

Raw byte IDs remain valid even when encoding the same word would prefer a learned merge. Decoding does not require the input to be the shortest or canonical encoding.

If the shell shows `dquote>` instead of its normal prompt, it is waiting for a double quote to close. Press Control+C, then re-enter the command with balanced quotes. That prompt comes from the shell, before Python executes the intended command.

## 17. Using the tokenizer from Python

```python
from tokenizer import ByteBPETokenizer

documents = ["hello hello", "banana banana"]
tokenizer = ByteBPETokenizer.train(documents, vocab_size=280)

text = "Hello 👋\nनमस्ते"
ids = tokenizer.encode(text)
restored = tokenizer.decode(ids)

assert restored == text
print(ids)
print(restored)

tokenizer.save("models/my_tokenizer.json")
loaded = ByteBPETokenizer.load("models/my_tokenizer.json")

assert loaded.encode(text) == ids
```

This example has a different training corpus from the robot-and-cat tokenizer, so its learned IDs can represent different pieces.

## 18. What the tests establish

| Test | Behavior checked |
|---|---|
| All-byte round trip | All 256 byte values can be encoded and reconstructed with the base tokenizer. |
| Unicode and whitespace round trip | Trained tokenizer preserves selected empty, multilingual, whitespace, literal-marker, and null-character examples. |
| Repetition compression | Learned rules shorten a repeated `banana` example. |
| Document boundaries | Pairs are not counted across separate documents. |
| Left-to-right overlap handling | Overlapping candidates are replaced without reusing positions. |
| Earlier merge rank | Available lower-ranked rules take precedence. |
| Deterministic ties | Pair-count ties produce stable rules for the tested document orderings. |
| Explicit special tokens | BOS/EOS insertion and literal marker text behave as designed. |
| Save/reload | A saved and loaded tokenizer preserves the tested encoding. |
| Invalid inputs | Selected invalid training inputs, merge operands, and unknown IDs are rejected. |
| Pre-split chunks | Known sentences split as expected; chunks of 200 random multilingual strings always join back into the original. |
| Merges stay inside chunks | With splitting, no learned token contains `.` together with other bytes; without it, one does. |
| Version 1 files | Old tokenizer files load without pre-splitting. |
| Fast trainer = reference | Identical merges and counts on the tiny corpus and 60 random corpora full of ties. |
| Chunk cache | Cached encoding equals uncached encoding, and the cache stays within its size limit. |
| Parallel encoding | `encode_many` with one or two processes equals `encode`, document by document. |
| Reading documents | `.jsonl` lines and whole files are read as documents. |

These checks support the intended behavior. Speed is measured separately with `scripts/bench_tokenizer.py`. The tests do not demonstrate every possible malformed-file case or establish the quality of a language model.

## 19. Limitations and what changed since the first version

### 19.1 Speed and memory

The first version recounted every pair after every merge and rescanned sequences while encoding: easy to inspect, too slow for real data. That version survives as `train_reference`, the test oracle. The fast trainer and the cached, parallel encoder (sections 7.6 and 8) replaced it for real work.

Remaining limits: encoding is pure Python and about 3× slower per core than tiktoken's Rust. Training keeps one entry per distinct chunk in memory, about 1.3 GB for 304 MB of web text.

### 19.2 Boundary and normalization policy

The tokenizer now pre-splits text with a GPT-2-style regular expression (section 7.5), so learned pieces stay within words, numbers, symbol runs or whitespace runs. It still does no lowercase conversion, accent stripping, or Unicode normalization.

A common tokenizer pipeline may include normalization, pre-tokenization, a segmentation model, and post-processing. Ours uses pre-tokenization, UTF-8 conversion and ranked BPE, with optional explicit BOS/EOS insertion. The [Hugging Face tokenization pipeline guide](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html) describes these general components.

Changing the boundary or normalization policy changes the vocabulary and encoded sequences, so it is fixed for the full language-model run: the 32k tokenizer's file hash is recorded and checked by the training code.

### 19.3 Data representativeness

The tiny corpus repeats robots, cats, doors, and greetings, and its learned pieces reflect that. It is useful for understanding the algorithm. The real tokenizer is trained on 304 MB of FineWeb-Edu, sampled evenly through one file.

Training documents and held-out documents come from different files, so the compression comparison with GPT-2 is measured on text the tokenizer never saw.

### 19.4 Integration work

Built since the first version: tokenized dataset shards with EOS between documents, model batches, the embedding table, next-token training and generation (see `MODEL_LESSON.md` and `TRAINING_LESSON.md`). Training uses fixed-length windows cut from the token stream, so no padding is needed. A reserved PAD ID alone does not create a padding mask. Generation stops when the model produces EOS.

## 20. Common misunderstandings

**“One token equals one word.”** A token can be smaller or larger than a word. Our sample contains both `llo` (part of a word) and ` little` (a word with its leading space) as tokens.

**“The token ID tells us its meaning.”** IDs are labels. ID 278 represents `llo` only in the specific tiny tokenizer discussed here. The model learns vectors associated with the IDs.

**“Unknown words cannot be encoded.”** Our complete byte vocabulary can represent unfamiliar UTF-8 text, though it may need many small tokens.

**“If an emoji token cannot decode alone, it is broken.”** A token may contain only part of its UTF-8 byte sequence. Decode the joined bytes.

**“Tokenizer training is language-model training.”** BPE training counts and merges sequences. Language-model training optimizes neural-network parameters.

**“Setting vocabulary size 280 forces 280 IDs.”** The threshold and available pairs can make training stop early. The demo requests 275 and reaches 269.

**“More tokens in the vocabulary always improves the model.”** Vocabulary choices affect sequence length, coverage efficiency, model size, and output computation. They must be evaluated against the corpus and budget.

**“Fewer output IDs means the tokenizer understands the sentence.”** It indicates that the text matches reusable byte sequences in the learned vocabulary.

**“The literal text `<eos>` automatically ends a document.”** Our API treats it as ordinary text unless the special ID is explicitly inserted.

**“The saved JSON contains the full training process.”** It preserves the tokenizer definition, while training settings, source data, and frequency history need separate records.

## 21. Practice exercises and answer guide

1. **Why does the base tokenizer start at 259 IDs?** It includes 256 possible byte values and three special markers.
2. **What does ID 259 represent in the demo?** The byte sequence `an`, learned from pair `(97, 110)`.
3. **Why can our tokenizer encode an unseen emoji?** Every constituent byte already has an initial ID.
4. **Why are there three `aa` pairs in `aaaa` but two replacements?** Pair counting includes overlapping positions; replacement consumes non-overlapping pairs.
5. **Why does the demo stop at 269 IDs?** The remaining pair counts fall below the default minimum frequency of two.
6. **Why is a document boundary different from a newline in this CLI?** Each plain file is one document, while its newlines are ordinary content (and their own whitespace chunks).
7. **Why does `Hello, little robot!` use seven tokens?** The tokenizer learned several recurring multi-byte pieces from the tiny corpus, including the whole words ` little` and ` robot`.
8. **Can two different ID sequences decode to the same string?** Yes. Raw bytes and learned combined pieces can reconstruct the same bytes.
9. **Does a shorter token sequence prove better model quality?** No. It measures representation efficiency; model quality requires separate evaluation.
10. **What must stay with a trained model checkpoint?** Its exact tokenizer definition, so every ID retains the intended meaning.
11. **Why does pre-splitting stop a token like `t sa` from being learned?** `t` and ` sa…` are in different chunks, and pairs are only counted inside a chunk.
12. **How do we know the fast trainer is correct?** Tests require it to learn exactly the same merges and counts as the simple reference trainer.

Practical experiment: train three tokenizers on the same tiny corpus with vocabulary sizes 259, 270, and 280. Inspect the same sentence with each, verify reconstruction, and compare bytes per token. Save them to separate files so the experiment does not overwrite the current tiny tokenizer.

## 22. Questions this source can support in NotebookLM

- Explain byte-level BPE to someone who knows basic Python.
- Walk through the `banana` merges using our exact ID layout.
- Explain why the two saved tokenizers can have different meanings for learned IDs.
- Explain the role of merge ranks during encoding.
- Make flashcards covering bytes, UTF-8, vocabulary, merges, and special IDs.
- Quiz me on the difference between tokenizer training and neural-network training.
- Explain why pre-splitting improves the learned vocabulary.
- Explain the three ideas that make the fast trainer fast, and why it gives the same answer.
- Explain how token IDs connect to embeddings and next-token prediction.

The project-specific facts in this source describe the October 31, 2026 implementation. The document is a source for learning; illustrative command blocks and questions do not change the implementation.

## 23. Sources and scope

Primary evidence for project behavior: the current `tokenizer/bpe.py`, `tokenizer/__main__.py`, `tests/test_tokenizer.py`, `scripts/bench_tokenizer.py`, and `examples/tiny_corpus.txt`. The sample IDs, byte counts, and vocabulary comparisons in this lesson were re-checked against those files on October 31, 2026.

Background references:

- [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html): strings, byte encodings, and Unicode behavior.
- [Hugging Face tokenization algorithms guide](https://huggingface.co/docs/transformers/tokenizer_summary): general tokenizer families and the distinction between algorithms and libraries.
- [Hugging Face tokenization pipeline guide](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html): normalization, pre-tokenization, model, and post-processing components.
- [Radford et al.: Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): byte-level BPE with pre-splitting so merges do not cross character categories.
- [Sennrich, Haddow, and Birch: Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): research context for BPE-based subword segmentation. Our educational byte implementation is its own simplified design.
