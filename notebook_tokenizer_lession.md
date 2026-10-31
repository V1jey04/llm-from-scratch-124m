# Tokenizers: a beginner lesson and the current implementation

Project: Write a tokenizer and Transformer from scratch, then train a roughly 0.1B parameter language model.

Implementation snapshot: September 30, 2026.

Audience: a learner who knows basic Python and is beginning machine learning.

This is a standalone learning source. The explanations and worked examples below can be understood without opening the project files. File names identify the implementation being described. Examples with learned IDs refer to specific training corpora; those IDs are not universal.

## 1. What we have built so far

We have built a small, functioning **byte-level byte pair encoding tokenizer**, usually called byte-level BPE, using the Python standard library.

It can:

- Learn repeated byte sequences from training documents.
- Encode text into integer token IDs.
- Decode those IDs back into the original text.
- Handle unfamiliar words, emoji, and valid UTF-8 text from different languages.
- Preserve case, punctuation, spaces, tabs, and newlines in strings passed to its API.
- Save its learned rules as JSON and reload them.
- Display the byte sequence represented by each token.

The project currently contains two small saved tokenizers: a demonstration tokenizer and a tokenizer trained on a short robot-and-cat text file. Ten correctness tests have passed. This is an educational reference implementation. We have not yet built the Transformer, embeddings, next-token predictor, or full language-model training pipeline.

Our long-term plan proposes a vocabulary of 16,384 total IDs for a roughly 98-million-parameter model. The current examples use only 269 or 280 total IDs. The small vocabulary helps us see exactly what the algorithm does before optimizing it for larger data.

## 2. Why a language model needs tokenization

Neural networks calculate with numbers. Tokenization provides a consistent way to represent text as a sequence of numbers.

The eventual pipeline will be:

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

The tokenizer is responsible for mapping between text and IDs. The future language model will learn which IDs are likely to follow which other IDs.

For example, our tiny tokenizer maps:

```text
Hello, little robot!
```

to:

```text
[72, 278, 44, 279, 270, 116, 33]
```

The tokenizer has not understood the sentence. It has represented the sentence using reusable pieces of bytes.

## 3. Terms you should understand

| Term | Meaning in this lesson |
|---|---|
| Token | One unit in the encoded sequence. In our tokenizer, an ordinary token represents one or more bytes. |
| Token ID | An integer identifying a token. |
| Vocabulary | The complete set of IDs and their meanings, including special IDs. |
| Vocabulary size | The number of distinct IDs available. |
| Corpus | The text collection used to train the tokenizer or a future language model. |
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

The current procedure is:

1. Receive a collection of separate text documents.
2. Encode each document as UTF-8 bytes.
3. Treat each byte as an initial token ID.
4. Count all adjacent ID pairs within each document.
5. Choose the pair with the highest count.
6. If several pairs have the same highest count, choose the smallest pair of IDs in lexicographic order.
7. Create a new ID representing the concatenated bytes of that pair.
8. Replace its occurrences from left to right, without overlapping replacements.
9. Count pairs again in the updated sequences.
10. Repeat until the requested vocabulary size is reached or no eligible pair remains.

The default minimum pair frequency is 2. A pair seen only once is therefore not merged under the default setting.

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

The command-line trainer treats each input file as one document. Lines inside a file are not automatically treated as separate documents. A newline is an ordinary byte, and learned tokens can include it. Merges do not cross the boundary between input files.

The file reader uses Python text mode, which can normalize platform line endings when loading files. Exact round-trip guarantees refer to the string supplied to `encode`, or the bytes supplied to `encode_bytes`, rather than every possible original file representation.

## 8. Encoding uses learned merge ranks

Encoding new text starts from its byte IDs and applies rules from the frozen merge list. It does not recount frequencies to train a new vocabulary.

The current encoding procedure is:

1. Convert the input into byte IDs.
2. Find adjacent pairs that appear in the learned merge list.
3. Choose the available pair with the earliest learned rank.
4. Replace non-overlapping occurrences of that pair.
5. Repeat until no learned pair applies.

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

Valid text passed through encode and decode is preserved exactly. Arbitrary IDs sampled by a future model can form invalid UTF-8, including a sequence ending halfway through an emoji. Strict decoding then raises `UnicodeDecodeError`. The text API supports `errors="replace"` for displaying such output, but replacement is not lossless.

## 10. Special tokens and their current behavior

Special IDs provide markers for the future training pipeline. Reserving an ID does not itself teach a model to respond to it.

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

When we later pack multiple documents for model training, we must explicitly insert EOS IDs if we want the model to see document boundaries. BPE training currently neither inserts EOS nor trains merges involving special tokens.

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

For this particular saved tokenizer:

```text
Hello, little robot!
→ [72, 278, 44, 279, 270, 116, 33]
```

| ID | Byte sequence | Interpretation |
|---:|---|---|
| 72 | `b'H'` | One byte. |
| 278 | `b'ello'` | Learned four-byte piece. |
| 44 | `b','` | One punctuation byte. |
| 279 | `b' little '` | Learned piece including both surrounding spaces. |
| 270 | `b'robo'` | Learned partial word. |
| 116 | `b't'` | One byte. |
| 33 | `b'!'` | One punctuation byte. |

The original sentence contains 20 UTF-8 bytes but encodes as seven tokens.

For the unfamiliar sentence `A spaceship visited Mars.`, this tokenizer produces 23 tokens for 25 UTF-8 bytes. Most of its byte sequences were not learned as larger pieces in the tiny corpus.

Both sentences decode exactly. The difference measures how efficiently the tokenizer represents these strings, not whether a language model understands them.

## 12. Vocabulary size and efficiency

On the same robot-and-cat corpus, with the same default minimum frequency, the verified results for `Hello, little robot!` are:

| Requested and achieved vocabulary size | Learned merges | Tokens for the sentence |
|---:|---:|---:|
| 259 | 0 | 20 |
| 270 | 11 | 14 |
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

The planned vocabulary size of 16,384 allows up to:

```text
16,384 − 259 = 16,125 learned merges
```

We will need a much more representative corpus and better performance before training that vocabulary.

## 13. How vocabulary size affects the future model

A future language model will have a token embedding table. If vocabulary size is `V` and embedding width is `d`, the table has:

```text
V × d parameters
```

At the planned `V = 16,384` and `d = 768`:

```text
16,384 × 768 = 12,582,912 embedding parameters
```

This embedding table has not been implemented yet. Token IDs currently have no learned embedding vectors.

A larger vocabulary increases the size of the embedding table and usually the cost of producing scores over the vocabulary. Efficient tokenization can shorten sequences and fit more raw text into a fixed token context. The best vocabulary therefore depends on both the corpus and the training budget.

The planned `0.1B` refers to roughly 100 million model parameters. It is different from vocabulary size, corpus size, and the number of tokens processed during training.

Changing a tokenizer changes the meaning of IDs. Freeze the tokenizer before a full language-model run, and keep its exact saved definition with the model checkpoint. Two tokenizers can assign different bytes to ID 259 even if their vocabulary sizes match.

## 14. Current implementation map

| File | Responsibility |
|---|---|
| `tokenizer/bpe.py` | Pair counting, pair replacement, training, encoding, decoding, JSON save/load. |
| `tokenizer/__init__.py` | Exposes `ByteBPETokenizer` for importing. |
| `tokenizer/__main__.py` | Implements `python3 -m tokenizer` commands. |
| `tests/test_tokenizer.py` | Ten tokenizer correctness tests. |
| `examples/tiny_corpus.txt` | Small robot-and-cat training corpus. |
| `models/demo_tokenizer.json` | Saved demo merge rules. |
| `models/tiny_tokenizer.json` | Saved rules for the 280-ID tiny tokenizer. |
| `TOKENIZER_LESSON.md` | Shorter walkthrough and exercises. |
| `README.md` | Commands for running the project. |
| `PLAN.md` | Broader learning, model, and training plan. |

The code uses `collections.Counter`, `json`, `pathlib.Path`, and `typing.Iterable`. The CLI uses `argparse`. The test suite uses `unittest` and `tempfile`. These are Python standard-library tools. No GPU or external ML package is needed for this stage.

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
| `ByteBPETokenizer.train(texts, vocab_size=280, min_frequency=2)` | Train a new tokenizer on an iterable of separate strings. |
| `ByteBPETokenizer(merges=())` | Construct a byte tokenizer or reconstruct one from ordered pairs. |
| `encode(text, add_bos=False, add_eos=False)` | Encode a UTF-8 string, optionally adding explicit markers. |
| `encode_bytes(raw)` | Encode a `bytes` value. |
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

The saved definition contains the format identifier, format version, special IDs, and ordered merge pairs. A short illustrative example is:

```json
{
  "format": "educational-byte-bpe",
  "version": 1,
  "special_tokens": {
    "<bos>": 256,
    "<eos>": 257,
    "<pad>": 258
  },
  "merges": [
    [97, 110],
    [98, 259]
  ]
}
```

The first pair becomes ID 259 and represents `an`. The second becomes ID 260 and represents `ban`.

The loader recreates the 256 byte tokens, then reconstructs each learned byte sequence from earlier operands. It validates the format/version and special-ID definitions through the current code.

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

Multiple input paths provide separate documents. The `--min-frequency` option changes the threshold; its default is 2.

### Encode text

```sh
python3 -m tokenizer encode --model models/tiny_tokenizer.json 'Hello, little robot!'
```

Expected output for the current tiny corpus:

```text
[72, 278, 44, 279, 270, 116, 33]
```

### Inspect token pieces

```sh
python3 -m tokenizer inspect --model models/tiny_tokenizer.json 'Hello, little robot!'
```

This shows the byte count, token count, ID, piece bytes, whether a piece is a base byte or learned token, and exact reconstruction.

### Decode the complete sentence

```sh
python3 -m tokenizer decode --model models/tiny_tokenizer.json 72 278 44 279 270 116 33
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

These checks support the intended reference behavior. They do not measure large-corpus speed, demonstrate every possible malformed-file case, or establish the quality of a future language model.

## 19. Current limitations and planned improvements

### 19.1 Speed and memory

The trainer loads documents into memory, stores IDs as Python integers, recounts all pairs after every merge, and rebuilds the sequences. The encoder scans its current sequence repeatedly to apply ranked merges. These choices make the algorithm easy to inspect but expensive at large scale.

Before training a 16k vocabulary, we need representative benchmarks, faster pair-count maintenance, and a more efficient encoder. Keeping the current version as a reference will help verify optimized versions.

### 19.2 Boundary and normalization policy

Our current API has no lowercase conversion, accent stripping, or Unicode normalization. It also has no separate regex or word-splitting stage. Learned pieces can include spaces, punctuation, newlines, and multiple words within one document.

A common tokenizer pipeline may include normalization, pre-tokenization, a segmentation model, and post-processing. Our current pipeline uses UTF-8 conversion and ranked BPE, with optional explicit BOS/EOS insertion. The [Hugging Face tokenization pipeline guide](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html) describes these general components.

Changing boundary or normalization policies later can change the vocabulary and encoded sequences, so those choices must be fixed before the full language-model run.

### 19.3 Data representativeness

The tiny corpus repeats robots, cats, doors, and greetings. Its learned pieces reflect those repetitions. It is useful for understanding the algorithm, but final tokenizer training must sample the intended language-model corpus more broadly.

Use training documents to learn the vocabulary. Hold out validation/test documents for assessing efficiency and future model performance. Tokenizer training on held-out text can influence later comparisons.

### 19.4 Integration work

Still to build: tokenized dataset shards, document packing with explicit EOS, model batches, padding/loss masks where needed, embeddings, next-token training, and generation. A reserved PAD ID alone does not create a padding mask, and an EOS ID alone does not create stopping behavior.

## 20. Common misunderstandings

**“One token equals one word.”** A token can be much smaller or larger than a word. Our sample contains both `robo` and ` little ` as tokens.

**“The token ID tells us its meaning.”** IDs are labels. ID 278 represents `ello` only in the specific tiny tokenizer discussed here. A future model learns vectors associated with the IDs.

**“Unknown words cannot be encoded.”** Our complete byte vocabulary can represent unfamiliar UTF-8 text, though it may need many small tokens.

**“If an emoji token cannot decode alone, it is broken.”** A token may contain only part of its UTF-8 byte sequence. Decode the joined bytes.

**“Tokenizer training is language-model training.”** BPE training counts and merges sequences. The future language-model training will optimize neural-network parameters.

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
6. **Why is a document boundary different from a newline in this CLI?** Each file is one document, while its newlines are ordinary content.
7. **Why does `Hello, little robot!` use seven tokens?** The tokenizer learned several recurring multi-byte pieces from the tiny corpus.
8. **Can two different ID sequences decode to the same string?** Yes. Raw bytes and learned combined pieces can reconstruct the same bytes.
9. **Does a shorter token sequence prove better model quality?** No. It measures representation efficiency; model quality requires separate evaluation.
10. **What must stay with a future trained model checkpoint?** Its exact tokenizer definition, so every ID retains the intended meaning.

Practical experiment: train three tokenizers on the same tiny corpus with vocabulary sizes 259, 270, and 280. Inspect the same sentence with each, verify reconstruction, and compare bytes per token. Save them to separate files so the experiment does not overwrite the current tiny tokenizer.

## 22. Questions this source can support in NotebookLM

- Explain byte-level BPE to someone who knows basic Python.
- Walk through the `banana` merges using our exact ID layout.
- Explain why the two saved tokenizers can have different meanings for learned IDs.
- Explain the role of merge ranks during encoding.
- Make flashcards covering bytes, UTF-8, vocabulary, merges, and special IDs.
- Quiz me on the difference between tokenizer training and neural-network training.
- Explain the current tokenizer's memory and speed limitations.
- Explain how token IDs will connect to embeddings and next-token prediction later.

The project-specific facts in this source describe the September 30, 2026 implementation. Planned features are explicitly described as future work. The document is a source for learning; illustrative command blocks and questions do not change the implementation.

## 23. Sources and scope

Primary evidence for project behavior: the current `tokenizer/bpe.py`, `tokenizer/__main__.py`, `tests/test_tokenizer.py`, and `examples/tiny_corpus.txt`. The sample IDs, byte counts, and vocabulary comparisons in this lesson were checked against those files.

Background references:

- [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html): strings, byte encodings, and Unicode behavior.
- [Hugging Face tokenization algorithms guide](https://huggingface.co/docs/transformers/tokenizer_summary): general tokenizer families and the distinction between algorithms and libraries.
- [Hugging Face tokenization pipeline guide](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html): normalization, pre-tokenization, model, and post-processing components.
- [Sennrich, Haddow, and Birch: Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909): research context for BPE-based subword segmentation. Our educational byte implementation is its own simplified design.
