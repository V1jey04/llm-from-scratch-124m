"""Byte-level BPE: a transparent reference trainer and a fast one that must agree."""

from collections import Counter, defaultdict
from functools import partial
import heapq
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path
import re
from typing import Iterable, Iterator

# GPT-2's pre-split rule, written for the standard library's `re`. In order:
# contractions ('s, 're, ...; case-insensitive as in GPT-4's version), letters
# (`[^\W\d_]` is a word character that is not a digit or underscore), digits,
# other symbols (including `_`), and whitespace. Each of the first four may take
# one leading space. Every character matches some branch, so the chunks
# always join back into the original text.
GPT2_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?(?:[^\s\w]|_)+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def pre_split(text: str, pattern: str | None) -> list[str]:
    """Cut text into chunks; BPE merges never cross a chunk boundary."""
    if pattern is None:
        return [text] if text else []
    return re.findall(pattern, text)


def count_pairs(documents: list[list[int]]) -> Counter:
    """Count adjacent IDs within each document, never between documents."""
    counts = Counter()
    for ids in documents:
        counts.update(zip(ids, ids[1:]))
    return counts


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping occurrences, scanning from left to right."""
    result = []
    position = 0
    while position < len(ids):
        if position + 1 < len(ids) and (ids[position], ids[position + 1]) == pair:
            result.append(new_id)
            position += 2
        else:
            result.append(ids[position])
            position += 1
    return result


def learn_merges_reference(sequences: list[list[int]], max_merges: int, min_frequency: int,
                           first_id: int) -> tuple[list, list]:
    """Recount every pair in every sequence before each merge. Slow but obvious."""
    merges = []
    history = []
    while len(merges) < max_merges:
        counts = count_pairs(sequences)
        if not counts:
            break
        # Highest frequency wins; equal counts use the smallest pair of IDs.
        pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
        if counts[pair] < min_frequency:
            break
        new_id = first_id + len(merges)
        history.append({"pair": pair, "new_id": new_id, "count": counts[pair]})
        sequences = [merge_pair(ids, pair, new_id) for ids in sequences]
        merges.append(pair)
    return merges, history


def learn_merges(chunk_counts: Counter, max_merges: int, min_frequency: int,
                 first_id: int) -> tuple[list, list]:
    """Learn the same merges as `learn_merges_reference`, doing far less work.

    1. Each distinct chunk is stored once with its count: " the" may occur a
       million times but is recounted as one entry with weight 1,000,000.
    2. Pair counts are kept up to date. A merge only touches the chunks that
       contain the merged pair (`where` remembers which ones).
    3. A heap finds the most frequent pair without scanning every count.
    """
    words = [list(chunk) for chunk in chunk_counts]
    weights = list(chunk_counts.values())
    pair_counts = Counter()
    where = defaultdict(set)
    for index, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_counts[pair] += weights[index]
            where[pair].add(index)

    # The smallest (-count, pair) is the most frequent pair, with ties going to
    # the smallest pair, exactly like the reference. Counts change after every
    # merge, so the heap holds old entries too; we skip any whose count is stale.
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    history = []
    while len(merges) < max_merges and heap:
        negative_count, pair = heapq.heappop(heap)
        count = -negative_count
        if pair_counts.get(pair) != count:
            continue
        if count < min_frequency:
            break
        new_id = first_id + len(merges)
        history.append({"pair": pair, "new_id": new_id, "count": count})
        merges.append(pair)

        changed = set()
        for index in where.pop(pair):
            old = words[index]
            new = merge_pair(old, pair, new_id)
            if len(new) == len(old):
                continue  # An earlier merge already removed the pair from this chunk.
            # Remove the chunk's old pairs and add its new ones.
            for old_pair in zip(old, old[1:]):
                pair_counts[old_pair] -= weights[index]
                changed.add(old_pair)
            for new_pair in zip(new, new[1:]):
                pair_counts[new_pair] += weights[index]
                changed.add(new_pair)
                where[new_pair].add(index)
            words[index] = new
        for changed_pair in changed:
            if pair_counts[changed_pair] > 0:
                heapq.heappush(heap, (-pair_counts[changed_pair], changed_pair))
            else:
                del pair_counts[changed_pair]
    return merges, history


class ByteBPETokenizer:
    """IDs 0–255 are bytes; 256–258 are special; learned IDs start at 259."""

    BOS_ID = 256
    EOS_ID = 257
    PAD_ID = 258
    BASE_VOCAB_SIZE = 259
    SPECIAL_TOKENS = {256: "<bos>", 257: "<eos>", 258: "<pad>"}
    CHUNK_CACHE_SIZE = 500_000

    def __init__(self, merges: Iterable[tuple[int, int]] = (), pattern: str | None = GPT2_SPLIT_PATTERN):
        if pattern is not None and not isinstance(pattern, str):
            raise ValueError("pattern must be a regular expression string or None.")
        self.pattern = pattern
        self.merges = []
        self.vocab = {byte: bytes([byte]) for byte in range(256)}
        self.training_history = []
        self._ranks = {}
        self._chunk_cache = {}

        for pair in merges:
            if len(pair) != 2:
                raise ValueError("Each merge must contain two IDs.")
            pair = tuple(pair)
            if any(type(token_id) is not int or token_id not in self.vocab for token_id in pair):
                raise ValueError("Merge operands must be bytes or earlier learned tokens.")
            if pair in self._ranks:
                raise ValueError("Duplicate merge pair.")
            new_id = self.BASE_VOCAB_SIZE + len(self.merges)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self._ranks[pair] = (len(self.merges), new_id)
            self.merges.append(pair)

    @property
    def vocab_size(self) -> int:
        """Total IDs, including the three special tokens."""
        return self.BASE_VOCAB_SIZE + len(self.merges)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 280, min_frequency: int = 2,
              pattern: str | None = GPT2_SPLIT_PATTERN):
        """Learn merges from separate documents.

        Each document is pre-split into chunks, and pairs are counted inside
        chunks only. With pattern=None, each whole document is one chunk.
        Documents are read one at a time and only distinct chunks are kept, so
        `texts` can be a generator over a large corpus.
        """
        cls._check_training_args(texts, vocab_size, min_frequency)
        chunk_counts = Counter()
        documents = 0
        for text in texts:
            documents += 1
            chunk_counts.update(chunk.encode("utf-8") for chunk in pre_split(text, pattern))
        if not documents:
            raise ValueError("Provide at least one training document.")
        merges, history = learn_merges(chunk_counts, vocab_size - cls.BASE_VOCAB_SIZE,
                                       min_frequency, cls.BASE_VOCAB_SIZE)
        tokenizer = cls(merges, pattern=pattern)
        tokenizer.training_history = history
        return tokenizer

    @classmethod
    def train_reference(cls, texts: Iterable[str], vocab_size: int = 280, min_frequency: int = 2,
                        pattern: str | None = GPT2_SPLIT_PATTERN):
        """The simple trainer: tests check that `train` learns exactly the same merges."""
        cls._check_training_args(texts, vocab_size, min_frequency)
        texts = list(texts)
        if not texts:
            raise ValueError("Provide at least one training document.")
        # From here on, "sequences" are chunks: merges cannot cross either boundary.
        sequences = [list(chunk.encode("utf-8")) for text in texts for chunk in pre_split(text, pattern)]
        merges, history = learn_merges_reference(sequences, vocab_size - cls.BASE_VOCAB_SIZE,
                                                 min_frequency, cls.BASE_VOCAB_SIZE)
        tokenizer = cls(merges, pattern=pattern)
        tokenizer.training_history = history
        return tokenizer

    @classmethod
    def _check_training_args(cls, texts, vocab_size, min_frequency) -> None:
        if isinstance(texts, str):
            raise TypeError("Pass a list of documents, for example [text].")
        if type(vocab_size) is not int or vocab_size < cls.BASE_VOCAB_SIZE:
            raise ValueError("vocab_size must be an integer of at least 259.")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer.")

    def encode_bytes(self, raw: bytes) -> list[int]:
        """Apply the earliest available learned merge until none apply.

        This treats `raw` as one chunk, so it works for any bytes, even invalid
        UTF-8. `encode` pre-splits text first and calls this once per chunk.
        """
        ids = list(raw)
        while len(ids) > 1:
            available = {pair for pair in zip(ids, ids[1:]) if pair in self._ranks}
            if not available:
                break
            pair = min(available, key=lambda candidate: self._ranks[candidate][0])
            ids = merge_pair(ids, pair, self._ranks[pair][1])
        return ids

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.BOS_ID] if add_bos else []
        for chunk in pre_split(text, self.pattern):
            ids.extend(self._encode_chunk(chunk))
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def _encode_chunk(self, chunk: str) -> tuple[int, ...]:
        """Encode one chunk, remembering the answer: the same chunks repeat endlessly."""
        ids = self._chunk_cache.get(chunk)
        if ids is None:
            if len(self._chunk_cache) >= self.CHUNK_CACHE_SIZE:
                self._chunk_cache.clear()  # Bound memory; common chunks come back quickly.
            ids = self._chunk_cache[chunk] = tuple(self.encode_bytes(chunk.encode("utf-8")))
        return ids

    def encode_many(self, texts: Iterable[str], *, add_eos: bool = False,
                    processes: int | None = None) -> Iterator[list[int]]:
        """Encode documents in parallel worker processes, yielding results in order.

        processes=None uses every CPU core; processes=1 stays in this process.
        Scripts that call this must be guarded by `if __name__ == "__main__":`.
        """
        if processes == 1:
            for text in texts:
                yield self.encode(text, add_eos=add_eos)
            return
        texts = iter(texts)
        encode = partial(_encode_in_worker, add_eos=add_eos)
        with multiprocessing.Pool(processes, _start_worker, (self.merges, self.pattern)) as pool:
            # Hand over a block at a time: given a whole stream, the pool would
            # read all of it into memory before the first result came back.
            while block := list(islice(texts, 64 * (processes or os.cpu_count()))):
                yield from pool.imap(encode, block, chunksize=16)

    def decode_bytes(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> bytes:
        pieces = []
        for token_id in ids:
            if type(token_id) is not int:
                raise ValueError("Token IDs must be integers.")
            if token_id in self.SPECIAL_TOKENS:
                if skip_special_tokens:
                    continue
                raise ValueError("Special tokens have no text bytes; skip them to decode text.")
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token ID: {token_id}")
            pieces.append(self.vocab[token_id])
        return b"".join(pieces)

    def decode(self, ids: Iterable[int], *, errors: str = "strict", skip_special_tokens: bool = True) -> str:
        # Join bytes before decoding: one token can hold only part of an emoji.
        return self.decode_bytes(ids, skip_special_tokens=skip_special_tokens).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "educational-byte-bpe",
            "version": 2,
            "special_tokens": {name: token_id for token_id, name in self.SPECIAL_TOKENS.items()},
            "pattern": self.pattern,
            "merges": self.merges,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_specials = {name: token_id for token_id, name in cls.SPECIAL_TOKENS.items()}
        if payload.get("format") != "educational-byte-bpe" or payload.get("version") not in (1, 2):
            raise ValueError("Unsupported tokenizer format or version.")
        if payload.get("special_tokens") != expected_specials:
            raise ValueError("Special token IDs do not match this tokenizer.")
        # Version 1 files were trained without pre-splitting.
        pattern = payload["pattern"] if payload["version"] == 2 else None
        return cls(payload["merges"], pattern=pattern)


# Worker processes for `encode_many`. Each worker builds its own tokenizer (and
# its own chunk cache) once, then encodes whole documents sent to it.
_worker_tokenizer = None


def _start_worker(merges, pattern) -> None:
    global _worker_tokenizer
    _worker_tokenizer = ByteBPETokenizer(merges, pattern=pattern)


def _encode_in_worker(text: str, add_eos: bool) -> list[int]:
    return _worker_tokenizer.encode(text, add_eos=add_eos)
