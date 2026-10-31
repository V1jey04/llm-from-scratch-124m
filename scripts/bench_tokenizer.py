"""Compare a trained tokenizer with tiktoken on held-out text.

    python3 scripts/bench_tokenizer.py --model models/tokenizer.json --text data/fineweb_edu/heldout.jsonl

Reports bytes per token (higher = better compression), single-process encode
speed, parallel encode speed, and exact round-trips on tricky strings. Use text
the tokenizer was NOT trained on, or the compression number flatters it.
tiktoken is only a baseline here; install it with `pip install tiktoken`.
"""

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer import ByteBPETokenizer  # noqa: E402
from tokenizer.documents import read_documents  # noqa: E402

TRICKY = [
    "",
    "Hello, world!",
    "  leading and trailing spaces  ",
    "tabs\tand\r\nWindows newlines\n\n\n",
    "def f(x):\n    return x ** 2  # comment\n",
    "emoji 👋🏽 family 👨‍👩‍👧 flag 🇮🇳",
    "日本語のテキスト 中文 한국어",
    "नमस्ते दुनिया",
    "naïve café résumé",
    "numbers 1234567890 3.14159 1,000,000",
    "a" * 1000,
    "<eos> <bos> are plain text here",
    "\x00 null and ​ zero-width",
]


def measure(name, encode, decode, documents, total_bytes):
    start = time.perf_counter()
    token_count = sum(len(encode(text)) for text in documents)
    seconds = time.perf_counter() - start
    failures = [text for text in TRICKY if decode(encode(text)) != text]
    print(f"{name:<22}{total_bytes / token_count:>12.2f}{total_bytes / 1e6 / seconds:>12.1f}"
          f"{len(TRICKY) - len(failures):>6}/{len(TRICKY)}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", nargs="+", required=True,
                        help="Held-out text files (one document each) or .jsonl files")
    parser.add_argument("--baseline", default="gpt2", help="tiktoken encoding name (default: gpt2)")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()

    documents = list(read_documents(args.text))
    total_bytes = sum(len(text.encode("utf-8")) for text in documents)
    ours = ByteBPETokenizer.load(args.model)
    print(f"{len(documents)} documents, {total_bytes / 1e6:.1f} MB\n")
    print(f"{'Tokenizer':<22}{'bytes/token':>12}{'MB/s':>12}{'round-trip':>11}")

    failures = measure(f"ours ({ours.vocab_size:,})", ours.encode, ours.decode, documents, total_bytes)
    try:
        import tiktoken
    except ImportError:
        print(f"(tiktoken not installed: skipping the {args.baseline} baseline)")
    else:
        baseline = tiktoken.get_encoding(args.baseline)
        measure(f"tiktoken {args.baseline} ({baseline.n_vocab:,})", baseline.encode_ordinary,
                baseline.decode, documents, total_bytes)

    start = time.perf_counter()
    for _ in ours.encode_many(documents, processes=args.processes):
        pass
    seconds = time.perf_counter() - start
    print(f"\nours, {args.processes} processes: {total_bytes / 1e6 / seconds:.1f} MB/s "
          "(includes starting the workers)")
    for text in failures:
        print(f"Round-trip FAILED for ours: {text!r}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
