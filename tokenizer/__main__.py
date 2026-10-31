"""Run with python3 -m tokenizer demo, train, encode, decode, or inspect."""

import argparse
import json
from pathlib import Path
import time

from .bpe import GPT2_SPLIT_PATTERN, ByteBPETokenizer, pre_split
from .documents import read_documents


def demo():
    documents = ["banana banana banana", "bandana banana", "Hello hello hello!"]
    tokenizer = ByteBPETokenizer.train(documents, vocab_size=275)
    print("Step 1: Text becomes UTF-8 bytes")
    print("banana ->", list("banana".encode("utf-8")))
    print("\nStep 1b: Pre-split text into chunks; merges stay inside a chunk")
    print("Hello hello hello! ->", pre_split("Hello hello hello!", tokenizer.pattern))
    print("\nStep 2: Learn frequent pairs (first eight merges)")
    for step in tokenizer.training_history[:8]:
        left, right = step["pair"]
        print(f"{tokenizer.vocab[left]!r} + {tokenizer.vocab[right]!r} "
              f"-> ID {step['new_id']}: {tokenizer.vocab[step['new_id']]!r} "
              f"(pair count {step['count']})")
    print("\nStep 3: Encode and decode unseen text")
    for text in ["banana banana", "Hello 👋\nनमस्ते", "<eos>"]:
        ids = tokenizer.encode(text)
        restored = tokenizer.decode(ids)
        assert restored == text
        print(f"{text!r}\n  {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens: {ids}")
        print(f"  Decoded: {restored!r}; exact match: {restored == text}")
    path = Path("models/demo_tokenizer.json")
    tokenizer.save(path)
    print(f"\nStep 4: Saved {tokenizer.vocab_size} total IDs to {path}")


def inspect_text(tokenizer, text):
    ids = tokenizer.encode(text)
    byte_count = len(text.encode("utf-8"))
    print(f"Text: {text!r}")
    print(f"Vocabulary: {tokenizer.vocab_size} IDs, including 3 special IDs")
    if tokenizer.pattern is None:
        print("Pre-split: off (whole text is one chunk)")
    else:
        print(f"Pre-split chunks: {pre_split(text, tokenizer.pattern)}")
    print(f"{byte_count} UTF-8 bytes -> {len(ids)} tokens")
    print("\nIndex   ID      Kind      Bytes represented")
    for index, token_id in enumerate(ids):
        kind = "byte" if token_id < 256 else "learned"
        print(f"{index:<8}{token_id:<8}{kind:<10}{tokenizer.vocab[token_id]!r}")
    print(f"\nDecoded: {tokenizer.decode(ids)!r}")
    print(f"Exact match: {tokenizer.decode(ids) == text}")


def main():
    parser = argparse.ArgumentParser(description="Your first byte-level BPE tokenizer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Show the steps on a tiny corpus")
    train = commands.add_parser("train", help="Learn merges from text files (one document each) or .jsonl files")
    train.add_argument("--input", nargs="+", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--vocab-size", type=int, default=280)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--no-pre-split", action="store_true",
                       help="Merge across whole documents instead of GPT-2-style chunks")
    encode = commands.add_parser("encode")
    encode.add_argument("--model", required=True)
    encode.add_argument("text")
    decode = commands.add_parser("decode")
    decode.add_argument("--model", required=True)
    decode.add_argument("ids", nargs="*", type=int)
    inspect = commands.add_parser("inspect", help="Show the bytes represented by each token")
    inspect.add_argument("--model", required=True)
    inspect.add_argument("text")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "train":
        start = time.perf_counter()
        texts = read_documents(args.input)
        pattern = None if args.no_pre_split else GPT2_SPLIT_PATTERN
        tokenizer = ByteBPETokenizer.train(texts, args.vocab_size, args.min_frequency, pattern)
        tokenizer.save(args.output)
        print(f"Saved {tokenizer.vocab_size} IDs ({len(tokenizer.merges)} merges) to {args.output} "
              f"in {time.perf_counter() - start:.1f} s")
    elif args.command == "encode":
        print(json.dumps(ByteBPETokenizer.load(args.model).encode(args.text)))
    elif args.command == "decode":
        print(ByteBPETokenizer.load(args.model).decode(args.ids))
    elif args.command == "inspect":
        inspect_text(ByteBPETokenizer.load(args.model), args.text)


if __name__ == "__main__":
    main()
