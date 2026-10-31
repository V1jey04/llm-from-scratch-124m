"""Tokenize documents into uint16 shards for training.

From FineWeb-Edu on Hugging Face (what the Kaggle CPU notebook runs):
    python scripts/prepare_data.py --tokenizer tokenizers/fineweb_edu_32k.json --out data/tokens \\
        --hf-train-files 000_00000 001_00000 002_00000 003_00000 --hf-val-file 013_00000 \\
        --train-tokens 2_100_000_000 --val-tokens 10_000_000

From local JSONL files (for testing the pipeline on a Mac):
    python scripts/prepare_data.py --tokenizer models/fineweb_edu_32k.json --out data/tokens_local \\
        --train-jsonl data/fineweb_edu/train.jsonl --val-jsonl data/fineweb_edu/heldout.jsonl

Training and validation documents come from different files, so the split is
by document, decided before any tokenization. File 013 is held out everywhere:
the tokenizer benchmark and validation both use it, and training never does.
"""

import argparse
from collections import deque
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm.data import ShardWriter, write_manifest  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402
from tokenizer.documents import read_documents  # noqa: E402

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"


def hf_documents(files, revision, cache_dir):
    """Download whole parquet files (about 2.15 GB, 726k documents each) and read them."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    for name in files:
        path = hf_hub_download(DATASET, f"{SUBSET}/{name}.parquet", repo_type="dataset",
                               revision=revision, cache_dir=cache_dir)
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            yield from parquet.read_row_group(group, columns=["text"]).column("text").to_pylist()


def tokenize_split(tokenizer, documents, writer, max_tokens, processes):
    """Encode documents in parallel and write them, each followed by <eos>.

    Workers read ahead of what is written, so each document's byte count waits
    in a queue until its tokens come back (results arrive in order). The totals
    then describe exactly the documents written, which bits-per-byte relies on.
    """
    pending_bytes = deque()

    def remember_size(texts):
        for text in texts:
            pending_bytes.append(len(text.encode("utf-8")))
            yield text

    documents_written, text_bytes = 0, 0
    start = time.perf_counter()
    for ids in tokenizer.encode_many(remember_size(documents), add_eos=True, processes=processes):
        writer.add(ids)
        documents_written += 1
        text_bytes += pending_bytes.popleft()
        if documents_written % 20_000 == 0:
            rate = text_bytes / 1e6 / (time.perf_counter() - start)
            print(f"  {writer.split}: {writer.tokens / 1e6:,.0f}M tokens ({rate:.0f} MB/s)", flush=True)
        if writer.tokens >= max_tokens:
            break
    return {"shards": writer.close(), "documents": documents_written, "tokens": writer.tokens,
            "text_bytes": text_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-jsonl", nargs="+")
    parser.add_argument("--val-jsonl", nargs="+")
    parser.add_argument("--hf-train-files", nargs="+", help="Parquet names in sample/10BT, e.g. 000_00000")
    parser.add_argument("--hf-val-file", default="013_00000")
    parser.add_argument("--hf-revision", default="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
    parser.add_argument("--hf-cache", default=None, help="Where to keep downloaded parquet files")
    parser.add_argument("--train-tokens", type=int, default=10**12)
    parser.add_argument("--val-tokens", type=int, default=10_000_000)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if bool(args.train_jsonl) == bool(args.hf_train_files):
        parser.error("Give exactly one of --train-jsonl or --hf-train-files.")
    if args.train_jsonl and not args.val_jsonl:
        parser.error("--train-jsonl needs --val-jsonl.")
    if args.hf_train_files and args.hf_val_file in args.hf_train_files:
        parser.error("The validation file must not also be a training file.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    if tokenizer.vocab_size > 65_536:
        raise ValueError("uint16 shards hold at most 65,536 token IDs.")
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")  # Data and tokenizer travel together.
    tokenizer_sha = hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()

    if args.train_jsonl:
        sources = {"train": read_documents(args.train_jsonl), "val": read_documents(args.val_jsonl)}
        source = {"type": "jsonl", "train": args.train_jsonl, "val": args.val_jsonl}
    else:
        sources = {"train": hf_documents(args.hf_train_files, args.hf_revision, args.hf_cache),
                   "val": hf_documents([args.hf_val_file], args.hf_revision, args.hf_cache)}
        source = {"type": "huggingface", "dataset": DATASET, "subset": SUBSET, "revision": args.hf_revision,
                  "train_files": args.hf_train_files, "val_file": args.hf_val_file}

    splits = {}
    for split, max_tokens in (("val", args.val_tokens), ("train", args.train_tokens)):
        writer = ShardWriter(out, split, args.shard_tokens)
        splits[split] = tokenize_split(tokenizer, sources[split], writer, max_tokens, args.processes)
        info = splits[split]
        print(f"{split}: {info['documents']:,} documents, {info['tokens']:,} tokens, "
              f"{info['text_bytes'] / info['tokens']:.2f} bytes/token, {len(info['shards'])} shards")
        if info["tokens"] < max_tokens and max_tokens < 10**12:
            print(f"WARNING: {split} ran out of documents at {info['tokens']:,} of the "
                  f"{max_tokens:,} tokens asked for; add more source files.")
    write_manifest(out, {"file": "tokenizer.json", "sha256": tokenizer_sha, "vocab_size": tokenizer.vocab_size,
                         "eos_id": tokenizer.EOS_ID}, splits, source)


if __name__ == "__main__":
    main()
