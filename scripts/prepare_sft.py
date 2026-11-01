"""Tokenize chat conversations into loss-masked shards for instruction tuning.

    python scripts/prepare_sft.py --tokenizer tokenizers/fineweb_edu_32k.json --out data/sft \\
        --train-files train-00000-of-00004 --train-tokens 100_000_000

The data is SmolTalk's mix for small models (HuggingFaceTB/smol-smoltalk,
Apache-2.0): about 460k conversations written for models under 1B parameters,
from Hugging Face's SmolLM2 work. Validation uses its separate test file.
Each conversation is written in lm/chat_format.py's format, and the top bit of
each token says whether the loss counts it (see lm/data.py).
"""

import argparse
import hashlib
from itertools import islice
from multiprocessing import Pool
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm import chat_format  # noqa: E402
from lm.data import ShardWriter, mark_trained, write_manifest  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402

DATASET = "HuggingFaceTB/smol-smoltalk"
REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"

_tokenizer = None


def _load_tokenizer(path):
    global _tokenizer
    _tokenizer = ByteBPETokenizer.load(path)


def _encode(messages):
    ids, trained = chat_format.encode_conversation(_tokenizer, messages)
    return mark_trained(ids, trained), sum(len(m["content"].encode("utf-8")) for m in messages)


def conversations(files, revision, cache_dir):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    for name in files:
        path = hf_hub_download(DATASET, f"data/{name}.parquet", repo_type="dataset", revision=revision,
                               cache_dir=cache_dir)
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            for messages in parquet.read_row_group(group, columns=["messages"]).column("messages").to_pylist():
                roles = {m["role"] for m in messages}
                if "assistant" in roles and roles <= {"system", "user", "assistant"}:
                    yield messages


def write_split(pool, processes, source, writer, max_tokens):
    """Encode in blocks: Pool.imap would read the whole source ahead, downloading every file."""
    count, text_bytes = 0, 0
    block_size = 64 * processes
    while writer.tokens < max_tokens:
        block = list(islice(source, block_size))
        if not block:
            break
        for ids, size in pool.map(_encode, block, chunksize=16):
            writer.add(ids)
            count, text_bytes = count + 1, text_bytes + size
            if count % 20_000 == 0:
                print(f"  {writer.split}: {count:,} conversations, {writer.tokens / 1e6:,.0f}M tokens", flush=True)
            if writer.tokens >= max_tokens:
                break
    return {"shards": writer.close(), "documents": count, "tokens": writer.tokens, "text_bytes": text_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-files", nargs="+", default=["train-00000-of-00004"])
    parser.add_argument("--val-file", default="test-00000-of-00001")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--cache", default=None, help="Where to keep downloaded parquet files")
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")
    tokenizer_sha = hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()
    tokenizer = ByteBPETokenizer.load(args.tokenizer)

    splits = {}
    with Pool(args.processes, initializer=_load_tokenizer, initargs=(args.tokenizer,)) as pool:
        for split, files, max_tokens in (("val", [args.val_file], args.val_tokens),
                                         ("train", args.train_files, args.train_tokens)):
            writer = ShardWriter(out, split, args.shard_tokens)
            info = splits[split] = write_split(pool, args.processes, conversations(files, args.revision, args.cache),
                                               writer, max_tokens)
            print(f"{split}: {info['documents']:,} conversations, {info['tokens']:,} tokens")
            if info["tokens"] < max_tokens:
                print(f"WARNING: {split} ran out of conversations at {info['tokens']:,} tokens.")
    write_manifest(out, {"file": "tokenizer.json", "sha256": tokenizer_sha, "vocab_size": tokenizer.vocab_size,
                         "eos_id": tokenizer.EOS_ID}, splits,
                   {"type": "huggingface", "dataset": DATASET, "revision": args.revision,
                    "train_files": args.train_files, "val_file": args.val_file},
                   # lm.train reads loss_mask; the trained model keeps chat_format for lm.chat.
                   {"loss_mask": "top_bit", "chat_format": chat_format.FORMAT})


if __name__ == "__main__":
    main()
