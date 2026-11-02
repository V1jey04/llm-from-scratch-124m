"""Tokenize chat conversations into loss-masked shards for instruction tuning.

    python scripts/prepare_sft.py --tokenizer tokenizers/fineweb_edu_32k.json --out data/sft

The data is SmolTalk's mix for small models (HuggingFaceTB/smol-smoltalk,
Apache-2.0): about 460k conversations written for models under 1B parameters,
from Hugging Face's SmolLM2 work. Validation uses its separate test file.
Each conversation is written in lm/chat_format.py's format, and the top bit of
each token says whether the loss counts it (see lm/data.py).

The mix is reshaped for a 124M chat model (see keep()):
- Coding, text-rewriting, summarizing and long-document conversations are
  dropped. A model this small does them badly, and the first tuning run,
  which kept them, answered "hi" with Python code or "Here's a revised
  version of your text".
- Small talk is added: SmolTalk's everyday-conversations set (2,260 short
  chats such as "Hi there" / "Hello! How can I help you today?") is mixed in
  after every --everyday-every training conversations, cycling through it.
--no-filter restores the first run's data.
"""

import argparse
import hashlib
from itertools import islice
from multiprocessing import Pool
import os
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm import chat_format  # noqa: E402
from lm.data import ShardWriter, mark_trained, write_manifest  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402

DATASET = "HuggingFaceTB/smol-smoltalk"
REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
EVERYDAY_DATASET = "HuggingFaceTB/smoltalk"
EVERYDAY_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
EVERYDAY_FILE = "data/everyday-conversations/train-00000-of-00001.parquet"

DROPPED_SOURCES = {"self-oss-instruct", "smollm-rewrite-30k", "explore-instruct-rewrite", "longalign",
                   "smol-summarize-20k", "smol-summarize-5k"}
EDIT_REQUEST = re.compile(r"\b(edit|edits|editing|rewrite|rewriting|revise|revising|proofread|rephrase|polish|"
                          r"improve (this|my|the following))\b", re.IGNORECASE)

_tokenizer = None


def _load_tokenizer(path):
    global _tokenizer
    _tokenizer = ByteBPETokenizer.load(path)


def _encode(messages):
    ids, trained = chat_format.encode_conversation(_tokenizer, messages)
    return mark_trained(ids, trained), sum(len(m["content"].encode("utf-8")) for m in messages)


def usable(messages) -> bool:
    roles = {m["role"] for m in messages}
    return "assistant" in roles and roles <= {"system", "user", "assistant"}


def keep(messages, source: str) -> bool:
    """Drop code, rewriting and summarizing: sources built for them, any code block, and edit requests."""
    if source in DROPPED_SOURCES or any("```" in m["content"] for m in messages):
        return False
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return not EDIT_REQUEST.search(first_user[:300])


def read_parquet(dataset, name, revision, cache_dir, columns):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(dataset, name, repo_type="dataset", revision=revision, cache_dir=cache_dir)
    parquet = pq.ParquetFile(path)
    for group in range(parquet.num_row_groups):
        yield from parquet.read_row_group(group, columns=columns).to_pylist()


def conversations(files, revision, cache_dir, filtered):
    for name in files:
        for row in read_parquet(DATASET, f"data/{name}.parquet", revision, cache_dir, ["messages", "source"]):
            if usable(row["messages"]) and (not filtered or keep(row["messages"], row["source"])):
                yield row["messages"]


def mix_in(main, extra, every):
    """After every `every` conversations from main, add the next one from extra (cycling)."""
    for count, messages in enumerate(main, 1):
        yield messages
        if extra and count % every == 0:
            yield extra[(count // every - 1) % len(extra)]


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
    parser.add_argument("--train-files", nargs="+",
                        default=["train-00000-of-00004", "train-00001-of-00004", "train-00002-of-00004"])
    parser.add_argument("--val-file", default="test-00000-of-00001")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--cache", default=None, help="Where to keep downloaded parquet files")
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--everyday-every", type=int, default=8,
                        help="Add one small-talk conversation after this many others; 0 adds none")
    parser.add_argument("--no-filter", action="store_true", help="Keep every source, as in the first run")
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")
    tokenizer_sha = hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest()
    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    filtered = not args.no_filter
    everyday = []
    if args.everyday_every:
        everyday = [row["messages"] for row in read_parquet(EVERYDAY_DATASET, EVERYDAY_FILE, EVERYDAY_REVISION,
                                                            args.cache, ["messages"]) if usable(row["messages"])]

    splits = {}
    with Pool(args.processes, initializer=_load_tokenizer, initargs=(args.tokenizer,)) as pool:
        for split, files, max_tokens in (("val", [args.val_file], args.val_tokens),
                                         ("train", args.train_files, args.train_tokens)):
            source = conversations(files, args.revision, args.cache, filtered)
            if split == "train":
                source = mix_in(source, everyday, args.everyday_every)
            writer = ShardWriter(out, split, args.shard_tokens)
            info = splits[split] = write_split(pool, args.processes, source, writer, max_tokens)
            print(f"{split}: {info['documents']:,} conversations, {info['tokens']:,} tokens")
            if info["tokens"] < max_tokens:
                print(f"WARNING: {split} ran out of conversations at {info['tokens']:,} tokens.")
    write_manifest(out, {"file": "tokenizer.json", "sha256": tokenizer_sha, "vocab_size": tokenizer.vocab_size,
                         "eos_id": tokenizer.EOS_ID}, splits,
                   {"type": "huggingface", "dataset": DATASET, "revision": args.revision,
                    "train_files": args.train_files, "val_file": args.val_file, "filtered": filtered,
                    "everyday": {"dataset": EVERYDAY_DATASET, "revision": EVERYDAY_REVISION, "file": EVERYDAY_FILE,
                                 "every": args.everyday_every, "conversations": len(everyday)}},
                   # lm.train reads loss_mask; the trained model keeps chat_format for lm.chat.
                   {"loss_mask": "top_bit", "chat_format": chat_format.FORMAT})


if __name__ == "__main__":
    main()
