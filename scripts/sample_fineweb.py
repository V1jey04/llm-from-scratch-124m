"""Download a small, reproducible FineWeb-Edu sample for tokenizer work.

    .venv/bin/python scripts/sample_fineweb.py --train-mb 300 --heldout-mb 20

Reads only the needed row groups (1,000 documents each) over HTTP, spread
evenly through one parquet file, instead of downloading whole 2 GB files.
Training and held-out documents come from different files, so they never
overlap. Writes JSONL ({"id", "text"} per line) plus a manifest recording the
exact dataset revision and row groups. Needs `pyarrow` and `huggingface_hub`.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample/10BT"
TRAIN_FILE = "000_00000.parquet"
HELDOUT_FILE = "013_00000.parquet"
TEXT_BYTES_PER_ROW_GROUP = 5_000_000  # Rough guess, used only to space the reads.


def sample_file(fs, path, target_bytes, output):
    parquet = pq.ParquetFile(fs.open(path))
    wanted_groups = max(1, target_bytes // TEXT_BYTES_PER_ROW_GROUP + 1)
    step = max(1, parquet.num_row_groups // wanted_groups)
    used_groups, documents, text_bytes = [], 0, 0
    with open(output, "w", encoding="utf-8") as file:
        for group in range(0, parquet.num_row_groups, step):
            table = parquet.read_row_group(group, columns=["id", "text"])
            for row_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
                file.write(json.dumps({"id": row_id, "text": text}, ensure_ascii=False) + "\n")
                documents += 1
                text_bytes += len(text.encode("utf-8"))
            used_groups.append(group)
            print(f"  {output.name}: {text_bytes / 1e6:.0f} / {target_bytes / 1e6:.0f} MB", end="\r")
            if text_bytes >= target_bytes:
                break
    print()
    return {"file": path, "row_groups": used_groups, "documents": documents, "text_bytes": text_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-mb", type=float, default=300)
    parser.add_argument("--heldout-mb", type=float, default=20)
    parser.add_argument("--out", default="data/fineweb_edu")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info(DATASET).sha
    fs = HfFileSystem()
    base = f"datasets/{DATASET}@{revision}/{SUBSET}"
    manifest = {
        "dataset": DATASET,
        "revision": revision,
        "subset": SUBSET,
        "train": sample_file(fs, f"{base}/{TRAIN_FILE}", int(args.train_mb * 1e6), out / "train.jsonl"),
        "heldout": sample_file(fs, f"{base}/{HELDOUT_FILE}", int(args.heldout_mb * 1e6), out / "heldout.jsonl"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for split in ("train", "heldout"):
        print(f"{split}: {manifest[split]['documents']:,} documents, {manifest[split]['text_bytes'] / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
