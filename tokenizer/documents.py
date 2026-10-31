"""Read training or evaluation documents from files."""

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_documents(paths: Iterable[str | Path]) -> Iterator[str]:
    """Yield one document at a time.

    A `.jsonl` file holds one document per line, in its "text" field. Any other
    file is a single UTF-8 document.
    """
    for path in map(Path, paths):
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        yield json.loads(line)["text"]
        else:
            yield path.read_text(encoding="utf-8")
