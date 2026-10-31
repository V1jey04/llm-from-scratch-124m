"""Train a ~1M-parameter model for a few minutes to check the whole path end to end.

    .venv/bin/python scripts/train_tiny.py --input data/fineweb_edu/train.jsonl --max-mb 5

Text -> a small tokenizer -> token IDs -> model -> loss going down -> samples.
This is a smoke test for Phase 2, not the real training loop: no validation
set, checkpoints or mixed precision yet (those come in Phase 3).
"""

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm import GPT, PRESETS, generate  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402
from tokenizer.documents import read_documents  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", nargs="+", default=["data/fineweb_edu/train.jsonl"])
    parser.add_argument("--max-mb", type=float, default=5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--prompt", default="The water cycle")
    parser.add_argument("--output", default="runs/tiny/model.pt")
    args = parser.parse_args()

    config = PRESETS["tiny"]
    documents, size = [], 0
    for text in read_documents(args.input):
        documents.append(text)
        size += len(text.encode("utf-8"))
        if size >= args.max_mb * 1e6:
            break
    tokenizer = ByteBPETokenizer.train(documents, config.vocab_size)
    ids = torch.tensor([i for text in documents for i in tokenizer.encode(text, add_eos=True)])
    print(f"{size / 1e6:.1f} MB -> {len(ids):,} tokens with a {tokenizer.vocab_size}-ID tokenizer")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = GPT(config).to(device)
    print(f"{model.num_parameters():,} parameters on {device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    T = config.context_length
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        # Random windows; targets are the same windows shifted one token left.
        starts = torch.randint(0, len(ids) - T - 1, (args.batch_size,))
        batch = torch.stack([ids[s:s + T + 1] for s in starts]).to(device)
        _, loss = model(batch[:, :-1], batch[:, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 250 == 0:
            print(f"step {step:>5}  loss {loss.item():.3f}  ({time.perf_counter() - start:.0f} s)")

    model.save(args.output)
    tokenizer.save(Path(args.output).with_name("tokenizer.json"))
    prompt = tokenizer.encode(args.prompt)
    for label, settings in [("greedy", {"temperature": 0}),
                            ("sampled", {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.1})]:
        new = generate(model, prompt, 60, eos_id=tokenizer.EOS_ID,
                       generator=torch.Generator().manual_seed(0), **settings)
        print(f"\n[{label}] {tokenizer.decode(prompt + new, errors='replace')}")


if __name__ == "__main__":
    main()
