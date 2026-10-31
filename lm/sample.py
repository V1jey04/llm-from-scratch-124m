"""Generate text from a saved model.

    python -m lm.sample --model runs/smoke/final_model.pt --tokenizer tokenizers/fineweb_edu_32k.json \\
        --prompt "The water cycle is" "Photosynthesis is"

Each prompt gets one greedy continuation and one sampled continuation.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm.generate import generate  # noqa: E402
from lm.model import GPT  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    model = GPT.load(args.model, device=args.device)
    print(f"{model.num_parameters():,} parameters, {tokenizer.vocab_size:,}-ID tokenizer\n")
    for prompt in args.prompt:
        ids = tokenizer.encode(prompt)
        for label, settings in [("greedy", {"temperature": 0}),
                                ("sampled", {"temperature": args.temperature, "top_p": args.top_p,
                                             "repetition_penalty": args.repetition_penalty})]:
            new = generate(model, ids, args.max_new_tokens, eos_id=tokenizer.EOS_ID,
                           generator=torch.Generator().manual_seed(args.seed), **settings)
            text = tokenizer.decode(ids + new, errors="replace")
            print(f"[{label}] {text}\n")
        print("-" * 72)


if __name__ == "__main__":
    main()
