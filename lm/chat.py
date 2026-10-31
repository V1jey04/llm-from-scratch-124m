"""Talk to a trained model in the terminal.

    .venv/bin/python -m lm.chat --model models/pilot_final_model.pt

This is a base model: it has only learned to continue text. Chat mode wraps
each message as "Question: ...\\nAnswer:" and stops at a blank line, which is
the closest a model without instruction tuning gets to answering. Each
question is answered on its own; the model does not remember earlier turns.
Text mode simply continues whatever you type.

Commands: /chat, /text, /temp 0.7, /tokens 150, /quit
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm.generate import generate  # noqa: E402
from lm.model import GPT  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402

STOPS = ("\n\n", "Question:", "Answer:")


def respond(model, tokenizer, prompt: str, mode: str, temperature: float, max_tokens: int) -> None:
    """Print the model's continuation as it is generated."""
    prompt_ids = tokenizer.encode(prompt)
    printed = 0
    hold_back = max(map(len, STOPS)) if mode == "chat" else 0

    def text_of(new_ids):
        # "ignore" drops a half-finished UTF-8 character until its last byte arrives.
        text = tokenizer.decode(new_ids, errors="ignore")
        if mode == "chat":
            text = text.lstrip()
            cut = min((i for i in (text.find(s) for s in STOPS) if i >= 0), default=-1)
            return (text[:cut], True) if cut >= 0 else (text, False)
        return text, False

    def on_token(new_ids):
        nonlocal printed
        text, stop = text_of(new_ids)
        # Hold back the last few characters: they may be the start of a stop string.
        safe = len(text) if stop else max(printed, len(text) - hold_back)
        print(text[printed:safe], end="", flush=True)
        printed = safe
        return stop

    new_ids = generate(model, prompt_ids, max_tokens, temperature=temperature, top_p=0.95,
                       repetition_penalty=1.1, eos_id=tokenizer.EOS_ID, on_token=on_token)
    text, _ = text_of(new_ids)
    print(text[printed:].rstrip() if mode == "chat" else text[printed:], end="\n\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="tokenizers/fineweb_edu_32k.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    model = GPT.load(args.model, device=args.device)
    mode, temperature, max_tokens = "chat", args.temperature, args.max_tokens
    print(f"Loaded {model.num_parameters():,} parameters on {args.device}. "
          "Commands: /chat, /text, /temp N, /tokens N, /quit\n")
    while True:
        try:
            message = input(f"[{mode}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        command, _, value = message.partition(" ")
        if command == "/quit":
            break
        if command in ("/chat", "/text"):
            mode = command[1:]
            continue
        if command in ("/temp", "/tokens"):
            try:
                if command == "/temp":
                    temperature = float(value)
                else:
                    max_tokens = int(value)
            except ValueError:
                print("Give a number, for example /temp 0.7\n")
            continue
        if mode == "chat":
            print("model> ", end="", flush=True)
            respond(model, tokenizer, f"Question: {message}\nAnswer:", mode, temperature, max_tokens)
        else:
            print(f"model> {message}", end="", flush=True)
            respond(model, tokenizer, message, mode, temperature, max_tokens)


if __name__ == "__main__":
    main()
