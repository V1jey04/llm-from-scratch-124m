"""Talk to a trained model in the terminal.

    .venv/bin/python -m lm.chat --model models/sft_final_model.pt

An instruction-tuned model (one whose file records a chat format) gets real
turns: the conversation so far is sent in lm/chat_format.py's format, and the
model answers and stops on its own. /reset starts a new conversation.

A base model has only learned to continue text. Chat mode then wraps each
message as "Question: ...\\nAnswer:" and stops at a blank line, the closest a
model without instruction tuning gets to answering, and each question stands
alone. Text mode simply continues whatever you type.

Commands: /chat, /text, /reset, /temp 0.7, /tokens 150, /quit
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lm import chat_format  # noqa: E402
from lm.generate import generate  # noqa: E402
from lm.model import GPT  # noqa: E402
from tokenizer import ByteBPETokenizer  # noqa: E402

BASE_STOPS = ("\n\n", "Question:", "Answer:")
INSTRUCT_STOPS = ("User:", "System:")  # The model normally ends with <eos>; these are a backstop.


def respond(model, tokenizer, prompt_ids: list[int], stops: tuple[str, ...], temperature: float,
            max_tokens: int) -> str:
    """Print the model's continuation as it is generated and return it."""
    printed = 0
    hold_back = max(map(len, stops), default=0)

    def text_of(new_ids):
        # "ignore" drops a half-finished UTF-8 character until its last byte arrives.
        text = tokenizer.decode(new_ids, errors="ignore")
        if stops:
            text = text.lstrip()
            cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
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
    text = text.rstrip() if stops else text
    print(text[printed:], end="\n\n")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="tokenizers/fineweb_edu_32k.json")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=None, help="Default: 300 tuned, 120 base")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = ByteBPETokenizer.load(args.tokenizer)
    model = GPT.load(args.model, device=args.device)
    instruct = model.metadata.get("chat_format") == chat_format.FORMAT
    mode, temperature = "chat", args.temperature
    max_tokens = args.max_tokens or (300 if instruct else 120)
    history = []
    kind = "instruction-tuned" if instruct else "base model"
    print(f"Loaded {model.num_parameters():,} parameters ({kind}) on {args.device}. "
          "Commands: /chat, /text, /reset, /temp N, /tokens N, /quit\n")
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
        if command == "/reset":
            history = []
            print("New conversation.\n")
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
        if mode == "text":
            print(f"model> {message}", end="", flush=True)
            respond(model, tokenizer, tokenizer.encode(message), (), temperature, max_tokens)
        elif instruct:
            print("model> ", end="", flush=True)
            history.append({"role": "user", "content": message})
            reply = respond(model, tokenizer, chat_format.prompt_ids(tokenizer, history), INSTRUCT_STOPS,
                            temperature, max_tokens)
            history.append({"role": "assistant", "content": reply})
        else:
            print("model> ", end="", flush=True)
            respond(model, tokenizer, tokenizer.encode(f"Question: {message}\nAnswer:"), BASE_STOPS,
                    temperature, max_tokens)


if __name__ == "__main__":
    main()
