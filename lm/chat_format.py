"""How conversations become token IDs for instruction tuning and chat.

A conversation is plain text with role labels, and each assistant reply ends
with <eos>:

    <bos>System: Be brief.

    User: What is 2 + 2?

    Assistant: 4.<eos>User: And 3 + 3?

    Assistant: 6.<eos>

Training only scores the model on what the assistant says (its words and the
<eos> that ends them); the user's and system's text is context. Chat prompts
end with "Assistant:" so the model writes the next reply and stops at <eos>.
"""

FORMAT = "plain-v1"
PREFIXES = {"system": "System: ", "user": "User: "}
ASSISTANT = "Assistant:"


def encode_conversation(tokenizer, messages: list[dict]) -> tuple[list[int], list[bool]]:
    """Return (ids, trained): trained[i] says whether ids[i] is assistant output.

    The reply is encoded as " " + text so its first word gets the same leading
    space token as in "Assistant: Hello", which is what generation produces.
    """
    ids, trained = [tokenizer.BOS_ID], [False]
    for message in messages:
        role, text = message["role"], message["content"].strip()
        if role == "assistant":
            prefix = tokenizer.encode(ASSISTANT)
            reply = tokenizer.encode(" " + text) + [tokenizer.EOS_ID]
            ids += prefix + reply
            trained += [False] * len(prefix) + [True] * len(reply)
        elif role in PREFIXES:
            part = tokenizer.encode(f"{PREFIXES[role]}{text}\n\n")
            ids += part
            trained += [False] * len(part)
        else:
            raise ValueError(f"Unknown role: {role}")
    return ids, trained


def prompt_ids(tokenizer, messages: list[dict]) -> list[int]:
    """IDs for a conversation so far, ending where the assistant's next reply begins."""
    if messages and messages[-1]["role"] == "assistant":
        raise ValueError("The last message should be the user's.")
    ids, _ = encode_conversation(tokenizer, messages)
    return ids + tokenizer.encode(ASSISTANT)
