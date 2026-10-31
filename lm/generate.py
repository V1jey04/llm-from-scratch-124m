"""Turn next-token logits into text, one token at a time."""

import torch


def next_token_distribution(logits: torch.Tensor, previous: torch.Tensor, temperature: float = 1.0,
                            top_k: int | None = None, top_p: float | None = None,
                            repetition_penalty: float = 1.0) -> torch.Tensor:
    """Reshape one position's logits (vocab,) into sampling probabilities.

    Each knob trades variety for safety:
    - repetition_penalty > 1 makes tokens already in `previous` less likely.
    - temperature < 1 sharpens the distribution, > 1 flattens it.
    - top_k keeps only the k most likely tokens.
    - top_p keeps the smallest set of top tokens whose probability reaches p.
    """
    logits = logits.float().clone()
    if repetition_penalty != 1.0 and previous.numel():
        seen = torch.unique(previous)
        # Dividing a positive logit or multiplying a negative one both lower it.
        logits[seen] = torch.where(logits[seen] > 0, logits[seen] / repetition_penalty,
                                   logits[seen] * repetition_penalty)
    logits = logits / temperature
    if top_k is not None and top_k < logits.numel():
        kth_largest = torch.topk(logits, top_k).values[-1]
        logits[logits < kth_largest] = float("-inf")
    probabilities = logits.softmax(dim=-1)
    if top_p is not None and top_p < 1.0:
        sorted_probabilities, order = probabilities.sort(descending=True)
        # Drop a token when the tokens ranked above it already reach top_p.
        # The first token is never dropped.
        drop = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities >= top_p
        probabilities[order[drop]] = 0.0
        probabilities = probabilities / probabilities.sum()
    return probabilities


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None, repetition_penalty: float = 1.0,
             eos_id: int | None = None, generator: torch.Generator | None = None) -> list[int]:
    """Return only the new token IDs. temperature=0 means greedy (always the top token).

    This recomputes the whole sequence for every new token. That is simple and
    fine for short samples; a key/value cache would avoid the repeated work.
    """
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    new_ids = []
    for _ in range(max_new_tokens):
        window = ids[-model.config.context_length:]  # The model sees at most its context length.
        logits, _ = model(window.unsqueeze(0))
        last = logits[0, -1]
        if temperature == 0:
            token = int(last.argmax())
        else:
            probabilities = next_token_distribution(last, ids, temperature, top_k, top_p, repetition_penalty)
            token = int(torch.multinomial(probabilities.cpu(), 1, generator=generator))
        if token == eos_id:
            break
        new_ids.append(token)
        ids = torch.cat((ids, torch.tensor([token], device=device)))
    return new_ids
