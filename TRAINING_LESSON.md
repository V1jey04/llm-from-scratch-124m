# Your first training pipeline: text → shards → steps → checkpoints

The model can predict; now it has to learn from 2 billion tokens on GPUs that
disappear every 12 hours. Read `lm/data.py`, then `lm/train.py`, alongside this
lesson.

## Step 1: Tokenize once, read many times

Encoding text is slow compared with training, so `scripts/prepare_data.py`
does it once and saves plain arrays of token IDs. Each ID fits in 2 bytes
(`uint16`, values up to 65,535) because the vocabulary has 32,768 IDs, so
2B tokens take 4 GB. Documents are joined with `<eos>`, which teaches the
model where texts end.

Two details matter more than they look:

- **Split by document, before tokenizing.** Validation documents come from a
  different file. If one document's first half were in training and its second
  half in validation, validation would reward memorizing, not learning.
- **Count bytes exactly.** The manifest stores how many text bytes the tokens
  came from, to report bits per byte. Worker processes read documents ahead of
  what is written, so each document's size waits in a queue until its tokens
  come back.

`TokenShards` opens the files with `numpy.memmap`: the operating system loads
only the parts actually read, so a 4 GB dataset needs almost no memory.

## Step 2: Choosing what to train on

`WindowSampler` cuts the token stream into windows of 1,025 tokens: 1,024
inputs and, shifted by one, 1,024 targets. Each epoch visits every window once
in a shuffled order.

- **Shuffled** so a batch mixes many documents rather than 32 slices of one.
- **Without repeats** so "trained on 2B tokens" means 2B *different* tokens.
- **Reproducible**: the order comes from `(seed, epoch)` alone, so a checkpoint
  only needs `epoch` and `position` to continue exactly where it stopped.

With two GPUs, both processes compute the same shuffled order and each takes
half of every batch. Together they see exactly what one process would.
`test_processes_split_the_single_process_batch` checks this.

## Step 3: One optimizer step

```text
for each of grad_accum_steps micro-batches:
    forward (in fp16)  → loss / grad_accum_steps
    scaled backward    → gradients add up
unscale, clip the gradient norm to 1.0
optimizer step, update the loss scale, zero the gradients
```

- **Gradient accumulation:** a T4 fits 8 windows at a time, but the run wants
  262,144 tokens per step. 16 micro-batches × 8 windows × 2 GPUs × 1,024 tokens
  gives that. Dividing each loss by 16 makes the summed gradients equal the
  average over the whole batch. `test_gradient_accumulation_matches_one_big_batch`
  proves it.
- **DDP:** each GPU holds a full copy of the model. After the backward pass,
  gradients are averaged across GPUs, so both copies take the same step. Inside
  accumulation, `no_sync()` skips that averaging until the last micro-batch.
- **fp16 and the loss scale:** fp16 has few bits, and tiny gradients round to
  zero. `GradScaler` multiplies the loss by a large number (65,536 at first)
  before backward, divides the gradients back before the step, and skips steps
  that overflowed, halving the scale. Watch `scale` in the logs: if it keeps
  falling, training is unstable.
- **Gradient clipping:** if the gradients' combined length exceeds 1.0, they
  are scaled down. One bad batch can't throw the model far.

## Step 4: The learning-rate schedule

`learning_rate()` rises linearly for the first 2% of steps (warmup: early
gradients are large and unreliable), then follows a cosine curve down to 10%
of the peak. Small steps at the end let the model settle.

Weight decay (0.1) pulls weights toward zero to discourage memorizing. It
applies to the weight matrices only: norm weights are scales around 1, and
decaying the embedding would shrink rarely-seen tokens for no reason.

## Step 5: Surviving a 12-hour session

A checkpoint holds everything needed to continue as if nothing happened:
model weights, AdamW's running averages, the loss scale, the sampler's
position, the step and token counts.

- **Atomic saves:** write `checkpoint.tmp`, then rename it. A crash during
  saving never leaves a half-written checkpoint.
- **Safety checks on resume:** a different model shape, tokenizer or batch size
  is refused. Any of those would silently corrupt the run.
- **`time_limit_hours`** saves and stops before Kaggle ends the session.
- **Rank 0 decides** when to save or stop and tells the other process, so they
  never disagree about which step is the last.

`test_stopping_and_resuming_gives_exactly_the_same_model` runs 6 steps, then
3 + 3 with a restart in between, and requires bit-identical weights.

## Step 6: Watching it learn

```sh
.venv/bin/python -m lm.train --config configs/smoke.json --set stop_after_steps=150
.venv/bin/python -m lm.train --config configs/smoke.json     # resumes at step 150
```

An 11.6M model on 4M tokens, about 6 minutes on the M4's GPU:

| Step | Tokens | Validation loss | Bits per byte |
|---:|---:|---:|---:|
| 100 | 0.8M | 6.74 | 2.10 |
| 200 | 1.6M | 6.27 | 1.96 |
| 400 | 3.3M | 5.88 | 1.84 |
| 488 | 4.0M | 5.82 | 1.82 |

Bits per byte converts loss into "how many bits to encode each byte of text".
Unlike loss, it can be compared across tokenizers, which the GPT-2 comparison
needs. For reference, plain gzip manages roughly 2–3 bits per byte on English.

## Check yourself

1. Why must validation documents be chosen before tokenizing?
2. Why divide the loss by `grad_accum_steps`?
3. What does `GradScaler` protect against, and what does a falling `scale` mean?
4. What would go wrong on resume if the sampler's position were not saved?
5. Why does only rank 0 decide when to stop?
