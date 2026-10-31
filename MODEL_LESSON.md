# Your first Transformer: token IDs → next-token predictions

The tokenizer turns text into IDs. The model reads a sequence of IDs and, at
every position, outputs a score (a *logit*) for each possible next token.
Training nudges the weights so the real next token gets a higher score.

Read `lm/model.py` from top to bottom alongside this lesson. Shapes use
`B` = batch, `T` = sequence length, `D` = width (`d_model`), `H` = heads and
`Dh = D / H`. Set up first:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest tests.test_model -v
```

## Step 1: The path of one batch

```text
ids (B, T)
 → embedding            (B, T, D)    look up one learned vector per token
 → 12 × Block           (B, T, D)    attention, then feed-forward, each added back
 → final RMSNorm        (B, T, D)
 → × embeddingᵀ         (B, T, V)    a score for every vocabulary entry
 → cross entropy with the next tokens → one number: the loss
```

For training, the targets are the inputs shifted one place: from
`The cat sat on`, the model predicts `cat` after `The`, `sat` after `The cat`,
and so on, in one pass.

**Try:** in a Python shell, build `GPT(PRESETS["tiny"])`, pass
`torch.zeros(2, 5, dtype=torch.long)`, and print the logits' shape.

## Step 2: Attention, written out

Look at `explicit_attention`. Each position makes a query `q` ("what am I
looking for?"), a key `k` ("what do I contain?") and a value `v` ("what I pass
on"). The score `q·k / √Dh` says how much one position should read from
another; softmax turns scores into weights that sum to 1. Dividing by `√Dh`
keeps scores from growing with the head size, which would make softmax pick
a single winner.

The **causal mask** sets scores for later positions to `-inf`, so they get
weight 0: while predicting token 5, the model must not see token 5.
`test_future_tokens_cannot_change_earlier_logits` checks exactly that.

The model itself calls PyTorch's fused `scaled_dot_product_attention`, which
is faster and uses less memory. `test_fused_attention_matches_explicit` checks
that it computes the same thing as your version.

With `H` heads, the width is split into `H` independent attentions of size
`Dh`, then joined back. Different heads can learn different relationships.

## Step 3: RoPE, positions by rotation

Attention by itself does not know word order. GPT-2 added a learned vector per
position; we rotate instead. `apply_rope` treats each head vector as `Dh/2`
pairs of numbers and turns each pair by `position × frequency`. After rotating
`q` at position `m` and `k` at position `n`, their dot product depends only on
`m − n`: the model sees *how far apart* tokens are, not where they are.
`test_rope_keeps_length_and_depends_only_on_relative_position` checks this
directly: positions (5, 2) and (25, 22) give the same score.

## Step 4: The rest of a block

- **RMSNorm** rescales each vector to root-mean-square 1, then multiplies by a
  learned weight. It is done in float32, because squaring fp16 numbers on the
  Kaggle GPUs can overflow.
- **SwiGLU** is the feed-forward part: `w2(silu(w1 x) ⊙ w3 x)`. One branch acts
  as a gate on the other. It has three matrices, so the hidden size is 2,560
  rather than the usual 4 × 768, keeping the parameter count similar.
- **Pre-norm residuals**: `x = x + attention(norm(x))`. The input passes
  straight through and each layer only adds a correction. That direct path
  lets gradients reach early layers in a deep stack.

## Step 5: Counting parameters

`ModelConfig.expected_parameters` counts by formula:

| Part | Formula | Size |
|---|---|---:|
| Embedding (shared with output) | 32,768 × 768 | 25,165,824 |
| Attention, per block | 4 × 768² | 2,359,296 |
| SwiGLU, per block | 3 × 768 × 2,560 | 5,898,240 |
| RMSNorm weights, per block | 2 × 768 | 1,536 |
| 12 blocks | | 99,108,864 |
| Final norm | | 768 |
| **Total** | | **124,275,456** |

The test builds the real model on PyTorch's `meta` device, which has shapes
but no memory, and checks it gets the same number.

**Tied weights:** the output layer reuses the embedding matrix, which saves
25M parameters. One side effect showed up while writing the tests: an untrained
model already favors repeating its own input token. A first version of the
"initial loss ≈ ln(vocab)" test used the inputs as targets and failed for that
reason; with unrelated targets the loss is ln(64) as expected.

## Step 6: Checks that catch real bugs

| Test | Catches |
|---|---|
| Initial loss ≈ ln(vocab) | broken initialization or loss |
| Every parameter gets a finite, nonzero gradient | layers that are disconnected or never used |
| Overfit one batch to ~0 loss | most wiring mistakes; if it can't memorize 32 tokens, nothing else matters |
| Save → load → identical logits | missing weights in checkpoints |

## Step 7: Generating text

`lm/generate.py` predicts one token, appends it, and repeats:

- **Greedy** (`temperature=0`) always picks the top token. It often loops.
- **Temperature** divides logits before softmax: below 1 is safer, above 1 more varied.
- **Top-k** keeps the k best tokens; **top-p** keeps the fewest top tokens whose
  probabilities add up to p.
- **Repetition penalty** lowers the scores of tokens already in the text.

## Step 8: See it learn

```sh
.venv/bin/python scripts/train_tiny.py --input data/fineweb_edu/train.jsonl --max-mb 5
```

The 918,656-parameter `tiny` model trains for 74 seconds on the Mac's GPU. The
loss starts at 6.27, about ln(512) = 6.24, which is uniform guessing, and falls
to about 3.0. Greedy output loops ("of the pressure of the pressure…"), while
sampling gives English-looking word salad. That is right for a model this small
after a minute. The real training loop, with validation, checkpoints and mixed
precision, is Phase 3.

## Check yourself

You understand this stage when you can answer:

1. Why divide attention scores by √Dh?
2. What exactly would go wrong without the causal mask?
3. Why does RoPE make attention depend on relative position?
4. Why is the initial loss about ln(vocab_size)?
5. Why does greedy decoding repeat itself?
