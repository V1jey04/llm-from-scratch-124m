# Build a tokenizer and a GPT-2-class 124M model from scratch, on free compute

## 1. Goal and definition of "done"

Build, from scratch, a byte-level BPE tokenizer and a ~124M-parameter decoder-only Transformer. Train it on about **2 billion tokens** of FineWeb-Edu using only **Kaggle's free GPUs** (total cash budget: **$0**, hard cap $10). Then measure it against OpenAI's GPT-2 (124M) and tiktoken.

This is a portfolio project. Its value comes from **measured results against known baselines**, a working demo, and a clear write-up, not from model size. You should be able to explain every line in an interview.

"From scratch" means: you write the tokenizer, model blocks, training loop, data pipeline, checkpointing, evaluation, and sampler. PyTorch supplies tensors, autograd, and GPU kernels (including `scaled_dot_product_attention`). Do not use a prebuilt Transformer class, a pretrained tokenizer, or pretrained weights in your model. Reference implementations (for example Karpathy's *build-nanogpt*) are for checking your work **after** your own first pass, never for copying.

**Completion criteria**

1. **Tokenizer:** 32,768-ID byte-level BPE with GPT-2-style regex pre-splitting. It round-trips arbitrary UTF-8 losslessly, and is benchmarked against tiktoken's `gpt2` encoding on compression (bytes/token) and speed (MB/s).
2. **Model:** ~124M unique parameters with a modern architecture (RoPE, RMSNorm, SwiGLU, tied embeddings). It passes the shape, causal-mask, overfit, and save/reload tests.
3. **Training:** at least **2B new training tokens** on Kaggle 2×T4 with resumable checkpoints across 12-hour sessions, and a single continuous loss curve.
4. **Evaluation:**
   - HellaSwag accuracy and held-out **bits per byte**, in the same table as GPT-2 124M
   - an ablation table from small (~20M) proxy models
5. **Release:**
   - weights, config, data manifest, and logs
   - a small instruction-tuned variant
   - a public Hugging Face Space demo
   - a write-up covering results, costs, and what went wrong

**Honest expectation:** at 2B tokens, expect to land at or somewhat below GPT-2 124M on HellaSwag (GPT-2 scores about 29–30%; random guessing is 25%). Report the gap as measured. Output text will be grammatical and on-topic for a few sentences. The model will not follow instructions or state facts reliably.

## 2. Key decisions

| Choice | Decision | Why |
|---|---|---|
| Framework | **PyTorch** | One codebase for `mps` (Mac) and `cuda` (Kaggle). |
| Compute | Mac for development and tests; **Kaggle T4×2** for every GPU run | $0. Kaggle has retired the P100, so T4×2 is the option. |
| Tokenizer | Your byte-level BPE, **32,768 IDs** (256 bytes + 3 special + 32,509 merges), regex pre-split | Close to GPT-2's design, so the comparison is fair. IDs fit in `uint16`. |
| Architecture | Pre-norm decoder, RoPE, RMSNorm, SwiGLU, no biases, tied input/output embedding | Modern LLaMA-style components, each a common interview topic. |
| Context | 1,024 tokens | Matches GPT-2. |
| Precision | **fp16 autocast + `GradScaler`**, loss and softmax in fp32 | T4 has no bf16. |
| Attention | Your own Q/K/V/RoPE code + `F.scaled_dot_product_attention(is_causal=True)` | The memory-efficient kernel runs on T4. FlashAttention needs newer GPUs; don't rely on it. |
| Parallelism | One T4 first, then **DDP on 2 T4s** via `torchrun` | DDP roughly doubles throughput and is itself a resume skill. |
| Optimizer | AdamW, warmup + cosine decay, gradient clipping 1.0 | Standard, well-understood baseline. |
| Data | FineWeb-Edu (`sample-10BT` subset), ~2.2B tokens train + fixed validation set | High-quality educational web text; you never download the whole thing. |
| Evaluation | Bits per byte (independent of tokenizer), HellaSwag, fixed-prompt samples | Perplexity can't be compared across different tokenizers; bits per byte and HellaSwag can. |
| Logging | Weights & Biases (API key stored in **Kaggle Secrets**, never in code) | One loss curve across sessions, viewable from your phone. |

## 3. Compute budget (Kaggle free tier)

Kaggle limits: about **30 GPU-hours/week** (check your account), **12-hour** sessions, **~20 GB** saved output in `/kaggle/working`, phone verification required for GPU and internet. CPU-only notebooks don't use GPU quota.

**Throughput is an estimate until Phase 4 measures it.** Planning assumes **20–30k tokens/s** for the 124M model on 2×T4.

| Run | Tokens | GPU-hours (est.) | When |
|---|---:|---:|---|
| Smoke tests, benchmarks, DDP check | small | ~3 | Week A |
| Ablations: 6 runs × ~20M params | 6 × 200M | ~3–4 | Week A |
| 124M pilot run | ~300M | ~3–4 | Week A |
| **Main run** | **2B** | **~19–28** | **Week B (fresh quota)** |
| Evals + instruction fine-tune | — | ~2–3 | Week C |

Rules:

- **Never debug on the GPU.** Everything must run end-to-end on the Mac (tiny model, CPU or `mps`) before it goes to Kaggle.
- Tokenize data in a **CPU-only** Kaggle notebook.
- Launch long runs with **Save Version → Save & Run All (Commit)** so they keep running after the browser closes.
- If throughput comes out below ~15k tokens/s, cut the main run to 1–1.5B tokens rather than spreading it over several weeks. Spending the $10 cap on a few rented GPU-hours is the fallback, not the plan.

## 4. Repository layout

```text
README.md                 # setup, commands, final results table
PLAN.md                   # this plan
pyproject.toml            # pinned dependencies (torch, numpy, datasets, wandb, tiktoken for benchmarks only)
tokenizer/                # your BPE: bpe.py, regex pre-split, fast encode, CLI (exists)
lm/config.py              # model/training configs, exact parameter count
lm/model.py               # embeddings, RoPE, attention, SwiGLU, RMSNorm, blocks
lm/data.py                # uint16 shard reader, random 1,024-token windows, DDP-aware
lm/train.py               # AMP, accumulation, schedule, DDP, checkpoint/resume, W&B
lm/generate.py            # greedy, temperature, top-k, top-p, repetition penalty
lm/evaluate.py            # validation loss, bits per byte, HellaSwag
scripts/prepare_data.py   # stream FineWeb-Edu, split, tokenize into shards + manifest
scripts/bench_tokenizer.py# compression and speed versus tiktoken
configs/*.json            # tiny, ablation_20m, pilot_124m, main_124m, sft
kaggle/                   # thin notebooks that call the scripts above
tests/                    # tokenizer, model, data, and checkpoint tests
models/ data/ runs/       # ignored by Git
```

Kaggle notebooks should only install, fetch the code, and call scripts. All logic lives in the repository so it's testable on the Mac.

## 5. Phases and exit gates

Part-time (5–8 h/week), expect roughly **8–12 weeks**. Each phase lists what to learn just enough of, what to build, and the gate that must pass before moving on.

### Phase 1 — Tokenizer to production quality (weeks 1–3)

**Already done:** byte-level BPE with special IDs 256–258, deterministic merges that never cross documents, save/load, CLI (`train`/`encode`/`decode`/`inspect`), round-trip tests.

**Learn:** regular expressions, Unicode categories, why pre-splitting matters, profiling Python (`cProfile`), `multiprocessing`.

1. **Regex pre-splitting.** Split text into chunks with a GPT-2-style pattern (letters, numbers, punctuation, whitespace, contractions) before BPE, so merges never cross chunk boundaries. Note that the pattern needs the `regex` package (for `\p{L}`) or a careful standard-library equivalent.
2. **Fast training.** Count pairs over *unique chunks weighted by frequency* instead of over the raw text. Update pair counts incrementally after each merge instead of recounting everything. Keep the current simple trainer as a test oracle: on small inputs both must produce identical merges.
3. **Fast encoding.** Cache the encoding of each unique chunk, and encode documents in parallel across processes. Target ≥ 5 MB/s total on the Mac, so the full corpus tokenizes in under ~2 hours.
4. **Train the 32,768-ID vocabulary** on a ~200–500 MB sample of the *training* split only. Freeze it and record its SHA-256 hash.
5. **Benchmark against tiktoken `gpt2`** on held-out FineWeb-Edu text (`scripts/bench_tokenizer.py`): bytes/token, encode MB/s, round-trip on tricky text (emoji, CJK, code, mixed whitespace). The results go straight into the write-up.

**Gate:** 100% round-trip on all test strings; fast trainer matches the oracle; a frozen 32k tokenizer with a benchmark table.

### Phase 2 — Model and correctness on the Mac (weeks 3–5)

**Learn:** embeddings, Q/K/V shapes, causal masking, softmax scaling, residuals, normalization, RoPE, SwiGLU (the Week 2–5 items of the old beginner path).

1. Write one causal attention head with explicit matrix operations, and test it against a hand-computed tiny example.
2. Build the full block: RMSNorm → multi-head attention with RoPE → residual; RMSNorm → SwiGLU MLP → residual. Then build the model with tied embeddings and a final RMSNorm.
3. Then swap in `scaled_dot_product_attention` and test that it matches your explicit version within tolerance.
4. Tests:
   - shapes at every module boundary
   - **causality:** changing token *t+1* leaves logits ≤ *t* unchanged
   - the exact parameter count
   - finite gradients
   - **overfitting a single batch** to near-zero loss
   - save → load → identical logits
5. Sampler: greedy first, then temperature, top-k, top-p, and repetition penalty. Stop on `<eos>` and at the maximum token count.

**Main configuration (≈124.3M parameters):**

| Item | Value |
|---|---:|
| Vocabulary | 32,768 |
| Layers / width / heads | 12 / 768 / 12 (head dim 64) |
| MLP | SwiGLU, hidden 2,560 |
| Context | 1,024 |
| Norm / positions | Pre-RMSNorm / RoPE |
| Biases | None |
| Output head | Tied to token embedding |

Count: embedding 32,768×768 = 25.17M; each block = attention 4×768² (2.36M) + SwiGLU 3×768×2,560 (5.90M) ≈ 8.26M; 12 blocks ≈ 99.1M; **total ≈ 124.3M**. The exact figure must come from code.

**Gate:** all model tests pass; a ~1M model trained on the tiny corpus produces readable samples.

### Phase 3 — Data pipeline and training loop (weeks 5–7)

**Learn:** train/validation leakage, streaming data, memory-mapped arrays, AdamW state, learning-rate schedules, mixed precision, gradient accumulation, DDP basics.

1. `scripts/prepare_data.py`: stream FineWeb-Edu `sample-10BT`, split **by document ID before tokenization** (fixed validation set of ~10M tokens), and append `<eos>` after each document. Write `uint16` shards of ~100M tokens plus a manifest (dataset revision, document counts, token counts, tokenizer hash, checksums). Run it in a CPU-only Kaggle notebook and publish the output as a **private Kaggle dataset**.
2. `lm/data.py`: memory-map the shards, sample random 1,024-token windows (inputs and targets shifted by one), and give each DDP rank a different random stream. Count **unique training tokens** and repeated tokens separately.
3. `lm/train.py`:
   - fp16 autocast + `GradScaler`
   - gradient accumulation and clipping
   - AdamW: weight decay 0.1 on matrices only, not on norms or embeddings
   - schedule: warmup, then cosine decay to 10% of peak
   - log every N steps; evaluate on fixed validation batches periodically
   - **atomic checkpoints every ~30 min** (write to a temp file, then rename), storing model, optimizer, scaler, scheduler, step, tokens seen, RNG state, config, and tokenizer hash
   - **resume from the latest checkpoint** automatically
4. **Resume test:** train N steps, stop, resume, train N more. Check the loss matches an uninterrupted 2N-step run within fp16 noise.
5. DDP: launch with `torchrun --nproc_per_node=2`; only rank 0 logs and saves. Test on the Mac with the `gloo` backend and 2 CPU processes before using Kaggle quota.

**Starting hyperparameters** (the pilot confirms or changes them):
- 262,144 tokens per optimizer step: 2 GPUs × microbatch 8 × 1,024 tokens × 16 accumulation steps
- peak learning rate 6e-4, warmup ~2% of steps
- 2B tokens ≈ 7,600 steps

**Gate:** a tiny run on the Mac completes, resumes exactly, and the DDP run on 2 CPU processes matches single-process results.

### Phase 4 — Kaggle pilot and ablations (week 8, "Week A" quota)

1. **Smoke test** on one T4 (100 steps), then 2×T4 DDP. Measure synchronized tokens/s, peak memory, and checkpoint size. Adjust the microbatch to fit 16 GB per GPU.
2. **Ablations** with ~20M-parameter proxy models (e.g. 6 layers, width 384), ~200M tokens each, same data and seed:
   - RoPE vs. learned position embeddings
   - SwiGLU vs. GELU MLP
   - 32k vs. 16k vocabulary (compare **bits per byte**, not loss)

   Report validation bits per byte and tokens/s for each. Small but real experiments like these set the project apart from a tutorial copy.
3. **Pilot:** the 124M config for ~300M tokens. Check that the loss decreases smoothly, with no fp16 overflow storms (watch the `GradScaler` scale value) and no NaNs. If unstable, lower the learning rate or increase warmup before the main run.
4. Fill in the real throughput and re-estimate the main run's GPU-hours.

**Gate:** a stable pilot loss curve, a measured tokens/s figure, a main-run estimate that fits one week's quota, and an ablation table.

### Phase 5 — Main run (week 9, "Week B" quota)

1. Start the 2B-token run when the weekly quota resets. Chain 2–3 committed sessions: each one attaches the previous version's output as input and resumes from its latest checkpoint.
2. Keep two checkpoints in `/kaggle/working` (latest + best validation) to stay under ~20 GB. A 124M checkpoint with optimizer state is about 1.5 GB.
3. Watch W&B between sessions. If the loss spikes, resume from the last good checkpoint with a lower learning rate, and note it in the write-up.

**Gate:** 2B unique training tokens processed; the final checkpoint generates text in a fresh process.

### Phase 6 — Evaluation, fine-tune, demo, write-up (weeks 10–12, "Week C")

1. **Evaluation table:** your model vs. GPT-2 124M (loaded via `transformers` **only for evaluation**), with both models scored the same way:
   - validation bits per byte on the same held-out FineWeb-Edu documents
   - HellaSwag (normalized accuracy, the standard zero-shot method)
   - fixed-prompt samples
2. **Instruction fine-tune:** a few thousand examples from a small instruction dataset with a license that allows your use (check it). Mask the loss to the response tokens only; train for ~1 GPU-hour. Expect it to learn the chat format, not reliable answers.
3. **Demo:** a Hugging Face Space (the free CPU tier runs a 124M model fine) with sampling controls and a clear limitations note.
4. **Write-up** (README + optional blog post):
   - architecture diagram
   - tokenizer benchmark
   - ablation table
   - loss curve
   - evaluation table
   - cost ($0) and GPU-hours used
   - what broke and how you fixed it

**Gate:** someone can open the Space link, read the results table, and reproduce inference from the README.

## 6. Failure checks and fixes

| Symptom | First check | Action |
|---|---|---|
| Train loss stuck near `ln(32768)` ≈ 10.4 | Targets shifted by one? Mask correct? Gradients nonzero? | Rerun the one-batch overfit test; inspect one example by hand. |
| `GradScaler` scale keeps falling, or loss is NaN | fp16 overflow in logits or attention | Compute loss/softmax in fp32; lower learning rate; longer warmup; check for bad token IDs. |
| Loss spike mid-run | Learning rate too high, bad data shard | Resume from the previous checkpoint with a lower learning rate; skip or inspect the shard. |
| Validation loss rises early | Train/validation overlap, learning rate, data quality | Audit the split by document ID; compare train vs. validation curves. |
| Slow steps on Kaggle | Running on CPU? Data loader starved? Unsynchronized timing? | Check device; pre-tokenized shards only; `torch.cuda.synchronize()` before timing. |
| Out of memory on a T4 | Microbatch, sequence length | Halve the microbatch and double accumulation; try activation checkpointing. |
| Resumed run differs | Missing optimizer, scaler, RNG, or sampler state | Save and restore every state; keep the resume test in CI. |
| DDP hangs | Ranks took different code paths | Make sure every rank runs the same number of forward/backward steps; only rank 0 does I/O. |
| Repetitive samples | Undertraining or sampling settings | Check validation loss; tune temperature, top-p, and repetition penalty. |

## 7. Definitions to keep at hand

- A **token** is one vocabulary ID, not necessarily a word. **Bits per byte** = total cross-entropy (in bits) ÷ total UTF-8 bytes. It is comparable across tokenizers; perplexity is not.
- **Attention:** `softmax(QKᵀ/√d_head + causal_mask) V`. The **causal mask** stops each position from seeing future tokens.
- **RoPE** rotates Q and K by position-dependent angles, so attention scores depend on relative position.
- **RMSNorm** rescales a vector by its root-mean-square, with no mean subtraction and no bias.
- **SwiGLU:** `W2(silu(W1 x) ⊙ W3 x)`, a gated MLP used in most modern LLMs.
- **Gradient accumulation** sums gradients over several microbatches before one optimizer step.
- **DDP** runs one model copy per GPU on different data and averages gradients every step.
- **`GradScaler`** multiplies the loss so small fp16 gradients don't underflow to zero, and skips steps that overflow.

## 8. First work sessions

1. Add regex pre-splitting to the tokenizer, with tests showing merges never cross chunk boundaries.
2. Write the fast trainer (frequency-weighted chunks, incremental pair counts); test it against the current trainer as the oracle.
3. Add chunk caching and multiprocessing to encoding; write `scripts/bench_tokenizer.py` against tiktoken.
4. Train a 32k vocabulary on a ~200 MB FineWeb-Edu sample; record the benchmark table.
5. Implement one causal attention head with tests (shapes, causality, hand-computed example).
6. Build the full model; overfit one batch; check the exact parameter count.
7. Build the data shards and training loop with checkpoint/resume on the Mac.

Revisit this plan after session 7 with what you have measured.

## 9. References

- Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2): the baseline and its regex pre-splitting tokenizer.
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Su et al., [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864); Zhang & Sennrich, [RMSNorm](https://arxiv.org/abs/1910.07467); Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202); Touvron et al., [LLaMA](https://arxiv.org/abs/2302.13971).
- Hoffmann et al., [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556): the ~20 tokens/parameter heuristic behind the 2B target.
- Zellers et al., [HellaSwag](https://arxiv.org/abs/1905.07830).
- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu): read its terms before publishing anything trained on it.
- [tiktoken](https://github.com/openai/tiktoken): the tokenizer benchmark baseline only.
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html), [DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html), [MPS backend](https://docs.pytorch.org/docs/main/notes/mps.html).
- [Kaggle notebooks docs](https://www.kaggle.com/docs/notebooks), [Kaggle GPU usage guidance](https://www.kaggle.com/docs/efficient-gpu-usage).
- Karpathy, [build-nanogpt](https://github.com/karpathy/build-nanogpt): cross-check after writing each part yourself.
