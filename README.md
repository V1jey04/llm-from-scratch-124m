# Tokenizer + a 0.1B language model

A byte-level BPE tokenizer and a ~124M-parameter GPT-style model, written from
scratch. The tokenizer uses only the Python standard library; the model uses
PyTorch. Run commands from this project folder with Python 3.10+.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 -m tokenizer demo
.venv/bin/python -m unittest discover -s tests -v
```

Plain `python3 -m unittest` also works; it skips the model tests if PyTorch is
missing. Follow [the tokenizer lesson](TOKENIZER_LESSON.md) and
[the model lesson](MODEL_LESSON.md) and [the training lesson](TRAINING_LESSON.md)
to understand the code, then
use [the project plan](PLAN.md) for the later model and training stages.

Train on your own small UTF-8 text files:

```sh
python3 -m tokenizer train --input examples/tiny_corpus.txt --output models/tiny_tokenizer.json --vocab-size 280
python3 -m tokenizer encode --model models/tiny_tokenizer.json 'Hello, little robot!'
python3 -m tokenizer inspect --model models/tiny_tokenizer.json 'Hello, little robot!'
python3 -m tokenizer decode --model models/tiny_tokenizer.json 72 101 108 108 111
```

Text is pre-split into GPT-2-style chunks (words, numbers, symbols,
whitespace) before merging; add `--no-pre-split` to `train` to compare
without it. The `inspect` command shows the chunks and what each ID represents. The last command decodes
raw byte IDs into `Hello`. Learned token IDs start at
259 and depend on the training corpus. Models are saved locally under `models/`
and ignored by Git. Training stores each distinct chunk once and updates pair
counts incrementally: 32,509 merges on 36 MB of Python source take about 6 s.
`ByteBPETokenizer.train_reference` keeps the simple recount-everything version,
and the tests check that both learn identical merges. Encoding caches each
chunk's IDs (~10 MB/s on one core), and `encode_many` spreads documents across
all cores (~50 MB/s on an M4 with 10 cores).

## Results: 32k tokenizer on FineWeb-Edu

Download a reproducible sample (304 MB for training, 25 MB held out from a
different file; `data/fineweb_edu/manifest.json` pins the dataset revision and
row groups), train, and compare with GPT-2's tokenizer:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/sample_fineweb.py --train-mb 300 --heldout-mb 20
python3 -m tokenizer train --input data/fineweb_edu/train.jsonl --output models/fineweb_edu_32k.json --vocab-size 32768
.venv/bin/python scripts/bench_tokenizer.py --model models/fineweb_edu_32k.json --text data/fineweb_edu/heldout.jsonl
```

Held-out FineWeb-Edu, 5,000 documents, 24.7 MB, Apple M4 (10 cores):

| Tokenizer | Vocabulary | Bytes/token | Encode MB/s (1 core) | Tricky round-trips |
|---|---:|---:|---:|---:|
| **Ours** (pure Python) | 32,768 | **4.63** | 11.5 | 13/13 |
| tiktoken `gpt2` (Rust) | 50,257 | 4.64 | 37.1 | 13/13 |

- **Compression matches GPT-2 with a 35% smaller vocabulary.** Caveat: ours
  was trained on FineWeb-Edu itself, while GPT-2's was trained on WebText, so
  ours has a home-field advantage on this text.
- Training: 32,509 merges on 304 MB in 46 s, peak memory 1.3 GB.
- Speed: tiktoken is written in Rust and is about 3× faster per core.
  `encode_many` uses every core and reaches about 50 MB/s on larger inputs.
- Tokenizer file SHA-256 (first 16 hex digits): `af1bf016d3825e14`.

## The model

`lm/` holds a decoder-only Transformer in the LLaMA style: RoPE positions,
RMSNorm, SwiGLU feed-forward layers, no biases, and an output layer tied to the
token embedding. `lm/config.py` defines the sizes:

| Preset | Layers | Width | Heads | Vocabulary | Context | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| `main_124m` | 12 | 768 | 12 | 32,768 | 1,024 | 124,275,456 |
| `ablation_20m` | 6 | 384 | 6 | 32,768 | 1,024 | ~23.2M |
| `tiny` | 4 | 128 | 4 | 512 | 128 | 918,656 |

`tests/test_model.py` checks the exact parameter count, causality (changing a
later token never changes earlier outputs), fused vs. hand-written attention,
RoPE's relative-position property, gradients, overfitting one batch, save/load,
and the sampler (greedy, temperature, top-k, top-p, repetition penalty).

End-to-end smoke run: a 512-ID tokenizer and the `tiny` model trained on 5 MB
of FineWeb-Edu (1,500 steps, 74 s on an M4's GPU; loss 6.27 → about 3.0):

```sh
.venv/bin/python scripts/train_tiny.py --input data/fineweb_edu/train.jsonl --max-mb 5
```

## Data and training

`scripts/prepare_data.py` tokenizes documents into uint16 shard files with a
`manifest.json` (counts, checksums, tokenizer hash) and a copy of the
tokenizer. Validation documents come from a different file than training ones
(FineWeb-Edu file 013 is never trained on). Locally, 304 MB of text becomes
66M training tokens in about 20 s:

```sh
.venv/bin/python scripts/prepare_data.py --tokenizer models/fineweb_edu_32k.json --out data/tokens_local \
    --train-jsonl data/fineweb_edu/train.jsonl --val-jsonl data/fineweb_edu/heldout.jsonl --val-tokens 2000000
```

`lm/train.py` trains from a JSON config. Any setting can be changed with
`--set`. Running the same command again resumes from `out_dir/checkpoint.pt`:

```sh
.venv/bin/python -m lm.train --config configs/smoke.json                 # Mac, ~6 min
torchrun --standalone --nproc_per_node=2 -m lm.train --config configs/main_124m.json   # Kaggle 2×T4
```

| Feature | How |
|---|---|
| Mixed precision | fp16 + `GradScaler` on T4s, bf16 on newer GPUs, fp32 on Mac/CPU |
| Batch size | `world_size × micro_batch_size × grad_accum_steps × context` tokens per step (262,144 for the main run) |
| Data order | Shuffled non-overlapping windows; each token is seen once per epoch, and the order is rebuilt from `(seed, epoch)` |
| Schedule | AdamW (β₂ 0.95), linear warmup then cosine to 10%, gradient clipping 1.0, no weight decay on norms or embedding |
| Checkpoints | Atomic writes every `checkpoint_minutes`; `time_limit_hours` saves and stops before a Kaggle session ends |
| Logs | Console, `out_dir/log.jsonl`, and Weights & Biases if `wandb_project` is set |

`tests/test_training.py` checks that a stopped-and-resumed run ends with
bit-identical weights, that gradient accumulation and two-process DDP match a
single big batch, and that fp16/bf16 training works.

Smoke run on the Mac (11.6M parameters, 4M tokens, M4 GPU, about 6 minutes
including a stop and resume at step 150): validation loss 5.82, 1.82 bits per
byte.

## Running on Kaggle (free 2×T4)

The trained 32k tokenizer is committed as `tokenizers/fineweb_edu_32k.json`
(SHA-256 starts `af1bf016d3825e14`), so Kaggle uses exactly the same one.

1. **`kaggle/01_prepare_data.ipynb`**, a CPU notebook with internet on:
   downloads FineWeb-Edu files 000–003 (training) and 013 (validation) and
   writes about 2.1B training tokens to `/kaggle/working/tokens`. It took 24
   minutes and wrote 2,100,005,190 training and 10,001,088 validation tokens
   (4.0 GB).
2. **`kaggle/02_train.ipynb`**, a GPU T4 ×2 notebook with the data notebook's
   output attached as an input (**Add Input → Notebook**; no separate dataset
   is needed): set `CONFIG`, `RUN` and `EXTRA` in the first cell. To resume,
   attach the previous version's output too; the newest checkpoint for `RUN`
   is copied in automatically.

Smoke tests of the 124M model on 2×T4 (20 steps, 262,144 tokens per step,
fp16). Losses were identical to three decimals in both runs:

| Setting | Tokens/s at steps 15–20 | Peak memory per GPU |
|---|---:|---:|
| Eager PyTorch | 15,096 | 11.2 GB |
| `compile: true` (now the default in the Kaggle configs) | 24,444 | 7.7 GB |

At about 24k tokens/s, 2B tokens take roughly 23 GPU-hours, within one week's
30-hour Kaggle quota.
