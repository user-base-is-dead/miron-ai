# Miron LLM

A from-scratch, GPT/LLaMA-style decoder-only language model you can **pretrain on
your own hardware** — from a 4GB laptop GPU up to multi-GPU nodes — with automatic
hardware detection and a single editable settings file.

> Status: this trains a **base model** (it continues/completes text). To make it
> hold a conversation it additionally needs instruction fine-tuning (SFT). The
> chat format + special tokens are already wired in (`core/tokenizer.py`).

---

## Architecture

A modern decoder-only transformer (`core/miron_llm.py`):

- **RMSNorm** (faster, more stable than LayerNorm)
- **RoPE** rotary position embeddings (no learned position table)
- **Grouped-Query Attention (GQA)** + **Flash Attention** (`F.scaled_dot_product_attention`)
- **SwiGLU** gated feed-forward (LLaMA-style)
- **KV-cache** for fast autoregressive generation
- **Weight tying** (token embedding == output head)
- **Gradient checkpointing** (train bigger models on small GPUs)

Tokenizer: `tiktoken` **cl100k_base** (handles English / Hindi / Hinglish) plus 6
special tokens (`<|pad|> <|sos|> <|eos|> <|user|> <|assistant|> <|sep|>`),
vocab size **100263**.

---

## Project structure

```
core/                 # Support modules (imported by scripts/)
  config.py           #   profiles, hardware detection, settings loader
  miron_llm.py   #   the model (Config + MironLLM)
  dataset.py          #   memory-mapped .bin token loader
  tokenizer.py        #   tiktoken wrapper + chat format
scripts/              # Runnable entry points
  prepare_data.py     #   tokenize raw text -> data/tokenized/*.bin
  train.py            #   the training loop
  chat.py             #   terminal chat with a trained checkpoint
data-download/        # One-off data downloaders (delete after use if you want)
  download_data.py    #   pretraining corpora (Wikipedia, OpenWebText, ...)
  download_sft_data.py#   chat/instruction data (OpenHermes, UltraChat, Dolly)
settings/
  settings.json       # EDIT THIS to change training settings (no code needed)
  README.md           # what every setting does
data/                 # (git-ignored) downloaded-data/ (raw) + tokenized/ (.bin)
saved_model/          # checkpoints (.pt are git-ignored) + config.json snapshot
requirements.txt
```

All commands are run **from the repo root**.

---

## Setup

```bash
# create + activate a Python 3.11 environment, then:
pip install -r requirements.txt
# (bitsandbytes is optional — only for 8-bit/paged optimizers on GPU;
#  if it isn't installed, training falls back to standard AdamW automatically.)
```

## Full pipeline

```bash
# 1) Download raw pretraining text  ->  data/downloaded-data/
python data-download/download_data.py

# 2) Tokenize it once               ->  data/tokenized/{train.bin, val.bin, meta.json}
python scripts/prepare_data.py

# 3) Train (auto-detects your GPU / CPU)
python scripts/train.py

# 4) Chat with the trained base model
python scripts/chat.py
```

Check what hardware + profile + settings will be used, without training:

```bash
python -m core.config
```

---

## Hardware profiles & multi-GPU

`core/config.py` defines per-hardware **profiles** (model size + sensible
defaults). The profile is chosen automatically from your GPU's **free** VRAM
(`cpu`, `gpu_4gb`, `gpu_8gb`, `gpu_12gb`, `gpu_24gb`, `gpu_40gb`); no GPU → CPU.

Force a profile (highest priority first):
1. `MIRON_PROFILE=gpu_8gb python scripts/train.py`
2. `ACTIVE_PROFILE` constant in `core/config.py`
3. `"profile"` in `settings/settings.json`
4. auto-detect (default)

Multi-GPU on one machine (data-parallel via DDP):

```bash
torchrun --nproc_per_node=NUM_GPUS scripts/train.py
```

Multiple machines (advanced; network-bound, usually only worth it on fast
interconnects):

```bash
torchrun --nnodes=N --node_rank=R --nproc_per_node=G \
         --rdzv_endpoint=MASTER_IP:29500 scripts/train.py
```

DDP scales the learning rate by `sqrt(world_size)`. DDP makes training **faster**;
it does not let you train a model larger than one GPU's memory (that needs
FSDP/ZeRO, which is not implemented here).

---

## Settings

Edit **`settings/settings.json`** (flat `key: value`) to change `max_steps`, `lr`,
`batch_size`, `grad_accum`, eval/log/save cadence, etc. — these override the
auto-detected profile at runtime. See **`settings/README.md`** for what each one
does, valid values, and when to change it.

Note: this project is **step-based** (no "epochs"). `max_steps` controls how long
it trains. To continue an existing checkpoint, set `max_steps` higher than the
saved step.

---

## Notes

- **Resume** is automatic: if `saved_model/miron.pt` exists, training loads
  the model + optimizer + step and continues. Change the model architecture and
  the old checkpoint won't load (training restarts cleanly).
- **Continual training on new data:** don't delete the old data and train only on
  new — the model forgets it (catastrophic forgetting). Mix some old data with the
  new (data replay).
- **OOM safety:** on CUDA out-of-memory the trainer prints actionable guidance and
  exits cleanly instead of dumping a raw traceback.
- Large files (`data/`, `*.pt` checkpoints) are git-ignored. If they go missing,
  re-run `scripts/prepare_data.py` (re-tokenize) and `scripts/train.py`.
