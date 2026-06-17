# Training Settings — Guide

This folder holds **`settings.json`**, the one place you edit to control training
**without touching any code**.

## How it works
1. Open `settings.json`.
2. Change a value (e.g. `"max_steps": 20000` → `40000`) and save.
3. Run training from the repo root: `python scripts/train.py`.
4. Check what is active any time: `python -m core.config`

These values **override** the auto-detected hardware profile in `core/config.py`.
If you delete a key from `settings.json`, that setting falls back to the
profile's default. Anything is allowed; the program reads this file every run.

> NOTE: `saved_model/config.json` is **output only** (written while training to
> record what was used). Do **NOT** edit that one — edit **this** `settings.json`.

---

## The settings

### `profile`
- **What:** Which hardware preset to use (sets model size + sensible defaults).
- **Options:** `"auto"`, `"cpu"`, `"gpu_4gb"`, `"gpu_8gb"`, `"gpu_12gb"`, `"gpu_24gb"`, `"gpu_40gb"`
- **Default:** `"auto"` (picks the biggest profile that fits your **free** VRAM).
- **When to change:** Force a specific size, or force `"cpu"`. For multi-GPU
  (`torchrun`), force it with the `MIRON_PROFILE` env var instead.

### `max_steps`
- **What:** Total number of training steps = **how long/how much it trains**.
  (This project is *step-based*; there is no "epochs". More steps = the model
  sees more data = more training.)
- **Options:** any positive integer (e.g. `2000` for a quick test, `20000`–`150000+` for real runs).
- **Default:** depends on profile (`gpu_4gb` = `20000`).
- **When to change:** Train longer (usually better) or shorter (quick test).
- **RESUME RULE:** To continue an existing checkpoint, set this **higher than the
  saved step** — otherwise the loop has nothing left to run.

### `lr`  (learning rate)
- **What:** Peak learning rate, reached after warmup, then cosine-decays to `min_lr`.
- **Options:** float, typical `0.0001`–`0.0006` (`1e-4` to `6e-4`).
- **Default:** `0.0006`.
- **When to change:** Tuning. Too high = unstable / NaN loss; too low = very slow.

### `min_lr`
- **What:** The lowest learning rate at the end of the cosine schedule.
- **Options:** float, usually about `lr / 10`.
- **Default:** `0.00006`.
- **When to change:** Rarely; keep it ~10x smaller than `lr`.

### `warmup_steps`
- **What:** Steps to linearly ramp LR from 0 up to `lr` at the very start.
- **Options:** integer, e.g. `100`–`2000`.
- **Default:** `200`.
- **When to change:** Increase if the start is unstable or the batch is very large.

### `batch_size`
- **What:** Micro-batch — how many sequences are processed at once per step.
- **Options:** positive integer. **Bigger = more VRAM (OOM risk).**
- **Default:** `1` (for a 4GB-class GPU).
- **When to change:** Raise if you have spare VRAM; lower if you hit out-of-memory.

### `grad_accum`  (gradient accumulation)
- **What:** How many micro-batches to accumulate before one optimizer update.
  **Effective batch = `batch_size` × `grad_accum` × (number of GPUs).**
- **Options:** positive integer.
- **Default:** `64`.
- **When to change:** To get a bigger *effective* batch **without** using more VRAM,
  raise this instead of `batch_size`.

### `eval_every`
- **What:** Run validation every N steps (prints `val_loss`, saves best checkpoint).
- **Options:** positive integer (steps).
- **Default:** `500`.
- **When to change:** Want more/less frequent validation.

### `eval_iters`
- **What:** How many validation batches to average per evaluation.
- **Options:** positive integer.
- **Default:** `50`.
- **When to change:** Higher = steadier `val_loss` reading, but slower eval.

### `log_every`
- **What:** Print a training log line every N steps.
- **Options:** positive integer.
- **Default:** `10`.
- **When to change:** Purely cosmetic (how chatty the logs are).

### `save_every`
- **What:** Write a checkpoint every N steps.
- **Options:** positive integer.
- **Default:** `500`.
- **When to change:** Lower = safer (saves more often) but slightly slower.

### `num_workers`
- **What:** Number of DataLoader worker processes that feed data to the GPU.
- **Options:** integer `>= 0` (`0` = load in the main process).
- **Default:** `2`.
- **When to change:** Raise for faster data loading; lower (even `0`) on low-RAM
  machines or if Windows throws worker errors.

### `weight_decay`
- **What:** AdamW weight decay (regularization).
- **Options:** float `>= 0`, typical `0.0`–`0.1`.
- **Default:** `0.1`.
- **When to change:** Advanced only; the default is fine.

### `grad_clip`
- **What:** Clip gradients to this maximum norm (training stability).
- **Options:** float `> 0` (use `0` to disable clipping).
- **Default:** `1.0`.
- **When to change:** Advanced only; keep ~`1.0` to prevent loss spikes.

---

## Things that are NOT in this file (on purpose)

- **Model shape** (`context_length`, `d_model`, `num_heads`, `num_kv_heads`,
  `num_layers`, `d_ff`, `dropout`, `tie_weights`, `grad_checkpoint`) lives in the
  profiles inside **`core/config.py`**. Changing the model shape **after** a
  checkpoint exists makes the old checkpoint incompatible → training restarts from
  scratch. Only change it before your first run.
- **Generation settings** (`temperature`, `repetition_penalty`, `top_k`, `top_p`)
  are **chat-time** settings, not training. Adjust them while running
  `python scripts/chat.py` (type `settings` there), or in its defaults.

---

## Common recipes

- **Train longer:** `"max_steps"` → bigger number (and make sure it's higher than
  your current saved step if resuming).
- **Hit out-of-memory (OOM):** lower `"batch_size"` (and/or raise `"grad_accum"`
  to keep the effective batch the same).
- **Quick smoke test:** `"max_steps": 50`, `"eval_every": 1000`, `"save_every": 1000`.
- **Force CPU:** `"profile": "cpu"`.
- **Bigger effective batch, same VRAM:** raise `"grad_accum"`.

Effective batch = `batch_size` × `grad_accum` × GPUs.
Tokens per step = effective batch × `context_length`.
