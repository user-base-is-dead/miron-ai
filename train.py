"""
─────────────────────────────────────────────────────────────────────────────
  TRAINING  —  enterprise-grade pretraining loop
─────────────────────────────────────────────────────────────────────────────
  Features:
    • Step-based training (not epoch-based) — standard for LLM pretraining
    • Mixed precision (bf16 if supported, else fp16 with GradScaler)
    • Gradient accumulation -> large effective batch on a 6GB GPU
    • Cosine LR schedule with linear warmup
    • Gradient clipping + fused AdamW with weight-decay groups
    • Periodic validation + best-checkpoint saving
    • Full resume (model + optimizer + scaler + step)
    • Optional torch.compile

  Workflow:
    1) python prepare_data.py   (one time)
    2) python train.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from config import get_active_config, build_model_config
from dataset import get_bin_dataloaders
from maninmiron_llm import ManinmironLLM


# ── Training hyperparameters ─────────────────────────────────────────────────
# Saari config ab config.py me hai (device profiles ke roop me). Yahan kuch
# define karne ki zaroorat nahi — get_active_config() se profile load hota hai.


def get_lr(step, c):
    if step < c.warmup_steps:
        return c.lr * (step + 1) / c.warmup_steps
    if step > c.max_steps:
        return c.min_lr
    ratio = (step - c.warmup_steps) / max(1, c.max_steps - c.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return c.min_lr + coeff * (c.lr - c.min_lr)


@torch.no_grad()
def evaluate(model, val_loader, ctx, device, max_iters):
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(max_iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_ckpt(path, raw_model, optimizer, scaler, step, best_val, model_cfg, train_cfg):
    Path(train_cfg.save_folder).mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":      raw_model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scaler":     scaler.state_dict() if scaler is not None else None,
        "step":       step,
        "best_val":   best_val,
        "model_cfg":  model_cfg.to_dict(),
    }, path)
    # human-readable config alongside
    cfg_out = dict(model_cfg.to_dict())
    cfg_out.update({"step": step, "best_val_loss": round(best_val, 4)})
    with open(f"{train_cfg.save_folder}/config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)


def train():
    c = SimpleNamespace(**get_active_config())
    torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"
    print(f"Profile: {c.profile_name}")
    print(f"Device: {device.upper()}")

    # precision: bf16 if GPU supports it, else fp16
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device_type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type="cuda", dtype=amp_dtype))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    print(f"AMP dtype: {amp_dtype}")

    # data
    train_loader, val_loader, vocab_size = get_bin_dataloaders(
        c.data_folder, c.batch_size, c.context_length, c.num_workers
    )
    print(f"vocab_size: {vocab_size}")

    # model
    model_cfg = build_model_config(vars(c), vocab_size)
    model = ManinmironLLM(model_cfg).to(device)

    optimizer = model.configure_optimizer(
        c.lr, c.weight_decay, (c.beta1, c.beta2), device_type, c.optimizer_type
    )

    # resume
    start_step = 0
    best_val = float("inf")
    ckpt_path = f"{c.save_folder}/maninmiron.pt"
    if Path(ckpt_path).exists():
        print(f"Resuming from {ckpt_path}...")
        ck = torch.load(ckpt_path, map_location=device)
        if "model" in ck:
            try:
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])
                if scaler is not None and ck.get("scaler"):
                    scaler.load_state_dict(ck["scaler"])
                start_step = ck.get("step", 0)
                best_val = ck.get("best_val", float("inf"))
                print(f"  resumed at step {start_step} (best_val {best_val:.4f})")
            except (RuntimeError, ValueError, KeyError) as e:
                # Profile/architecture/optimizer badal gaya -> purana checkpoint
                # fit nahi hoga. Crash ke bajaye fresh training shuru karo.
                print(f"  checkpoint current profile se match nahi karta ({e})")
                print("  -> fresh training shuru kar rahe hain")
                start_step, best_val = 0, float("inf")
        else:
            print("  old-format checkpoint found, incompatible architecture -> training fresh")

    # compile (after load to avoid state_dict key prefixing issues)
    raw_model = model
    if c.compile_model and device_type == "cuda":
        print("Compiling model (first step will be slow)...")
        model = torch.compile(model)

    print(f"\n{'='*56}")
    print("  TRAINING START")
    print(f"  steps {start_step} -> {c.max_steps}")
    print(f"  micro-batch {c.batch_size} x accum {c.grad_accum} = "
          f"{c.batch_size * c.grad_accum} effective")
    print(f"  ctx {c.context_length} | tokens/step "
          f"{c.batch_size * c.grad_accum * c.context_length:,}")
    print(f"{'='*56}\n")

    # First few steps log every step (so it's clear training isn't stuck on a
    # slow GPU); after that we log every `log_every` steps.
    dense_log_steps = 20
    print(f"First optimizer step can be slow (data priming / compile). "
          f"Logging every step for the first {dense_log_steps} steps.\n", flush=True)

    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    window_loss = 0.0    # loss summed over the current logging window
    window_steps = 0     # number of steps in the current logging window

    for step in range(start_step, c.max_steps):
        lr = get_lr(step, c)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(c.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with ctx:
                _, loss = model(x, y)
                loss = loss / c.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item()

        if c.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        window_loss += step_loss
        window_steps += 1

        # Dense logging for the first few steps, then every `log_every` steps.
        # Loss and throughput are both averaged over the current window, so the
        # window is always reset together after a log line is printed.
        if (step + 1) % c.log_every == 0 or step < dense_log_steps:
            dt = max(1e-6, time.time() - t0)
            avg = window_loss / window_steps
            tokens = window_steps * c.batch_size * c.grad_accum * c.context_length
            print(f"step {step+1:>6}/{c.max_steps} | loss {avg:.4f} | "
                  f"lr {lr:.2e} | ~{tokens / dt / 1e3:.1f}k tok/s", flush=True)
            window_loss = 0.0
            window_steps = 0
            t0 = time.time()

        if (step + 1) % c.eval_every == 0:
            val_loss = evaluate(model, val_loader, ctx, device, c.eval_iters)
            print(f"  eval @ step {step+1}: val_loss {val_loss:.4f} "
                  f"(best {best_val:.4f})")
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt(f"{c.save_folder}/maninmiron_best.pt", raw_model,
                          optimizer, scaler, step + 1, best_val, model_cfg, c)
                print(f"  -> new best (val_loss {best_val:.4f}) saved")
            # don't count eval / checkpoint time against training throughput
            window_loss = 0.0
            window_steps = 0
            t0 = time.time()

        if (step + 1) % c.save_every == 0:
            save_ckpt(ckpt_path, raw_model, optimizer, scaler,
                      step + 1, best_val, model_cfg, c)

    save_ckpt(ckpt_path, raw_model, optimizer, scaler, c.max_steps, best_val, model_cfg, c)
    print(f"\nTraining complete. Checkpoints in {c.save_folder}/")


if __name__ == "__main__":
    train()
