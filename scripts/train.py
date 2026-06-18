"""
─────────────────────────────────────────────────────────────────────────────
  TRAINING  —  enterprise-grade pretraining loop
─────────────────────────────────────────────────────────────────────────────
  Features:
    • Hardware auto-detect      — GPU count / VRAM / driver, warna CPU fallback
    • Free-VRAM-aware profile    — chooses model size from FREE (not total) VRAM
    • Multi-GPU (DDP)            — `torchrun --nproc_per_node=N train.py`
    • Step-based training        — standard for LLM pretraining
    • Mixed precision            — bf16 if supported, else fp16 + GradScaler
    • Gradient accumulation      — large effective batch even on small GPUs
    • Cosine LR + linear warmup
    • Grad clipping + AdamW (fused / 8-bit / paged) with weight-decay groups
    • Periodic validation + best-checkpoint saving + full resume
    • Optional torch.compile

  Workflow:
    1) python scripts/prepare_data.py                       (one time)
    2) python scripts/train.py                              (single GPU / CPU)
       torchrun --nproc_per_node=NUM_GPUS scripts/train.py  (all GPUs on one machine)
─────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

# ── Python version guard (torch import se pehle saaf error) ──────────────────
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"[Miron] Python 3.11.x chahiye (abhi {sys.version.split()[0]} chal raha hai).\n"
        "        venv activate karo -> Windows: Miron311\\Scripts\\activate"
        "  |  Linux/Mac: source Miron311/bin/activate"
    )

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Repo root ko sys.path pe daalo taaki 'core' package import ho sake (yeh file
# scripts/ ke andar hai). Isse `python scripts/train.py` (repo root se) chalta hai.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (build_model_config, format_hardware_report,
                         get_active_config, resolve_profile_name)
from core.dataset import get_bin_dataloaders
from core.miron_llm import MironLLM


# ── Distributed / device helpers ──────────────────────────────────────────────
def ddp_info():
    """torchrun se mile env padho. Returns (is_ddp, rank, local_rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (True,
                int(os.environ["RANK"]),
                int(os.environ.get("LOCAL_RANK", 0)),
                int(os.environ["WORLD_SIZE"]))
    return (False, 0, 0, 1)


def pick_best_gpu() -> int:
    """Single-process multi-GPU: sabse zyada FREE VRAM wala card chuno."""
    best_idx, best_free = 0, -1.0
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            free = torch.cuda.get_device_properties(i).total_memory
        if free > best_free:
            best_idx, best_free = i, free
    return best_idx


def cuda_mem() -> str:
    """Chhota GPU memory string logging ke liye (4GB cards pe pressure jaldi pakadne)."""
    if not torch.cuda.is_available():
        return ""
    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return f" | GPU {alloc:.0f}/{reserv:.0f}MiB (peak {peak:.0f})"
    except Exception:
        return ""


def estimate_fixed_vram_gb(param_count: int, optimizer_type: str) -> float:
    """Weights + grads + optimizer states ka rough FIXED VRAM cost (GB).

    Activations isme shaamil NAHI hain (woh batch / ctx / grad-checkpointing pe
    depend karte hain). Ye sirf ek 'floor' estimate hai — OOM se pehle heads-up
    dene ke liye, exact number nahi.
    """
    ot = optimizer_type.lower()
    weights = 4 * param_count                  # fp32 master weights
    grads = 4 * param_count                     # fp32 gradients
    if ot.startswith("paged"):
        optim = 0                               # paged: optimizer state CPU RAM me
    elif "8bit" in ot:
        optim = 2 * param_count                 # 2 states x 1 byte
    else:
        optim = 8 * param_count                 # AdamW: 2 fp32 states
    return (weights + grads + optim) / 1e9


# ── LR schedule ───────────────────────────────────────────────────────────────
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
    # ── Distributed context (sirf torchrun ke under active hota hai) ──────────
    is_ddp, rank, local_rank, world_size = ddp_info()
    is_master = (rank == 0)

    def log(*args, **kwargs):
        """Sirf rank 0 print kare (DDP me N copies na chhape)."""
        if is_master:
            print(*args, **kwargs)

    # ── Device selection (nccl ke liye device init se pehle set karna safe) ───
    if torch.cuda.is_available():
        device_type = "cuda"
        dev_index = local_rank if is_ddp else pick_best_gpu()
        torch.cuda.set_device(dev_index)
        device = f"cuda:{dev_index}"
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev_index)
        free_gb, total_gb = free_bytes / 1e9, total_bytes / 1e9
    else:
        device_type, device, dev_index = "cpu", "cpu", None
        free_gb, total_gb = None, None

    if is_ddp:
        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    # ── Hardware summary (master only) ────────────────────────────────────────
    log(format_hardware_report())
    if is_ddp:
        log(f"Distributed   : DDP across {world_size} process(es) "
            f"[backend {dist.get_backend()}]")
    elif device_type == "cuda" and torch.cuda.device_count() > 1:
        n = torch.cuda.device_count()
        log(f"Note          : {n} GPUs mile, par plain `python scripts/train.py` sirf "
            f"{device} use karega.")
        log(f"                Sab {n} GPUs use karne ke liye:  "
            f"torchrun --nproc_per_node={n} scripts/train.py")
    log("-" * 60)

    # ── Resolve profile ───────────────────────────────────────────────────────
    # DDP: rank-0 free VRAM dekh ke decide karta hai, phir sab ranks ko broadcast
    # (taaki har rank bilkul same model banaye). Single-process: is device ka
    # free VRAM use karo.
    if is_ddp:
        name = resolve_profile_name() if is_master else None
        obj = [name]
        dist.broadcast_object_list(obj, src=0)
        c = SimpleNamespace(**get_active_config(name=obj[0]))
    else:
        c = SimpleNamespace(**get_active_config(free_gb=free_gb))

    # per-rank seed offset -> har rank alag random windows sample karta hai
    torch.manual_seed(c.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Large-batch LR scaling (DDP) ──────────────────────────────────────────
    # DDP me effective batch world_size guna ho jaata hai (har GPU apna batch
    # chalata hai, gradients average hote hain). Bade batch ke saath LR bhi
    # badhana padta hai warna training dhimi/under-fit hoti hai. sqrt-rule use
    # karte hain (linear rule bade N pe unstable ho sakta hai). 1 GPU / laptop pe
    # koi change nahi (world_size == 1).
    if is_ddp and world_size > 1:
        lr_scale = world_size ** 0.5
        c.lr = c.lr * lr_scale
        c.min_lr = c.min_lr * lr_scale
        log(f"[lr] world_size={world_size} -> LR x{lr_scale:.2f} (sqrt rule): "
            f"lr={c.lr:.2e}, min_lr={c.min_lr:.2e}")
        log("[lr] NOTE: 100s of GPUs jaise extreme scale pe LR + warmup proper "
            "tuning maangte hain; ye ek reasonable default hai, magic nahi.")

    log(f"Profile: {c.profile_name}")
    log(f"Device : {device}")

    if c.profile_name == "gpu_4gb":
        log("\n[gpu_4gb] 4GB laptop mode — bahut tight. Training se pehle:")
        log("  - browser / VSCode / Electron apps band karo (~300-500MiB VRAM lete hain)")
        log("  - ho sake to clean terminal (no desktop) se chalao")
        log("  - tip: pkill -f 'code --type=gpu-process' ; pkill chrome ; pkill firefox\n")

    # CUDA allocator fragmentation guard (4GB pe OOM se bachata hai)
    if device_type == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()
        log(f"[mem] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
        log(f"[mem] idle before model:{cuda_mem()}")

    # 4GB profile: har step aggressive empty_cache (throughput thoda kam par no OOM).
    # Bade GPU / multi-GPU pe ye nahi chahiye (sirf time waste karta hai).
    aggressive_empty = (c.profile_name == "gpu_4gb")

    # precision: bf16 if GPU supports it, else fp16, CPU pe fp32
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device_type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type="cuda", dtype=amp_dtype))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    log(f"AMP dtype: {amp_dtype}")

    # ── Data ────────────────────────────────────────────────────────────────────
    train_loader, val_loader, vocab_size = get_bin_dataloaders(
        c.data_folder, c.batch_size, c.context_length, c.num_workers
    )
    log(f"vocab_size: {vocab_size}")

    # ── Model ───────────────────────────────────────────────────────────────────
    model_cfg = build_model_config(vars(c), vocab_size)
    model = MironLLM(model_cfg).to(device)

    # ── VRAM preflight: kitna chahiye vs kitna free (OOM se pehle heads-up) ────
    if device_type == "cuda":
        n_params = model.count_params()
        need_gb = estimate_fixed_vram_gb(n_params, c.optimizer_type)
        used_gb = total_gb - free_gb
        log(f"[vram] total {total_gb:.1f} GB | used {used_gb:.1f} GB | free {free_gb:.1f} GB")
        log(f"[vram] is run ka estimate ~{need_gb:.1f} GB "
            f"(weights+grads+optim; activations extra)")
        if need_gb > free_gb:
            log("[vram] WARNING: estimate free VRAM se zyada hai -> OOM ka strong risk.")
            log("       -> chhota profile (MIRON_PROFILE=gpu_4gb / cpu), ya config.py me "
                "is profile ka ctx/batch ghatao, ya doosri GPU apps band karo.")

    optimizer = model.configure_optimizer(
        c.lr, c.weight_decay, (c.beta1, c.beta2), device_type, c.optimizer_type
    )

    # ── Resume (load into the raw module BEFORE compile/DDP wrap) ──────────────
    start_step = 0
    best_val = float("inf")
    ckpt_path = f"{c.save_folder}/miron.pt"
    if Path(ckpt_path).exists():
        log(f"Resuming from {ckpt_path}...")
        ck = torch.load(ckpt_path, map_location=device)
        if "model" in ck:
            try:
                model.load_state_dict(ck["model"])
                optimizer.load_state_dict(ck["optimizer"])
                if scaler is not None and ck.get("scaler"):
                    scaler.load_state_dict(ck["scaler"])
                start_step = ck.get("step", 0)
                best_val = ck.get("best_val", float("inf"))
                log(f"  resumed at step {start_step} (best_val {best_val:.4f})")
            except (RuntimeError, ValueError, KeyError) as e:
                # Profile/architecture/optimizer badal gaya -> purana checkpoint
                # fit nahi hoga. Crash ke bajaye fresh training shuru karo.
                log(f"  checkpoint current profile se match nahi karta ({e})")
                log("  -> fresh training shuru kar rahe hain")
                start_step, best_val = 0, float("inf")
        else:
            log("  old-format checkpoint mila, incompatible architecture -> training fresh")

    # raw_model = clean handle for checkpointing (compile/DDP se pehle pakad lo,
    # taaki state_dict keys saaf rahein aur chat.py load kar sake).
    raw_model = model
    if c.compile_model and device_type == "cuda":
        log("Compiling model (first step will be slow)...")
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[dev_index] if device_type == "cuda" else None)

    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log(f"[mem] after model+optim load:{cuda_mem()}")

    eff_batch = c.batch_size * c.grad_accum * world_size
    log(f"\n{'='*56}")
    log("  TRAINING START")
    log(f"  steps {start_step} -> {c.max_steps}")
    log(f"  micro-batch {c.batch_size} x accum {c.grad_accum}"
        + (f" x {world_size} gpus" if is_ddp else "")
        + f" = {eff_batch} effective")
    log(f"  ctx {c.context_length} | tokens/step "
        f"{eff_batch * c.context_length:,}")
    log(f"{'='*56}\n")

    # First few steps log every step (so it's clear training isn't stuck on a
    # slow GPU); after that we log every `log_every` steps.
    dense_log_steps = 20
    log(f"First optimizer step can be slow (data priming / compile). "
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
        for micro in range(c.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # DDP: gradient sirf aakhri micro-step pe all-reduce karo. Baaki
            # accumulation steps no_sync() me -> bekaar network sync se bachte hain.
            last_micro = (micro == c.grad_accum - 1)
            sync_ctx = model.no_sync() if (is_ddp and not last_micro) else nullcontext()
            with sync_ctx:
                with ctx:
                    _, loss = model(x, y)
                    loss = loss / c.grad_accum
                scaler.scale(loss).backward()
            step_loss += loss.item()

            if aggressive_empty and (micro + 1) % 8 == 0:
                torch.cuda.empty_cache()

        if c.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if aggressive_empty:
            torch.cuda.empty_cache()

        window_loss += step_loss
        window_steps += 1

        # Dense logging for the first few steps, then every `log_every` steps.
        # Loss and throughput are both averaged over the current window, so the
        # window is always reset together after a log line is printed.
        if is_master and ((step + 1) % c.log_every == 0 or step < dense_log_steps):
            dt = max(1e-6, time.time() - t0)
            avg = window_loss / window_steps
            tokens = window_steps * eff_batch * c.context_length
            print(f"step {step+1:>6}/{c.max_steps} | loss {avg:.4f} | "
                  f"lr {lr:.2e} | ~{tokens / dt / 1e3:.1f}k tok/s{cuda_mem()}", flush=True)
            window_loss = 0.0
            window_steps = 0
            t0 = time.time()

        if (step + 1) % c.eval_every == 0:
            # eval har rank pe chalta hai (apni val windows pe); checkpoint master.
            val_loss = evaluate(model, val_loader, ctx, device, c.eval_iters)
            if is_master:
                print(f"  eval @ step {step+1}: val_loss {val_loss:.4f} "
                      f"(best {best_val:.4f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(f"{c.save_folder}/miron_best.pt", raw_model,
                              optimizer, scaler, step + 1, best_val, model_cfg, c)
                    print(f"  -> new best (val_loss {best_val:.4f}) saved")
            # best_val ko sab ranks pe sync rakho (warna logic diverge ho jaata)
            if is_ddp:
                bv = [best_val]
                dist.broadcast_object_list(bv, src=0)
                best_val = bv[0]
            if device_type == "cuda":
                torch.cuda.empty_cache()
            # eval / checkpoint time ko throughput me mat gino
            window_loss = 0.0
            window_steps = 0
            t0 = time.time()

        if is_master and (step + 1) % c.save_every == 0:
            save_ckpt(ckpt_path, raw_model, optimizer, scaler,
                      step + 1, best_val, model_cfg, c)
            if device_type == "cuda":
                torch.cuda.empty_cache()

    if is_master:
        save_ckpt(ckpt_path, raw_model, optimizer, scaler, c.max_steps, best_val, model_cfg, c)
        print(f"\nTraining complete. Checkpoints in {c.save_folder}/")

    if is_ddp:
        dist.destroy_process_group()


def _print_oom_help():
    print("\n" + "=" * 60)
    print("  CUDA OUT OF MEMORY — training saaf tarike se roki gayi")
    print("  (Linux pe raw crash/stack-trace ke bajaye yeh message).")
    print("  Model is GPU ki free VRAM me fit nahi hua. Try (asaan se mushkil):")
    print("   1) Doosri GPU apps band karo (browser/VSCode/Electron), phir dobara chalao")
    print("   2) Chhota profile :  MIRON_PROFILE=gpu_4gb python scripts/train.py   (ya cpu)")
    print("   3) core/config.py ke us profile ka context_length / batch_size ghatao")
    print("   4) Free VRAM dekho :  python -m core.config    ya    nvidia-smi")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train()
    except torch.cuda.OutOfMemoryError:
        _print_oom_help()
        sys.exit(1)
    except RuntimeError as e:
        # kuch OOM generic RuntimeError ke roop me aate hain
        if "out of memory" in str(e).lower():
            _print_oom_help()
            sys.exit(1)
        raise
