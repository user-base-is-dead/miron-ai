"""
─────────────────────────────────────────────────────────────────────────────
  SFT  —  instruction fine-tuning (base model ko "baat-cheet" sikhana)
─────────────────────────────────────────────────────────────────────────────
  train.py ek BASE model banata hai (raw text complete karta hai). SFT us base
  ke UPAR chalti hai aur use chat/instructions follow karna sikhati hai. Loss
  SIRF assistant ke jawab pe lagta hai (prepare_sft.py ka mask -> -100).

  Zaroori: pehle base train ho (saved_model/miron.pt ya miron_best.pt). SFT use
  load karke fine-tune karta hai aur ALAG file me save karta hai
  (saved_model/miron_sft.pt) — base overwrite NAHI hota.

  Workflow:
    1) python data-download/download_sft_data.py     (chat data)
    2) python scripts/prepare_sft.py                 (tokenize + mask)
    3) python scripts/sft.py                          (fine-tune)
       torchrun --nproc_per_node=N scripts/sft.py     (multi-GPU)

  Tuning — settings/sft_settings.json (base ke settings.json ki tarah, par
  SIRF SFT ke liye). Keys (clean, bina `sft_` prefix):
    max_steps, lr, min_lr, warmup_steps,
    eval_every, eval_iters, save_every, log_every,
    batch_size, grad_accum, optimizer_type, num_workers
    (env override: MIRON_SFT_MAX_STEPS, MIRON_SFT_LR, ... bade-case me)
    (purana settings.json `sft_max_steps` etc. abhi bhi fallback ke roop me chalega)
  Paths (env):
    MIRON_BASE_CKPT  -> base se load (default: saved_model/miron_best.pt|miron.pt)
    MIRON_SFT_DATA   -> SFT bins folder (default: data/tokenized_sft)
    MIRON_SFT_OUT    -> output checkpoint (default: saved_model/miron_sft.pt)
  batch/accum/optimizer/workers sft_settings.json se override ho sakte hain,
  warna profile (train.py jaisa) se aate hain; precision auto; model
  architecture + context length BASE checkpoint se aate hain.
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

# Repo root pe anchor -> 'core' import + default data/model paths CWD-independent.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.config import (format_hardware_report, get_active_config,
                         resolve_profile_name)
from core.dataset import get_sft_dataloaders
from core.miron_llm import Config, MironLLM


# ── SFT hyperparam defaults (sft_settings.json / settings.json `sft_*` / env) ─
SFT_DEFAULTS = dict(
    max_steps=2000,        # SFT chhoti hoti hai — 1-3 epochs over chat data
    lr=2e-5,               # base pretrain LR se kaafi kam (warna base "bhool" jaata)
    min_lr=2e-6,
    warmup_steps=50,
    eval_every=200,
    eval_iters=50,
    save_every=200,
    log_every=10,
)

# SFT ka apna dedicated settings file (base train ke settings.json ki tarah).
# settings.json sirf BASE training ke liye; sft_settings.json sirf SFT ke liye.
_SFT_SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings" / "sft_settings.json"
_SETTINGS_FILE     = Path(__file__).resolve().parent.parent / "settings" / "settings.json"


def _read_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_settings() -> tuple[dict, dict]:
    """(sft_settings.json, settings.json) dono padho.
    sft_settings.json -> SFT-dedicated (clean keys: max_steps, lr, ...).
    settings.json     -> purane `sft_` prefixed keys (backward-compat fallback)."""
    return _read_json(_SFT_SETTINGS_FILE), _read_json(_SETTINGS_FILE)


def _sft_param(key: str, cast, sft_settings: dict, settings: dict):
    """Priority: env MIRON_SFT_<KEY> > sft_settings.json `<key>`
    > settings.json `sft_<key>` (purana) > default."""
    env = os.environ.get("MIRON_SFT_" + key.upper())
    if env not in (None, ""):
        return cast(env)
    if key in sft_settings:                 # dedicated SFT file (clean key)
        return cast(sft_settings[key])
    skey = "sft_" + key
    if skey in settings:                    # purana settings.json `sft_` key
        return cast(settings[skey])
    return SFT_DEFAULTS[key]


# ── Distributed / device helpers (train.py jaise) ─────────────────────────────
def ddp_info():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (True, int(os.environ["RANK"]),
                int(os.environ.get("LOCAL_RANK", 0)), int(os.environ["WORLD_SIZE"]))
    return (False, 0, 0, 1)


def pick_best_gpu() -> int:
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
    if not torch.cuda.is_available():
        return ""
    try:
        alloc = torch.cuda.memory_allocated() / (1024 ** 2)
        reserv = torch.cuda.memory_reserved() / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return f" | GPU {alloc:.0f}/{reserv:.0f}MiB (peak {peak:.0f})"
    except Exception:
        return ""


def get_lr(step, s):
    """Cosine decay + linear warmup (train.py jaisa, par SFT schedule `s` pe)."""
    if step < s.warmup_steps:
        return s.lr * (step + 1) / max(1, s.warmup_steps)
    if step > s.max_steps:
        return s.min_lr
    ratio = (step - s.warmup_steps) / max(1, s.max_steps - s.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return s.min_lr + coeff * (s.lr - s.min_lr)


@torch.no_grad()
def evaluate(model, val_loader, amp_ctx, device, max_iters):
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(max_iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx:
            _, loss = model(x, y)
        if torch.isfinite(loss):           # poori-masked window -> NaN, skip
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_ckpt(path, raw_model, optimizer, scaler, step, best_val, model_cfg):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":     raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict() if scaler is not None else None,
        "step":      step,
        "best_val":  best_val,
        "model_cfg": model_cfg.to_dict(),
        "sft":       True,
    }, path)


def _resolve_base_ckpt(save_folder: str) -> Path | None:
    env = os.environ.get("MIRON_BASE_CKPT")
    if env:
        p = Path(env)
        return p if p.exists() else None
    for name in ("miron_best.pt", "miron.pt"):
        p = Path(save_folder) / name
        if p.exists():
            return p
    return None


def train_sft():
    is_ddp, rank, local_rank, world_size = ddp_info()
    is_master = (rank == 0)

    def log(*a, **k):
        if is_master:
            print(*a, **k)

    # ── Device selection ──────────────────────────────────────────────────────
    # NOTE: is_available() CUDA-built torch pe True ho sakta hai bhale hi koi
    # device visible na ho (e.g. CUDA_VISIBLE_DEVICES=""). device_count() bhi
    # check karo warna "Invalid device id" pe crash.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
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

    log(format_hardware_report())
    if is_ddp:
        log(f"Distributed   : DDP across {world_size} process(es)")
    log("-" * 60)

    # ── Profile (batch/accum/optimizer/precision ke liye) — train.py jaisa ────
    if is_ddp:
        name = resolve_profile_name() if is_master else None
        obj = [name]
        dist.broadcast_object_list(obj, src=0)
        c = SimpleNamespace(**get_active_config(name=obj[0]))
    else:
        c = SimpleNamespace(**get_active_config(free_gb=free_gb))

    # SFT settings load karo (sft_settings.json + purana settings.json).
    # batch/accum/optimizer/workers ko SFT ke liye override karo — base training
    # (settings.json) ko chhede bina. Profile/hardware defaults baaki same.
    sft_settings, settings = _load_settings()
    for _k, _cast in (("batch_size", int), ("grad_accum", int),
                      ("num_workers", int), ("optimizer_type", str)):
        if _k in sft_settings:
            setattr(c, _k, _cast(sft_settings[_k]))

    torch.manual_seed(c.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── SFT hyperparams (settings/sft_settings.json) ──────────────────────────
    # sft_settings/settings upar load ho chuke (profile block me).
    s = SimpleNamespace(
        max_steps=_sft_param("max_steps", int, sft_settings, settings),
        lr=_sft_param("lr", float, sft_settings, settings),
        min_lr=_sft_param("min_lr", float, sft_settings, settings),
        warmup_steps=_sft_param("warmup_steps", int, sft_settings, settings),
        eval_every=_sft_param("eval_every", int, sft_settings, settings),
        eval_iters=_sft_param("eval_iters", int, sft_settings, settings),
        save_every=_sft_param("save_every", int, sft_settings, settings),
        log_every=_sft_param("log_every", int, sft_settings, settings),
    )

    # DDP large-batch LR scaling (train.py jaisa sqrt rule)
    if is_ddp and world_size > 1:
        scale = world_size ** 0.5
        s.lr *= scale
        s.min_lr *= scale
        log(f"[lr] world_size={world_size} -> SFT LR x{scale:.2f}: lr={s.lr:.2e}")

    log(f"Profile : {c.profile_name}  (batch {c.batch_size} x accum {c.grad_accum})")
    log(f"Device  : {device}")

    if device_type == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.empty_cache()

    # precision: bf16 if supported, else fp16, CPU pe fp32 (train.py jaisa)
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device_type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    amp_ctx = (nullcontext() if device_type == "cpu"
               else torch.amp.autocast(device_type="cuda", dtype=amp_dtype))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    log(f"AMP dtype: {amp_dtype}")

    # ── Base checkpoint load (architecture + weights yahin se) ────────────────
    base_path = _resolve_base_ckpt(c.save_folder)
    if base_path is None:
        log("\n[sft] Base checkpoint nahi mila. SFT base model ke UPAR hoti hai.")
        log("      Pehle base train karo:")
        log("        python scripts/prepare_data.py")
        log("        python scripts/train.py")
        log("      (ya MIRON_BASE_CKPT=path se base ki location do)")
        if is_ddp:
            dist.destroy_process_group()
        sys.exit(1)

    log(f"Base    : loading {base_path}")
    ckpt = torch.load(base_path, map_location=device)
    if "model_cfg" not in ckpt or "model" not in ckpt:
        log("[sft] Base checkpoint purane/incompatible format ka hai. "
            "Dobara train karo: python scripts/train.py")
        if is_ddp:
            dist.destroy_process_group()
        sys.exit(1)

    model_cfg = Config.from_dict(ckpt["model_cfg"])
    ctx_len = model_cfg.context_length          # SFT ctx = base ctx (must match)
    model = MironLLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    log(f"  base loaded | {model.count_params()/1e6:.1f}M params | ctx {ctx_len} "
        f"| was step {ckpt.get('step', '?')}")

    # ── SFT data ──────────────────────────────────────────────────────────────
    sft_data = os.environ.get("MIRON_SFT_DATA", str(_ROOT / "data" / "tokenized_sft"))
    train_loader, val_loader, vocab_size = get_sft_dataloaders(
        sft_data, c.batch_size, ctx_len, c.num_workers
    )
    if vocab_size != model_cfg.vocab_size:
        log(f"[sft] vocab mismatch: data {vocab_size} vs base {model_cfg.vocab_size}. "
            "Same tokenizer se data banao (prepare_sft.py).")
        if is_ddp:
            dist.destroy_process_group()
        sys.exit(1)
    log(f"SFT data: {sft_data} | vocab {vocab_size}")

    # ── Optimizer (SFT LR) ──────────────────────────────────────────────────────
    optimizer = model.configure_optimizer(
        s.lr, c.weight_decay, (c.beta1, c.beta2), device_type, c.optimizer_type
    )

    # ── Resume SFT (agar pehle se chal raha tha) — base se ALAG file ──────────
    sft_out = Path(os.environ.get("MIRON_SFT_OUT", f"{c.save_folder}/miron_sft.pt"))
    best_out = sft_out.with_name(sft_out.stem + "_best" + sft_out.suffix)
    start_step, best_val = 0, float("inf")
    if sft_out.exists():
        log(f"Resuming SFT from {sft_out}...")
        sck = torch.load(sft_out, map_location=device)
        try:
            model.load_state_dict(sck["model"])
            optimizer.load_state_dict(sck["optimizer"])
            if scaler is not None and sck.get("scaler"):
                scaler.load_state_dict(sck["scaler"])
            start_step = sck.get("step", 0)
            best_val = sck.get("best_val", float("inf"))
            log(f"  resumed at SFT step {start_step} (best_val {best_val:.4f})")
        except (RuntimeError, ValueError, KeyError) as e:
            log(f"  SFT checkpoint match nahi hua ({e}) -> base se fresh SFT")
            start_step, best_val = 0, float("inf")

    raw_model = model
    if c.compile_model and device_type == "cuda":
        log("Compiling model (first step slow)...")
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[dev_index] if device_type == "cuda" else None)

    eff_batch = c.batch_size * c.grad_accum * world_size
    log(f"\n{'='*56}")
    log("  SFT START")
    log(f"  steps {start_step} -> {s.max_steps} | lr {s.lr:.2e} -> {s.min_lr:.2e}")
    log(f"  micro-batch {c.batch_size} x accum {c.grad_accum}"
        + (f" x {world_size} gpus" if is_ddp else "")
        + f" = {eff_batch} effective | ctx {ctx_len}")
    log(f"  save -> {sft_out}  (base safe rahega)")
    log(f"{'='*56}\n", flush=True)

    train_iter = iter(train_loader)
    model.train()
    t0 = time.time()
    window_loss = 0.0
    window_steps = 0
    dense_log_steps = 20

    for step in range(start_step, s.max_steps):
        lr = get_lr(step, s)
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

            last_micro = (micro == c.grad_accum - 1)
            sync_ctx = model.no_sync() if (is_ddp and not last_micro) else nullcontext()
            with sync_ctx:
                with amp_ctx:
                    _, loss = model(x, y)
                    loss = loss / c.grad_accum
                if not torch.isfinite(loss):
                    # poora-masked batch (saare targets -100) -> NaN; skip backward
                    continue
                scaler.scale(loss).backward()
            step_loss += loss.item()

        if c.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        window_loss += step_loss
        window_steps += 1

        if is_master and ((step + 1) % s.log_every == 0 or step < dense_log_steps):
            dt = max(1e-6, time.time() - t0)
            avg = window_loss / max(1, window_steps)
            tokens = window_steps * eff_batch * ctx_len
            print(f"step {step+1:>6}/{s.max_steps} | loss {avg:.4f} | "
                  f"lr {lr:.2e} | ~{tokens / dt / 1e3:.1f}k tok/s{cuda_mem()}",
                  flush=True)
            window_loss = 0.0
            window_steps = 0
            t0 = time.time()

        if (step + 1) % s.eval_every == 0:
            val_loss = evaluate(model, val_loader, amp_ctx, device, s.eval_iters)
            if is_master:
                print(f"  eval @ step {step+1}: val_loss {val_loss:.4f} "
                      f"(best {best_val:.4f})")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(best_out, raw_model, optimizer, scaler,
                              step + 1, best_val, model_cfg)
                    print(f"  -> new best (val_loss {best_val:.4f}) saved -> {best_out}")
            if is_ddp:
                bv = [best_val]
                dist.broadcast_object_list(bv, src=0)
                best_val = bv[0]
            if device_type == "cuda":
                torch.cuda.empty_cache()
            window_loss = 0.0
            window_steps = 0
            t0 = time.time()

        if is_master and (step + 1) % s.save_every == 0:
            save_ckpt(sft_out, raw_model, optimizer, scaler,
                      step + 1, best_val, model_cfg)

    if is_master:
        save_ckpt(sft_out, raw_model, optimizer, scaler,
                  s.max_steps, best_val, model_cfg)
        print(f"\nSFT complete. Checkpoint: {sft_out}")
        print("Chat karne ke liye: chat.py ko is checkpoint pe point karo "
              "(ya miron_sft_best.pt ko miron_best.pt ki jagah copy karo).")

    if is_ddp:
        dist.destroy_process_group()


def _print_oom_help():
    print("\n" + "=" * 60)
    print("  CUDA OUT OF MEMORY — SFT saaf tarike se roki gayi")
    print("  Model is GPU ki free VRAM me fit nahi hua. Try (asaan se mushkil):")
    print("   1) Doosri GPU apps band karo (browser/VSCode), phir dobara chalao")
    print("   2) Chhota profile :  MIRON_PROFILE=gpu_4gb python scripts/sft.py  (ya cpu)")
    print("   3) settings.json me batch_size / grad_accum ghatao")
    print("   4) Free VRAM dekho :  python -m core.config")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train_sft()
    except torch.cuda.OutOfMemoryError:
        _print_oom_help()
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _print_oom_help()
            sys.exit(1)
        raise
