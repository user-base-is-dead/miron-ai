"""
─────────────────────────────────────────────────────────────────────────────
  CONFIG  —  Single source of truth for the whole project
─────────────────────────────────────────────────────────────────────────────
  Har device ka apna "profile" hai. Ek profile me model ka size AUR training
  ke settings dono hote hain. Naye device pe bas profile switch karo —
  model aur training apne aap us hardware ke hisaab se set ho jayenge.

  Profile kaise chunta hai (priority order):
    1. Environment variable   ->  MIRON_PROFILE=gpu_8gb python train.py
    2. ACTIVE_PROFILE constant ->  neeche "gpu_4gb" likh do
    3. "auto" (default)        ->  GPU ki VRAM padh ke khud chun leta hai

  vocab_size yahan NAHI hai — wo training ke waqt data/meta.json se aata hai.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os


# ── Yahan badlo: kaunsa profile use karna hai ────────────────────────────────
# "auto"  = GPU dekh ke khud chun lega (recommended)
# ya seedha likho: "cpu", "gpu_4gb", "gpu_8gb", "gpu_12gb", "gpu_24gb", "gpu_40gb"
ACTIVE_PROFILE = "auto"


# ── Model Config ke fields (maninmiron_llm.Config se match karte hain) ────────
# vocab_size yahan jaan-bujh ke nahi — wo meta.json se runtime pe aata hai.
MODEL_FIELDS = {
    "context_length", "d_model", "num_heads", "num_kv_heads", "num_layers",
    "d_ff", "dropout", "rope_theta", "norm_eps", "tie_weights", "grad_checkpoint",
}


# ── Har profile me common cheezein (taaki har profile chhota rahe) ────────────
DEFAULTS = dict(
    data_folder="data",
    save_folder="saved_model",
    # LR schedule
    warmup_steps=200,
    min_lr=6e-5,
    # optimizer extras
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    # eval / logging / saving
    eval_every=500,
    eval_iters=50,
    log_every=10,
    save_every=500,
    seed=1337,
    # model defaults (profile inhe override kar sakta hai)
    dropout=0.0,
    rope_theta=10000.0,
    norm_eps=1e-5,
    tie_weights=True,
)


# ── Device Profiles (chhote se bade ke order me) ──────────────────────────────
# min_vram_gb = auto-detect ke liye: itni VRAM ho to ye profile chal sakta hai.
PROFILES = {
    # CPU / GPU-less smoke test. Chhota model bas ye check karne ke liye ki
    # pipeline chal raha hai. Embedding (100263 rows) ki wajah se ~27M se neeche
    # nahi ja sakte bina tokenizer badle.
    "cpu": dict(
        min_vram_gb=0,
        context_length=256, d_model=256, num_heads=4, num_kv_heads=2,
        num_layers=4, d_ff=768, grad_checkpoint=False,
        batch_size=8, grad_accum=8, max_steps=2000, lr=6e-4,
        optimizer_type="adamw", compile_model=False, num_workers=2,
    ),

    # Tera RTX 3050 Laptop (4GB). Bahut tight zone hai.
    # paged_adamw_8bit + grad_checkpoint=True + expandable_segments zaroori.
    # GNOME + desktop already ~200-400MiB kha lete hain.
    # 152M model (L12 d768) ka base load hi ~1.77GB le leta hai -> 4GB pe almost no headroom.
    # Model ko chhota kiya: L10 d640 h10/kv2 (strong GQA) + ctx256.
    # Params ~115-125M range expected. Base load ~1.3-1.5GB target.
    # Agar phir bhi OOM aaye to ctx=128 kar denge.
    "gpu_4gb": dict(
        min_vram_gb=3.5,
        context_length=256, d_model=640, num_heads=10, num_kv_heads=2,
        num_layers=10, d_ff=1792, grad_checkpoint=True,
        batch_size=1, grad_accum=64, max_steps=20000, lr=6e-4,
        optimizer_type="paged_adamw_8bit", compile_model=False, num_workers=2,
    ),

    # 8GB GPU: wahi 152M, par ctx1024 + batch2. On-GPU 8-bit (paging nahi -> tez).
    "gpu_8gb": dict(
        min_vram_gb=7.0,
        context_length=1024, d_model=768, num_heads=12, num_kv_heads=4,
        num_layers=12, d_ff=2048, grad_checkpoint=True,
        batch_size=2, grad_accum=32, max_steps=40000, lr=6e-4,
        optimizer_type="adam8bit", compile_model=False, num_workers=4,
    ),

    # 12GB GPU: bada model (~283M), full fp32 AdamW (fused) + torch.compile.
    "gpu_12gb": dict(
        min_vram_gb=11.0,
        context_length=1024, d_model=1024, num_heads=16, num_kv_heads=4,
        num_layers=16, d_ff=2816, grad_checkpoint=True,
        batch_size=4, grad_accum=24, max_steps=60000, lr=4e-4, min_lr=4e-5,
        optimizer_type="adamw", compile_model=True, num_workers=4,
    ),

    # 24GB GPU (3090/4090): ~374M model.
    "gpu_24gb": dict(
        min_vram_gb=23.0,
        context_length=1024, d_model=1024, num_heads=16, num_kv_heads=4,
        num_layers=24, d_ff=2816, grad_checkpoint=True,
        batch_size=8, grad_accum=16, max_steps=100000, lr=3e-4, min_lr=3e-5,
        optimizer_type="adamw", compile_model=True, num_workers=8,
    ),

    # 40GB+ GPU (A100): ~750M model. L28/d2048 karke ~1B tak le ja sakte hain.
    "gpu_40gb": dict(
        min_vram_gb=39.0,
        context_length=2048, d_model=1536, num_heads=16, num_kv_heads=4,
        num_layers=24, d_ff=4096, grad_checkpoint=True,
        batch_size=8, grad_accum=24, max_steps=150000, lr=3e-4, min_lr=3e-5,
        optimizer_type="adamw", compile_model=True, num_workers=8,
    ),
}


# ── Hardware detection ────────────────────────────────────────────────────────
def detect_hardware() -> dict:
    """Compute hardware ka snapshot: kitne GPU, har ek ka naam + total/free VRAM,
    CUDA/driver available hai ya nahi, aur CPU thread count. Startup summary aur
    free-VRAM-based profile selection dono yahin se chalte hain.
    """
    info = {
        "cuda_available": False,
        "torch_cuda_version": None,
        "num_gpus": 0,
        "gpus": [],                       # [{index, name, total_gb, free_gb}, ...]
        "cpu_count": os.cpu_count() or 1,
    }
    try:
        import torch
    except ImportError:
        return info

    info["torch_cuda_version"] = torch.version.cuda
    if not torch.cuda.is_available():
        return info

    info["cuda_available"] = True
    info["num_gpus"] = torch.cuda.device_count()
    for i in range(info["num_gpus"]):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            # mem_get_info kuch setups me fail ho sakta hai -> total pe fallback
            free, total = props.total_memory, props.total_memory
        info["gpus"].append({
            "index": i,
            "name": props.name,
            "total_gb": total / 1e9,
            "free_gb": free / 1e9,
        })
    return info


def format_hardware_report() -> str:
    """detect_hardware() ko insaan-padhne-layak summary me badalta hai."""
    hw = detect_hardware()
    lines = []
    if hw["cuda_available"]:
        lines.append(f"CUDA          : available (torch built for CUDA {hw['torch_cuda_version']})")
        lines.append(f"GPUs detected : {hw['num_gpus']}")
        for g in hw["gpus"]:
            used = g["total_gb"] - g["free_gb"]
            lines.append(
                f"  [cuda:{g['index']}] {g['name']} | "
                f"{g['free_gb']:.1f} free / {used:.1f} used / {g['total_gb']:.1f} GB total"
            )
    else:
        if hw["torch_cuda_version"] is None:
            why = " (torch CPU-only build hai — GPU chahiye to CUDA wheel install karo)"
        else:
            why = " (NVIDIA driver/GPU nahi mila — `nvidia-smi` check karo)"
        lines.append(f"CUDA          : NOT available{why}")
        lines.append(f"Mode          : CPU ({hw['cpu_count']} threads)")
    return "\n".join(lines)


# ── Profile resolution ────────────────────────────────────────────────────────
def _detect_profile(free_gb: float | None = None) -> str:
    """Available VRAM dekh ke sabse bada profile chuno jo us memory me fit ho.

    TOTAL nahi, FREE memory use karte hain — taaki agar koi aur process (browser,
    desktop, doosri training) pehle se VRAM kha raha ho to bhi safe profile mile.
    free_gb na do to SAARE GPU me se sabse KAM free wala liya jaata hai (multi-GPU
    DDP ke liye safe: model har card pe fit hoga).
    """
    hw = detect_hardware()
    if not hw["cuda_available"] or hw["num_gpus"] == 0:
        return "cpu"

    if free_gb is None:
        free_gb = min(g["free_gb"] for g in hw["gpus"])

    best = "cpu"
    for name, p in sorted(PROFILES.items(), key=lambda kv: kv[1]["min_vram_gb"]):
        if name != "cpu" and free_gb >= p["min_vram_gb"]:
            best = name                    # sabse bada jo available VRAM me fit ho
    return best


def resolve_profile_name(free_gb: float | None = None) -> str:
    """Priority: env var MIRON_PROFILE > ACTIVE_PROFILE constant > auto-detect."""
    env = os.environ.get("MIRON_PROFILE")
    if env:
        if env not in PROFILES:
            raise ValueError(
                f"MIRON_PROFILE={env!r} galat hai. Valid: {list(PROFILES)}"
            )
        return env
    if ACTIVE_PROFILE and ACTIVE_PROFILE != "auto":
        if ACTIVE_PROFILE not in PROFILES:
            raise ValueError(
                f"ACTIVE_PROFILE={ACTIVE_PROFILE!r} galat hai. Valid: {list(PROFILES)}"
            )
        return ACTIVE_PROFILE
    return _detect_profile(free_gb)


def get_active_config(name: str | None = None, free_gb: float | None = None) -> dict:
    """Resolved profile ko DEFAULTS ke saath merge karke flat dict deta hai.
    `name` diya ho to wahi use hota hai (DDP me rank-0 decide karke baaki ranks
    ko broadcast karta hai). profile ki value DEFAULTS ko override karti hai
    (e.g. gpu_12gb ka min_lr)."""
    name = name or resolve_profile_name(free_gb)
    cfg = {**DEFAULTS, **PROFILES[name]}
    cfg.pop("min_vram_gb", None)               # ye sirf auto-detect ke liye tha
    cfg["profile_name"] = name
    return cfg


def build_model_config(cfg: dict, vocab_size: int):
    """Flat config dict + meta.json ka vocab_size -> maninmiron_llm.Config."""
    from maninmiron_llm import Config
    fields = {k: cfg[k] for k in MODEL_FIELDS if k in cfg}
    return Config(vocab_size=vocab_size, **fields)


# ── Quick check: `python config.py` chala ke dekh lo hardware + kaunsa profile ─
if __name__ == "__main__":
    print(format_hardware_report())
    print("-" * 60)
    cfg = get_active_config()
    print(f"Resolved profile : {cfg['profile_name']}")
    print(f"Optimizer        : {cfg['optimizer_type']}")
    print(f"Model            : d{cfg['d_model']} L{cfg['num_layers']} "
          f"h{cfg['num_heads']}/kv{cfg['num_kv_heads']} ctx{cfg['context_length']}")
    print(f"Batch            : {cfg['batch_size']} x accum {cfg['grad_accum']} "
          f"= {cfg['batch_size'] * cfg['grad_accum']} effective")
