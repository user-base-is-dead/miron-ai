"""
─────────────────────────────────────────────────────────────────────────────
  DOWNLOAD SFT DATA  —  chat/instruction data ko data/downloaded-data/ me daalta hai
─────────────────────────────────────────────────────────────────────────────
  Pretraining data (download_data.py) raw text deta hai. SFT ke liye humein
  (user -> assistant) conversation pairs chahiye. Yeh script HuggingFace se
  high-quality, minimal-refusal instruction datasets STREAMING me kheech ke ek
  unified JSONL chat format me likhta hai (poora dataset RAM/disk me nahi aata).

  Har line ek example:
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

  Run (repo root se):
    python data-download/download_sft_data.py                  # default (openhermes)
    python data-download/download_sft_data.py dolly            # ek specific dataset
    python data-download/download_sft_data.py openhermes dolly # ek se zyada
─────────────────────────────────────────────────────────────────────────────
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset

# Repo root = is file ke folder (data-download/) ka parent -> output CWD pe
# depend nahi karta. SFT jsonl bhi raw download ke saath data/downloaded-data/ me.
_ROOT = Path(__file__).resolve().parent.parent
OUT_FOLDER = str(_ROOT / "data" / "downloaded-data")
Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)


# Alag-alag datasets role ke liye alag naam use karte hain -> normalize
_ROLE_MAP = {
    "human": "user", "user": "user", "prompter": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
    "system": "system",
}


def row_to_messages(row: dict):
    """Kisi bhi common instruction-dataset row ko unified messages list me
    badalta hai. Samajh na aaye to None (us row ko skip kar dete hain).
    """
    # 1) ShareGPT style: {"conversations": [{"from": "...", "value": "..."}]}
    convs = row.get("conversations")
    if isinstance(convs, list) and convs:
        out = []
        for turn in convs:
            role = _ROLE_MAP.get(str(turn.get("from", "")).lower())
            content = turn.get("value")
            if role and content:
                out.append({"role": role, "content": content})
        return out or None

    # 2) Already chat style: {"messages": [{"role": "...", "content": "..."}]}
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            role = _ROLE_MAP.get(str(m.get("role", "")).lower())
            content = m.get("content")
            if role and content:
                out.append({"role": role, "content": content})
        return out or None

    # 3) Instruction style: {"instruction", "input"/"context", "output"/"response"}
    instr = row.get("instruction")
    resp = row.get("output") or row.get("response")
    if instr and resp:
        extra = row.get("input") or row.get("context") or ""
        user = f"{instr}\n\n{extra}".strip() if extra else instr
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": resp}]

    return None


# name -> dataset spec. Sab high-quality + minimal corporate-refusal hain.
SOURCES = {
    # Recommended: bada + high quality + bekaar ke refusals nahi
    "openhermes": {"path": "teknium/OpenHermes-2.5", "config": None,
                   "split": "train", "max_samples": 100_000},
    # Bada multi-turn chat, quality-filtered
    "ultrachat":  {"path": "HuggingFaceH4/ultrachat_200k", "config": None,
                   "split": "train_sft", "max_samples": 100_000},
    # Chhota, human-written -> quick test ke liye accha
    "dolly":      {"path": "databricks/databricks-dolly-15k", "config": None,
                   "split": "train", "max_samples": None},
}

DEFAULT = ["openhermes"]


def download_one(name: str) -> None:
    spec = SOURCES[name]
    out_path = f"{OUT_FOLDER}/{name}.jsonl"
    split = spec.get("split", "train")
    cap = spec["max_samples"]
    print(f"\n[{name}] {spec['path']} ({split}) download shuru"
          + (f", cap {cap:,}" if cap else "") + "...")

    args = [spec["path"]] + ([spec["config"]] if spec["config"] else [])
    ds = load_dataset(*args, split=split, streaming=True)

    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            try:
                msgs = row_to_messages(row)
            except Exception:
                msgs = None
            if not msgs:
                skipped += 1
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            written += 1
            if written % 5000 == 0:
                print(f"   {written:,} likhe...")
            if cap and written >= cap:
                break

    print(f"[{name}] done -> {out_path} ({written:,} examples, {skipped:,} skipped)")


def main(argv) -> None:
    names = argv or DEFAULT
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"Unknown dataset(s): {unknown}")
        print(f"Available: {list(SOURCES)}")
        sys.exit(1)

    print("=" * 60)
    print("  SFT DATA DOWNLOAD")
    print(f"  Available : {list(SOURCES)}")
    print(f"  Is run me : {names}")
    print("=" * 60)

    for name in names:
        download_one(name)

    print("\n" + "=" * 60)
    print("  Done. Files data/downloaded-data/ me hain (JSONL chat format).")
    print("  Agla step: SFT training script (sft.py) — bol to bana doon.")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
