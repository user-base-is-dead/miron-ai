"""
─────────────────────────────────────────────────────────────────────────────
  DOWNLOAD DATA  —  raw text corpora ko data/ folder me download karta hai
─────────────────────────────────────────────────────────────────────────────
  HuggingFace datasets se streaming mode me text kheech ke plain .txt likhta
  hai (poora dataset RAM/disk me nahi aata). Har source ka apna config neeche
  SOURCES list me hai — naya source add karna ho to bas ek dict aur daal do.

  Run:  python download_data.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from pathlib import Path

from datasets import load_dataset

DATA_FOLDER = "data"
Path(DATA_FOLDER).mkdir(exist_ok=True)


# ── Sources ──────────────────────────────────────────────────────────────────
# Har source ke fields:
#   name        -> log me dikhne ka label
#   path        -> HuggingFace dataset id
#   config      -> dataset config name (jis source me na ho wahan None)
#   min_chars   -> isse chhoti cleaned entries skip
#   max_samples -> itni entries ke baad ruk jao
#   output      -> data/ ke andar output file ka naam
#   separator   -> har entry ke baad file me likhne wala separator
SOURCES = [
    {
        "name": "Wikipedia (English)",
        "path": "wikimedia/wikipedia",
        "config": "20231101.en",
        "min_chars": 300,
        "max_samples": 100_000,
        "output": "wikipedia_en.txt",
        "separator": "\n\n" + "=" * 60 + "\n\n",
    },
    {
        "name": "OpenWebText",
        "path": "Skylion007/openwebtext",
        "config": None,
        "min_chars": 200,
        "max_samples": 200_000,
        "output": "openwebtext.txt",
        "separator": "\n\n",
    },
    {
        "name": "BookCorpus",
        "path": "bookcorpus/bookcorpus",
        "config": None,
        "min_chars": 50,
        "max_samples": 100_000,
        "output": "books.txt",
        "separator": "\n",
    },
]


def clean_text(text: str) -> str:
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_source(spec: dict) -> None:
    """Ek source ko stream karke uske output .txt me likhta hai."""
    output_file = f"{DATA_FOLDER}/{spec['output']}"
    print(f"\n[{spec['name']}] {spec['max_samples']:,} samples download ho rahe hain...")

    # config tabhi pass karo jab source ko zaroorat ho
    args = [spec["path"]] + ([spec["config"]] if spec["config"] else [])
    dataset = load_dataset(*args, split="train", streaming=True)

    text_field = spec.get("text_field", "text")
    log_every = max(1, spec["max_samples"] // 20)   # ~20 progress lines per source
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            text = clean_text(sample[text_field])
            if len(text) < spec["min_chars"]:
                continue
            f.write(text + spec["separator"])
            count += 1
            if count % log_every == 0:
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"   {count:,}/{spec['max_samples']:,} | {size_mb:.0f} MB")
            if count >= spec["max_samples"]:
                break

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[{spec['name']}] done -> {output_file} ({size_mb:.0f} MB)")


def main() -> None:
    print("=" * 55)
    print("  MANINMIRON - Full Dataset Download")
    print("  " + " + ".join(s["name"] for s in SOURCES))
    print("=" * 55)

    for spec in SOURCES:
        download_source(spec)

    print("\n" + "=" * 55)
    print("  Sab datasets download complete! Files:")
    for spec in SOURCES:
        print(f"    data/{spec['output']}")
    print("\n  Ab tokenize karo:  python prepare_data.py")
    print("  Phir train karo :  python train.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
