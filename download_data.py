"""
MANINMIRON - Huge Dataset Downloader
Automatically sab datasets download karta hai data/ folder mein
"""

from datasets import load_dataset
import os
import re
from pathlib import Path

DATA_FOLDER = "data"
Path(DATA_FOLDER).mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def download_wikipedia():
    print("\n📚 [1/3] Wikipedia English download ho raha hai (100K articles ~1GB)...")
    output_file = f"{DATA_FOLDER}/wikipedia_en.txt"
    count = 0
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for article in dataset:
            text = clean_text(article["text"])
            if len(text) > 300:
                f.write(text + "\n\n" + "=" * 60 + "\n\n")
                count += 1
                if count % 5000 == 0:
                    size = os.path.getsize(output_file) / (1024*1024)
                    print(f"   {count}/100000 | {size:.0f} MB")
                if count >= 100000:
                    break
    size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"✅ Wikipedia done → {size_mb:.0f} MB")


def download_openwebtext():
    print("\n🌐 [2/3] OpenWebText download ho raha hai (200K samples ~2GB)...")
    output_file = f"{DATA_FOLDER}/openwebtext.txt"
    count = 0
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            text = clean_text(sample["text"])
            if len(text) > 200:
                f.write(text + "\n\n")
                count += 1
                if count % 10000 == 0:
                    size = os.path.getsize(output_file) / (1024*1024)
                    print(f"   {count}/200000 | {size:.0f} MB")
                if count >= 200000:
                    break
    size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"✅ OpenWebText done → {size_mb:.0f} MB")


def download_books():
    print("\n📖 [3/3] Books download ho raha hai (100K lines ~500MB)...")
    output_file = f"{DATA_FOLDER}/books.txt"
    count = 0
    dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            text = clean_text(sample["text"])
            if len(text) > 50:
                f.write(text + "\n")
                count += 1
                if count % 10000 == 0:
                    size = os.path.getsize(output_file) / (1024*1024)
                    print(f"   {count}/100000 | {size:.0f} MB")
                if count >= 100000:
                    break
    size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"✅ Books done → {size_mb:.0f} MB")


if __name__ == "__main__":
    print("=" * 55)
    print("  MANINMIRON - Full Dataset Download")
    print("  Wikipedia + OpenWebText + Books = ~3.5 GB")
    print("=" * 55)

    download_wikipedia()
    download_openwebtext()
    download_books()

    print()
    print("=" * 55)
    print("  ✅ Sab datasets download complete!")
    print()
    print("  data/")
    print("  ├── wikipedia_en.txt  ~1 GB")
    print("  ├── openwebtext.txt   ~2 GB")
    print("  └── books.txt         ~500 MB")
    print()
    print("  Ab train karo: python train.py")
    print("=" * 55)
