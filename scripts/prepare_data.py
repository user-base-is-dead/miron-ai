"""
─────────────────────────────────────────────────────────────────────────────
  PREPARE DATA  —  tokenize raw .txt -> binary memmap (.bin)
─────────────────────────────────────────────────────────────────────────────
  Bade text files (GBs) ko har epoch tokenize karna bahut slow + RAM-heavy hai.
  Isliye ek hi baar tokenize karke uint16/uint32 .bin file bana lete hain, jise
  training ke time np.memmap se bina RAM bhare padha jaa sakta hai.

  Input :  data/downloaded-data/*.txt        (download_data.py se aaya raw text)
  Output:  data/tokenized/{train.bin, val.bin, meta.json}

  Run:  python scripts/prepare_data.py
"""

import json
import os
import sys
from pathlib import Path

# ── Python version guard (numpy/tqdm import se pehle saaf error) ─────────────
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"[Miron] Python 3.11.x chahiye (abhi {sys.version.split()[0]} chal raha hai).\n"
        "        venv activate karo -> Windows: Miron311\\Scripts\\activate"
        "  |  Linux/Mac: source Miron311/bin/activate"
    )

import numpy as np
from tqdm import tqdm

# Repo root ko sys.path pe daalo taaki 'core' package import ho sake (yeh file
# scripts/ ke andar hai).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.tokenizer import MironTokenizer

RAW_FOLDER   = "data/downloaded-data"   # raw .txt yahan se (download_data.py output)
OUT_FOLDER   = "data/tokenized"         # bins + meta yahan (training yahin se padhta hai)
TRAIN_BIN    = f"{OUT_FOLDER}/train.bin"
VAL_BIN      = f"{OUT_FOLDER}/val.bin"
META_FILE    = f"{OUT_FOLDER}/meta.json"
VAL_FRACTION = 0.01            # 1% held out for validation
CHUNK_CHARS  = 2_000_000       # read text in ~2MB chunks to bound memory


def main():
    tok = MironTokenizer()
    # cl100k_base max id < 100263 -> fits in uint32 (uint16 would overflow)
    dtype = np.uint32

    txt_files = sorted(Path(RAW_FOLDER).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files in {RAW_FOLDER}/. "
            f"Pehle data download karo:  python data-download/download_data.py"
        )

    Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)
    print(f"Found {len(txt_files)} file(s): {[f.name for f in txt_files]}")

    # First pass: stream-tokenize everything into a temp flat .bin (train candidate),
    # then split. We write directly and split by index afterwards.
    tmp_bin = f"{OUT_FOLDER}/_all.bin"
    total_tokens = 0

    with open(tmp_bin, "wb") as out:
        for fpath in txt_files:
            size = os.path.getsize(fpath)
            print(f"\nTokenizing {fpath.name} ({size/1e6:.0f} MB)...")
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                pbar = tqdm(total=size, unit="B", unit_scale=True)
                buf = ""
                while True:
                    chunk = f.read(CHUNK_CHARS)
                    if not chunk:
                        break
                    pbar.update(len(chunk.encode("utf-8", errors="ignore")))
                    buf += chunk
                    # tokenize up to the last newline to avoid splitting tokens badly
                    cut = buf.rfind("\n")
                    if cut == -1:
                        continue
                    piece, buf = buf[:cut], buf[cut + 1:]
                    ids = tok.enc.encode_ordinary(piece)
                    np.asarray(ids, dtype=dtype).tofile(out)
                    total_tokens += len(ids)
                if buf:
                    ids = tok.enc.encode_ordinary(buf)
                    np.asarray(ids, dtype=dtype).tofile(out)
                    total_tokens += len(ids)
                pbar.close()

    print(f"\nTotal tokens: {total_tokens:,}")

    # Split into train / val via memmap (no full load into RAM)
    all_mm = np.memmap(tmp_bin, dtype=dtype, mode="r", shape=(total_tokens,))
    n_val = int(total_tokens * VAL_FRACTION)
    n_train = total_tokens - n_val

    print(f"Writing {TRAIN_BIN} ({n_train:,} tokens) and {VAL_BIN} ({n_val:,} tokens)...")
    train_mm = np.memmap(TRAIN_BIN, dtype=dtype, mode="w+", shape=(n_train,))
    val_mm   = np.memmap(VAL_BIN,   dtype=dtype, mode="w+", shape=(n_val,))

    # copy in blocks
    block = 10_000_000
    for i in tqdm(range(0, n_train, block), desc="train"):
        j = min(i + block, n_train)
        train_mm[i:j] = all_mm[i:j]
    for i in tqdm(range(0, n_val, block), desc="val"):
        j = min(i + block, n_val)
        val_mm[i:j] = all_mm[n_train + i: n_train + j]

    train_mm.flush(); val_mm.flush()
    del all_mm, train_mm, val_mm
    os.remove(tmp_bin)

    meta = {
        "dtype": "uint32",
        "vocab_size": tok.vocab_size,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "total_tokens": total_tokens,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {META_FILE} written.")
    print(f"  train.bin : {n_train:,} tokens")
    print(f"  val.bin   : {n_val:,} tokens")
    print("\nAb train karo:  python scripts/train.py")


if __name__ == "__main__":
    main()
