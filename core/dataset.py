"""
─────────────────────────────────────────────────────────────────────────────
  DATASET  —  memory-mapped binary token loader
─────────────────────────────────────────────────────────────────────────────
  Reads the .bin files produced by prepare_data.py using np.memmap, so even
  multi-GB corpora never get fully loaded into RAM. Each __getitem__ returns a
  random (x, y) window of length context_length for next-token prediction.

  Two modes:
    • get_bin_dataloaders(...)  -> fast path, needs `python scripts/prepare_data.py` first
    • get_dataloader(...)       -> legacy fallback that tokenizes raw .txt in RAM
─────────────────────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DTYPE_MAP = {"uint16": np.uint16, "uint32": np.uint32}


class BinDataset(Dataset):
    """Random fixed-length windows sampled from a token .bin memmap."""

    def __init__(self, bin_path: str, context_length: int, dtype=np.uint32,
                 samples_per_epoch: int | None = None):
        self.bin_path = bin_path
        self.context_length = context_length
        self.dtype = dtype
        # lazily opened per-worker to stay fork/spawn safe
        self._data = None
        self.n_tokens = Path(bin_path).stat().st_size // np.dtype(dtype).itemsize
        self.max_start = self.n_tokens - context_length - 1
        assert self.max_start > 0, f"{bin_path} too small for context_length={context_length}"
        # one "epoch" = scan-equivalent number of windows unless overridden
        self.length = samples_per_epoch or (self.n_tokens // context_length)

    def _mm(self):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        return self._data

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        data = self._mm()
        i = np.random.randint(0, self.max_start)
        chunk = np.asarray(data[i: i + self.context_length + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def _seed_worker(worker_id: int) -> None:
    # PyTorch har worker ke python/torch RNG ko to per-worker seed karta hai,
    # par numpy ko nahi. Bina iske saare workers ek jaisi random windows
    # nikaalte hain (duplicate batches -> kharab training). Worker ke andar
    # torch.initial_seed() already per-worker unique hota hai; usse numpy seed.
    np.random.seed(torch.initial_seed() % 2**32)


def get_bin_dataloaders(data_folder: str, batch_size: int, context_length: int,
                        num_workers: int = 2):
    folder = Path(data_folder)
    meta_path = folder / "meta.json"
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{meta_path} missing ya empty hai. "
            f"Pehle data taiyaar karo:  python scripts/prepare_data.py"
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{meta_path} corrupt JSON hai ({e}). "
            f"Dobara banao:  python scripts/prepare_data.py"
        ) from e
    if meta.get("dtype") not in DTYPE_MAP:
        raise ValueError(f"meta.json me galat/missing 'dtype': {meta.get('dtype')!r}")
    dtype = DTYPE_MAP[meta["dtype"]]

    for fname in ("train.bin", "val.bin"):
        p = folder / fname
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(
                f"{p} missing ya empty hai. "
                f"Pehle data taiyaar karo:  python scripts/prepare_data.py"
            )

    train_ds = BinDataset(str(folder / "train.bin"), context_length, dtype)
    val_ds = BinDataset(str(folder / "val.bin"), context_length, dtype)
    val_ds.length = min(200, val_ds.length)   # validation pass ko chhota rakho

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, drop_last=True,
                  persistent_workers=num_workers > 0,
                  worker_init_fn=_seed_worker)
    train_loader = DataLoader(train_ds, shuffle=False, **common)
    val_loader   = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, meta["vocab_size"]


# ── Legacy in-RAM fallback (small datasets only) ──────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data_folder: str, context_length: int = 512):
        from core.tokenizer import ManinmironTokenizer
        self.tokenizer = ManinmironTokenizer()
        self.context_length = context_length

        txt_files = list(Path(data_folder).glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt file in {data_folder}")

        all_text = ""
        for f in txt_files:
            print(f"  Reading -> {f.name}")
            all_text += f.read_text(encoding="utf-8", errors="ignore") + "\n"

        print(f"Total text: {len(all_text):,} chars | tokenizing...")
        self.tokens = self.tokenizer.enc.encode_ordinary(all_text)
        print(f"Total tokens: {len(self.tokens):,}")

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        s = idx * self.context_length
        e = s + self.context_length
        x = torch.tensor(self.tokens[s:e], dtype=torch.long)
        y = torch.tensor(self.tokens[s + 1:e + 1], dtype=torch.long)
        return x, y


def get_dataloader(data_folder: str, batch_size: int = 8, context_length: int = 512):
    dataset = TextDataset(data_folder, context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
    return loader, dataset.tokenizer.vocab_size
