"""
─────────────────────────────────────────────────────────────────────────────
  DATASET  —  memory-mapped binary token loader
─────────────────────────────────────────────────────────────────────────────
  Reads the .bin files produced by prepare_data.py using np.memmap, so even
  multi-GB corpora never get fully loaded into RAM. Each __getitem__ returns a
  random (x, y) window of length context_length for next-token prediction.

  get_bin_dataloaders(...) -> (train_loader, val_loader, vocab_size). Needs
  `python scripts/prepare_data.py` to have produced the .bin files first.
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



# ─────────────────────────────────────────────────────────────────────────────
#  SFT (instruction) data — ids + loss-mask windows
# ─────────────────────────────────────────────────────────────────────────────
#  prepare_sft.py do parallel arrays likhta hai (same length):
#    *_ids.bin  (uint32)  -> chat-formatted token stream (saare convos joined)
#    *_mask.bin (uint8)   -> 1 jahan token "seekhna" hai (assistant reply + eos),
#                            0 prompt/user/system tokens pe.
#  Pretraining ki tarah hi random fixed-length windows nikaalte hain, par target
#  par mask laga ke -100 (model ka ignore_index) kar dete hain -> loss SIRF
#  assistant tokens pe. Agar koi window poora masked nikle (saare -100 -> loss
#  NaN), to dusra window resample karte hain.

class SFTBinDataset(Dataset):
    """Random windows from an SFT ids/mask memmap pair, with -100 on masked targets."""

    def __init__(self, ids_path: str, mask_path: str, context_length: int,
                 dtype=np.uint32, samples_per_epoch: int | None = None,
                 max_resample: int = 10):
        self.ids_path = ids_path
        self.mask_path = mask_path
        self.context_length = context_length
        self.dtype = dtype
        self.max_resample = max_resample
        # lazily opened per-worker (fork/spawn safe)
        self._ids = None
        self._mask = None
        self.n_tokens = Path(ids_path).stat().st_size // np.dtype(dtype).itemsize
        n_mask = Path(mask_path).stat().st_size // np.dtype(np.uint8).itemsize
        assert n_mask == self.n_tokens, (
            f"ids/mask length mismatch ({self.n_tokens} vs {n_mask}); "
            f"dobara banao:  python scripts/prepare_sft.py"
        )
        self.max_start = self.n_tokens - context_length - 1
        assert self.max_start > 0, (
            f"{ids_path} too small for context_length={context_length}"
        )
        self.length = samples_per_epoch or (self.n_tokens // context_length)

    def _mm(self):
        if self._ids is None:
            self._ids = np.memmap(self.ids_path, dtype=self.dtype, mode="r")
            self._mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        return self._ids, self._mask

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        ids, mask = self._mm()
        id_win = mk_win = None
        for _try in range(self.max_resample):
            i = np.random.randint(0, self.max_start)
            id_win = np.asarray(ids[i: i + self.context_length + 1], dtype=np.int64)
            # target mask = mask of the *next* token at each position (y is shifted)
            mk_win = np.asarray(mask[i + 1: i + self.context_length + 1], dtype=np.int64)
            if mk_win.any():
                break
        x = torch.from_numpy(id_win[:-1])
        targets = id_win[1:].copy()
        targets[mk_win == 0] = -100          # ignore_index -> loss sirf assistant pe
        y = torch.from_numpy(targets)
        return x, y


def get_sft_dataloaders(data_folder: str, batch_size: int, context_length: int,
                        num_workers: int = 2):
    """prepare_sft.py ke output se (train_loader, val_loader, vocab_size) banata hai."""
    folder = Path(data_folder)
    meta_path = folder / "meta.json"
    if not meta_path.exists() or meta_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"{meta_path} missing ya empty hai. "
            f"Pehle SFT data taiyaar karo:  python scripts/prepare_sft.py"
        )
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{meta_path} corrupt JSON hai ({e}). "
            f"Dobara banao:  python scripts/prepare_sft.py"
        ) from e
    if meta.get("dtype") not in DTYPE_MAP:
        raise ValueError(f"meta.json me galat/missing 'dtype': {meta.get('dtype')!r}")
    dtype = DTYPE_MAP[meta["dtype"]]

    for fname in ("train_ids.bin", "train_mask.bin", "val_ids.bin", "val_mask.bin"):
        p = folder / fname
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(
                f"{p} missing ya empty hai. "
                f"Pehle SFT data taiyaar karo:  python scripts/prepare_sft.py"
            )

    train_ds = SFTBinDataset(str(folder / "train_ids.bin"),
                             str(folder / "train_mask.bin"), context_length, dtype)
    val_ds = SFTBinDataset(str(folder / "val_ids.bin"),
                           str(folder / "val_mask.bin"), context_length, dtype)
    val_ds.length = min(200, val_ds.length)   # validation pass ko chhota rakho

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True, drop_last=True,
                  persistent_workers=num_workers > 0,
                  worker_init_fn=_seed_worker)
    train_loader = DataLoader(train_ds, shuffle=False, **common)
    val_loader   = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, meta["vocab_size"]
