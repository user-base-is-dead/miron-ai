"""
─────────────────────────────────────────────────────────────────────────────
  PREPARE SFT DATA  —  chat JSONL -> masked token bins (.bin)
─────────────────────────────────────────────────────────────────────────────
  download_sft_data.py ne har line ek conversation di:
    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

  SFT me model ko sirf ASSISTANT ka jawab "seekhna" hai — user ka sawaal nahi.
  Isliye hum har conversation ko chat format me tokenize karte hain AUR uske
  saath ek loss-mask banate hain:

    <|user|> ...sawaal... <|sep|> <|assistant|> ...jawab... <|eos|>
       0       0   0   0    0          0          1  1  1     1
    (mask: 0 = ignore in loss, 1 = learn — yaani assistant reply + eos)

  Output (data/tokenized_sft/):
    train_ids.bin / val_ids.bin    (uint32)  -> token id stream
    train_mask.bin / val_mask.bin  (uint8)   -> 1 = learn, 0 = ignore
    meta.json

  Input :  data/downloaded-data/*.jsonl   (download_sft_data.py se)
  Run   :  python scripts/prepare_sft.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
from pathlib import Path

# ── Python version guard (numpy import se pehle saaf error) ──────────────────
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"[Miron] Python 3.11.x chahiye (abhi {sys.version.split()[0]} chal raha hai).\n"
        "        venv activate karo -> Windows: Miron311\\Scripts\\activate"
        "  |  Linux/Mac: source Miron311/bin/activate"
    )

import numpy as np

# Repo root pe anchor -> 'core' import + data paths CWD-independent (kahin se bhi chalao).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from core.tokenizer import MironTokenizer

RAW_FOLDER   = str(_ROOT / "data" / "downloaded-data")   # download_sft_data.py ke *.jsonl
OUT_FOLDER   = str(_ROOT / "data" / "tokenized_sft")     # bins + meta (sft.py yahin se padhta hai)
VAL_FRACTION = 0.01                     # 1% held out for validation


def messages_to_ids_mask(messages, tok: MironTokenizer):
    """Ek conversation (messages list) ko (ids, mask) me badalta hai.

    Format tokenizer.encode_chat se match karta hai taaki training aur inference
    same dikhein. Mask sirf assistant reply + uske eos pe 1 hota hai.

    Saaf alternating user/assistant chahiye (optional leading system, jo pehle
    user turn me fold ho jaata hai). Jo fit na ho -> None (skip).
    """
    if not isinstance(messages, list) or not messages:
        return None

    sep_token = tok.special_tokens["<|sep|>"]

    # khaali content waale turns hata do
    msgs = [m for m in messages
            if isinstance(m, dict) and (m.get("content") or "").strip()]
    if not msgs:
        return None

    # leading system message -> pehle user turn me prepend karenge
    system = None
    if msgs[0].get("role") == "system":
        system = msgs[0]["content"].strip()
        msgs = msgs[1:]

    ids, mask = [], []
    expect = "user"            # clean alternation enforce karo
    saw_assistant = False

    for m in msgs:
        role = m.get("role")
        content = m["content"].strip()
        if role != expect:
            return None        # alternation toot gaya -> conversation skip

        if role == "user":
            if system:
                content = f"{system}\n\n{content}"
                system = None
            ids.append(tok.user_token);      mask.append(0)
            body = tok.enc.encode_ordinary(content)
            ids += body;                     mask += [0] * len(body)
            ids.append(sep_token);           mask.append(0)
            ids.append(tok.assistant_token); mask.append(0)
            expect = "assistant"
        else:  # assistant
            body = tok.enc.encode_ordinary(content)
            ids += body;                     mask += [1] * len(body)
            ids.append(tok.eos_token);       mask.append(1)
            saw_assistant = True
            expect = "user"

    # khatm assistant turn pe hona chahiye (expect waapas 'user'); trailing
    # bina-jawab user turn ko reject (bin saaf rakhne ke liye)
    if not saw_assistant or expect != "user":
        return None
    return ids, mask


def _copy_blocks(src_mm, dst_mm, src_off, n, block=10_000_000):
    """memmap se memmap block-wise copy (RAM bhare bina)."""
    for i in range(0, n, block):
        j = min(i + block, n)
        dst_mm[i:j] = src_mm[src_off + i: src_off + j]


def main(raw_folder: str = RAW_FOLDER, out_folder: str = OUT_FOLDER,
         val_fraction: float = VAL_FRACTION) -> dict:
    tok = MironTokenizer()
    dtype = np.uint32   # cl100k + special ids < 100263 -> uint32 (uint16 overflow)

    jsonl_files = sorted(Path(raw_folder).glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No .jsonl files in {raw_folder}/. "
            f"Pehle SFT data download karo:  python data-download/download_sft_data.py"
        )

    Path(out_folder).mkdir(parents=True, exist_ok=True)
    print(f"Found {len(jsonl_files)} file(s): {[f.name for f in jsonl_files]}")

    # Pass 1: sab conversations ko ek flat ids/mask stream me likho (temp files).
    tmp_ids  = f"{out_folder}/_all_ids.bin"
    tmp_mask = f"{out_folder}/_all_mask.bin"
    total = n_conv = n_skip = n_learn = 0

    with open(tmp_ids, "wb") as fids, open(tmp_mask, "wb") as fmask:
        for fpath in jsonl_files:
            print(f"\nTokenizing {fpath.name} ...")
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        res = messages_to_ids_mask(row.get("messages"), tok)
                    except Exception:
                        res = None
                    if not res:
                        n_skip += 1
                        continue
                    ids, mask = res
                    np.asarray(ids,  dtype=dtype).tofile(fids)
                    np.asarray(mask, dtype=np.uint8).tofile(fmask)
                    total   += len(ids)
                    n_learn += sum(mask)
                    n_conv  += 1
                    if n_conv % 5000 == 0:
                        print(f"   {n_conv:,} convos | {total:,} tokens")

    if total == 0:
        # temp cleanup phir saaf error
        for p in (tmp_ids, tmp_mask):
            if os.path.exists(p):
                os.remove(p)
        raise ValueError(
            "Koi valid conversation nahi mila (sab skip ho gaye). JSONL format "
            "check karo: har line {\"messages\": [{role,content}, ...]} honi chahiye."
        )

    learn_pct = 100.0 * n_learn / total
    print(f"\nConversations: {n_conv:,} likhe, {n_skip:,} skipped")
    print(f"Total tokens : {total:,}  (learn-targets {n_learn:,} = {learn_pct:.1f}%)")

    # Pass 2: train/val split (memmap copy, RAM safe)
    all_ids  = np.memmap(tmp_ids,  dtype=dtype,    mode="r", shape=(total,))
    all_mask = np.memmap(tmp_mask, dtype=np.uint8, mode="r", shape=(total,))
    n_val   = int(total * val_fraction)
    n_train = total - n_val
    if n_val == 0:
        print("WARNING: val split khaali (data bahut chhota). val_fraction badhao "
              "ya zyada data do.")

    paths = {
        "train_ids":  f"{out_folder}/train_ids.bin",
        "train_mask": f"{out_folder}/train_mask.bin",
        "val_ids":    f"{out_folder}/val_ids.bin",
        "val_mask":   f"{out_folder}/val_mask.bin",
    }
    print(f"Writing train ({n_train:,}) + val ({n_val:,}) tokens...")

    tr_ids  = np.memmap(paths["train_ids"],  dtype=dtype,    mode="w+", shape=(n_train,))
    tr_mask = np.memmap(paths["train_mask"], dtype=np.uint8, mode="w+", shape=(n_train,))
    _copy_blocks(all_ids,  tr_ids,  0, n_train)
    _copy_blocks(all_mask, tr_mask, 0, n_train)
    tr_ids.flush(); tr_mask.flush()
    del tr_ids, tr_mask

    if n_val > 0:
        va_ids  = np.memmap(paths["val_ids"],  dtype=dtype,    mode="w+", shape=(n_val,))
        va_mask = np.memmap(paths["val_mask"], dtype=np.uint8, mode="w+", shape=(n_val,))
        _copy_blocks(all_ids,  va_ids,  n_train, n_val)
        _copy_blocks(all_mask, va_mask, n_train, n_val)
        va_ids.flush(); va_mask.flush()
        del va_ids, va_mask
    else:
        # khaali val files banao (loader saaf error de) — actually skip, taaki
        # loader ka "missing" message clear rahe.
        pass

    del all_ids, all_mask
    os.remove(tmp_ids)
    os.remove(tmp_mask)

    meta = {
        "dtype": "uint32",
        "vocab_size": tok.vocab_size,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "total_tokens": total,
        "n_conversations": n_conv,
        "n_skipped": n_skip,
        "learn_token_pct": round(learn_pct, 2),
    }
    with open(f"{out_folder}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. {out_folder}/meta.json written.")
    print(f"  train: {n_train:,} tokens | val: {n_val:,} tokens")
    print("\nAb SFT train karo:  python scripts/sft.py")
    return meta


if __name__ == "__main__":
    main()
