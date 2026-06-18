# Python Version Guide — Miron LLM

Ye doc batata hai ki Miron LLM ko **konsa Python version chahiye**, kyun chahiye,
aur galat version pe kya hota hai. (Last checked: 2026-06-19)

---

## TL;DR

- **Required: Python 3.11.x** (is repo ka venv `Miron311` 3.11 pe hai).
- Hamesha venv activate karke kaam karo — **system default `python` use mat karo**.
- System default abhi **3.14** hai, jisme project ke deps (torch/numpy) install hi nahi honge.

```bash
# Windows
Miron311\Scripts\activate
python --version      # ab 3.11.9 dikhna chahiye
```

---

## Is machine ka current snapshot

| Kya | Version | Note |
|-----|---------|------|
| System default `python` | **3.14.4** | ⚠️ PATH pe yahi hai — galat |
| Project venv `Miron311` | **3.11.9** | ✅ sahi, isi ko use karo |
| Installed (py launcher) | 3.9, 3.11, 3.14 | `py -0p` se list |

Matlab: agar venv activate kiye **bina** `python scripts/train.py` chalaya, to
3.14 chalega — galat. Venv activate karke chalaya, to 3.11.9 — sahi.

Khud check karne ke liye:

```bash
python --version                       # current default
py -0p                                 # saare installed Pythons (Windows)
Miron311\Scripts\python.exe --version  # venv ka Python
```

---

## Kyun exactly 3.11? (3.14 kyun nahi)

`requirements.txt` ke deps **pinned** hain:

```
torch==2.5.1
numpy==2.4.6
tiktoken==0.13.0
datasets==4.8.5
```

In versions ke pre-built wheels Python **3.14 ke liye exist hi nahi karte**
(3.14 bahut naya hai; torch 2.5.1 us se pehle ka release hai). To 3.14 pe:

- `pip install -r requirements.txt` → torch pe **install fail** ho jaata hai, ya
- agar kisi tarah install ho bhi gaya to **import pe crash** hota hai.

3.11 pe ye sab wheels available hain → clean install + run. Isliye venv 3.11 pe banaya gaya hai.

---

## Sahi tarika — setup & daily use

Pehli baar venv banana (`uv` recommended — repo isi ko use karta hai):

```bash
uv venv Miron311 --python 3.11
Miron311\Scripts\activate          # Windows  (Linux/Mac: source Miron311/bin/activate)
uv pip install -r requirements.txt # ya: pip install -r requirements.txt
```

Har baar kaam shuru karne se pehle:

```bash
Miron311\Scripts\activate
python --version                   # confirm: 3.11.9
python scripts/train.py
```

Activate hone ke baad `python` apne aap venv ka 3.11 wala ban jaata hai — system
3.14 ko ignore kar deta hai (sirf is terminal session ke liye).

---

## `requires-python` kaise kaam karta hai (optional enforcement)

Agar `pyproject.toml` mein ye add karein:

```toml
[project]
name = "miron-llm"
requires-python = ">=3.11,<3.12"
```

To ye **declaration** hai "project sirf 3.11.x pe chalega". Ye in jagah enforce hota hai:

- **`uv` ke saath:** `uv venv` khud `requires-python` padh ke automatically 3.11
  pick kar lega (`--python 3.11` likhne ki zaroorat nahi). `uv run` / `uv sync`
  bhi galat interpreter pe complain karenge.
- **`pip install .`** (project ko package ki tarah install karne pe): pip 3.14 pe
  **refuse** karega — *"Requires-Python >=3.11,<3.12 not satisfied"*.

Jo ye **NAHI** karta (honest limitation):

- Plain `python scripts/train.py` ko nahi rokta — `requires-python` sirf
  install/tooling time pe check hota hai, script run hote waqt nahi.
- `pip install -r requirements.txt` isko consult nahi karta (sirf har package ke
  apne constraints lagte hain).

---

## Sabse pakka guard (kaise bhi launch karo)

Agar chahte ho ki **kisi bhi tarah galat version pe chalaane par turant saaf
error** aaye, to `scripts/train.py` (aur `scripts/chat.py`) ke bilkul top pe ye
chhota check daal do:

```python
import sys
if sys.version_info[:2] != (3, 11):
    sys.exit(
        f"Miron ko Python 3.11.x chahiye — tum {sys.version.split()[0]} pe ho.\n"
        f"venv activate karo:  Miron311\\Scripts\\activate"
    )
```

Ye runtime pe chalta hai, isliye venv bhulne pe bhi safety milti hai.

---

## Troubleshooting

| Error / symptom | Wajah | Fix |
|-----------------|-------|-----|
| `Could not find a version that satisfies the requirement torch==2.5.1` | Galat Python (3.14) pe install kar rahe ho | Venv (3.11) activate karke install karo |
| `python --version` 3.14 dikha raha hai | Venv activate nahi hua | `Miron311\Scripts\activate` |
| `ModuleNotFoundError: torch` chalate waqt | Galat interpreter / venv activate nahi | Venv activate, phir `python scripts/train.py` |
| venv corrupt / missing | — | `uv venv Miron311 --python 3.11` se dobara banao |

---

## Summary

- **3.11.x** = supported. Repo ka `Miron311` venv isi pe hai.
- System default **3.14** project ke liye kaam nahi karega.
- Hamesha venv activate karke chalo; chahein to `pyproject.toml` + runtime guard
  se isko enforce bhi kar sakte ho.
