"""
─────────────────────────────────────────────────────────────────────────────
  MIRON CHAT  —  terminal interface for the trained base model
─────────────────────────────────────────────────────────────────────────────
  Loads the enterprise checkpoint (saved_model/) and generates text with the
  new architecture: KV-cache decoding + token streaming + top-k/top-p +
  repetition penalty.

  NOTE: This is a BASE model (pretrained on raw text). It continues/completes
  text rather than answering questions. For chatbot-style replies it needs
  instruction fine-tuning (SFT) first.

  Run:  python scripts/chat.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path

import torch

# Repo root ko sys.path pe daalo taaki 'core' package import ho sake (yeh file
# scripts/ ke andar hai).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.miron_llm import Config, MironLLM
from core.tokenizer import MironTokenizer

SAVE_FOLDER = "saved_model"


def pick_checkpoint() -> Path:
    best = Path(f"{SAVE_FOLDER}/miron_best.pt")
    last = Path(f"{SAVE_FOLDER}/miron.pt")
    if best.exists():
        return best
    if last.exists():
        return last
    print("Model nahi mila! Pehle train karo:")
    print("   python scripts/prepare_data.py")
    print("   python scripts/train.py")
    sys.exit(1)


def load_model():
    ckpt_path = pick_checkpoint()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_cfg" not in ckpt or "model" not in ckpt:
        print("Purane format ka checkpoint mila — yeh naye architecture se "
              "incompatible hai. Dobara train karo: python scripts/train.py")
        sys.exit(1)

    cfg = Config.from_dict(ckpt["model_cfg"])
    model = MironLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    step = ckpt.get("step", "?")
    best = ckpt.get("best_val", "?")
    best = f"{best:.4f}" if isinstance(best, float) else best
    print(f"Loaded {ckpt_path.name} | {model.count_params()/1e6:.1f}M params "
          f"| step {step} | val_loss {best} | device {device.upper()}")
    return model, cfg, device


def generate_stream(model, tokenizer, prompt, device, cfg,
                    max_new_tokens=200, temperature=0.8, top_k=40,
                    top_p=0.95, repetition_penalty=1.15):
    tokens = tokenizer.enc.encode_ordinary(prompt)
    if not tokens:
        tokens = [tokenizer.sos_token]
    idx = torch.tensor([tokens[-cfg.context_length:]], dtype=torch.long, device=device)

    # stream tokens to stdout as they are produced
    buffer = []

    def on_token(tok_id):
        buffer.append(tok_id)
        # decode incrementally and print the newly resolved text
        text = tokenizer.decode(buffer)
        sys.stdout.write(text[on_token.printed:])
        sys.stdout.flush()
        on_token.printed = len(text)

    on_token.printed = 0

    model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token=tokenizer.eos_token,
        stream_cb=on_token,
    )
    print()


def chat():
    print("\n" + "=" * 56)
    print("  MIRON  —  Chat (base model)")
    print("=" * 56)
    print("  'quit' / 'exit'  ->  band karo")
    print("  'clear'          ->  screen saaf")
    print("  'settings'       ->  generation settings dekho/badlo")
    print("=" * 56 + "\n")

    model, cfg, device = load_model()
    tokenizer = MironTokenizer()

    settings = dict(max_new_tokens=200, temperature=0.8, top_k=40,
                    top_p=0.95, repetition_penalty=1.15)
    print(f"\nSettings: {settings}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            low = user_input.lower()
            if low in ("quit", "exit", "band karo"):
                print("\nMiron: Theek hai bhai, phir milenge!")
                break
            if low == "clear":
                import os
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if low == "settings":
                print(f"  current: {settings}")
                print("  badalne ke liye: key=value  (e.g. temperature=0.7)")
                continue
            if "=" in low and low.split("=")[0].strip() in settings:
                k, v = user_input.split("=", 1)
                k = k.strip()
                try:
                    settings[k] = type(settings[k])(v.strip())
                    print(f"  {k} -> {settings[k]}")
                except ValueError:
                    print("  galat value")
                continue

            print("Miron: ", end="", flush=True)
            generate_stream(model, tokenizer, user_input, device, cfg, **settings)
            print()

        except KeyboardInterrupt:
            print("\n\nMiron: Theek hai bhai, phir milenge!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    chat()
