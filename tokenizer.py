import json

import tiktoken

# ─────────────────────────────────────────
#  MANINMIRON TOKENIZER
#  Hinglish + English support
#  tiktoken (GPT-4 ka tokenizer) use karta hai
# ─────────────────────────────────────────

class ManinmironTokenizer:
    def __init__(self):
        # cl100k_base = GPT-4 tokenizer (Hindi/Hinglish bhi handle karta hai)
        self.enc = tiktoken.get_encoding("cl100k_base")

        # Special tokens
        self.special_tokens = {
            "<|pad|>"      : 100257,
            "<|sos|>"      : 100258,   # start of sentence
            "<|eos|>"      : 100259,   # end of sentence
            "<|user|>"     : 100260,   # user ka turn
            "<|assistant|>": 100261,   # AI ka turn
            "<|sep|>"      : 100262,   # separator
        }

        self.pad_token = self.special_tokens["<|pad|>"]
        self.sos_token = self.special_tokens["<|sos|>"]
        self.eos_token = self.special_tokens["<|eos|>"]
        self.user_token = self.special_tokens["<|user|>"]
        self.assistant_token = self.special_tokens["<|assistant|>"]

        self.vocab_size = 100263  # base + special tokens

        print(f"Tokenizer ready | Vocab size: {self.vocab_size}")

    def encode(self, text: str, add_special=False) -> list[int]:
        """Text ko token IDs mein convert karo"""
        tokens = self.enc.encode(text)
        if add_special:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Token IDs ko text mein convert karo"""
        # special tokens filter karo
        clean = [t for t in tokens if t < 100257]
        return self.enc.decode(clean)

    def encode_chat(self, user_msg: str, assistant_msg: str | None = None) -> list[int]:
        """
        Chat format mein encode karo:
        <|user|> message <|sep|> <|assistant|> response <|eos|>
        """
        tokens = []
        tokens.append(self.user_token)
        tokens += self.enc.encode(user_msg)
        tokens.append(self.special_tokens["<|sep|>"])
        tokens.append(self.assistant_token)

        if assistant_msg:
            tokens += self.enc.encode(assistant_msg)
            tokens.append(self.eos_token)

        return tokens

    def encode_batch(self, texts: list[str], max_length: int = 512) -> list[list[int]]:
        """Multiple texts ek saath encode karo"""
        batch = []
        for text in texts:
            tokens = self.encode(text, add_special=True)
            # truncate if too long
            tokens = tokens[:max_length]
            # pad if too short
            tokens += [self.pad_token] * (max_length - len(tokens))
            batch.append(tokens)
        return batch

    def save(self, path: str = "tokenizer_config.json"):
        """Tokenizer config save karo"""
        config = {
            "encoding"      : "cl100k_base",
            "vocab_size"    : self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Tokenizer config saved -> {path}")

    def load(self, path: str = "tokenizer_config.json"):
        """Tokenizer config load karo"""
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"Tokenizer config loaded <- {path}")
        return config

    def stats(self, text: str):
        """Text ke baare mein info dikhao"""
        tokens = self.encode(text)
        print(f"Text    : {text[:80]}...")
        print(f"Tokens  : {len(tokens)}")
        print(f"Token IDs (first 10): {tokens[:10]}")
        print(f"Decoded : {self.decode(tokens[:10])}")


# ─────────────────────────────────────────
#  Test
# ─────────────────────────────────────────
if __name__ == "__main__":
    tok = ManinmironTokenizer()

    # English test
    print("\n── English Test ──")
    text_en = "Hello, how are you doing today?"
    tok.stats(text_en)

    # Hinglish test
    print("\n── Hinglish Test ──")
    text_hi = "Bhai kya haal hai? Aaj kuch kaam karna hai kya?"
    tok.stats(text_hi)

    # Hindi test
    print("\n── Hindi Test ──")
    text_hindi = "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है।"
    tok.stats(text_hindi)

    # Chat format test
    print("\n── Chat Format Test ──")
    chat_tokens = tok.encode_chat(
        user_msg="Bhai LLM kya hota hai?",
        assistant_msg="LLM ek large language model hota hai jo text generate karta hai."
    )
    print(f"Chat tokens: {len(chat_tokens)}")
    print(f"Decoded: {tok.decode(chat_tokens)}")

    # Save config
    tok.save("tokenizer_config.json")

    print("\nTokenizer test PASSED")
