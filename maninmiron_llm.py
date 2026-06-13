"""
─────────────────────────────────────────────────────────────────────────────
  MANINMIRON LLM  —  Enterprise-grade GPT-style decoder transformer
─────────────────────────────────────────────────────────────────────────────
  Modern architecture:
    • RMSNorm                 (faster, more stable than LayerNorm)
    • Rotary Position Emb     (RoPE — no learned position table, extrapolates)
    • SwiGLU feed-forward     (LLaMA-style gated MLP)
    • Grouped-Query Attention (GQA — fewer KV heads = less memory)
    • Flash Attention         (F.scaled_dot_product_attention)
    • KV-cache                (fast autoregressive decoding)
    • Weight tying            (embedding == lm_head)
    • Gradient checkpointing  (train bigger models on small GPUs)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    vocab_size:     int   = 100263   # cl100k_base + special tokens
    context_length: int   = 1024     # max sequence length
    d_model:        int   = 768      # embedding / residual width
    num_heads:      int   = 12       # query heads
    num_kv_heads:   int   = 4        # key/value heads (GQA); == num_heads -> MHA
    num_layers:     int   = 12       # transformer blocks
    d_ff:           int   = 2048     # SwiGLU hidden width (~ 8/3 * d_model)
    dropout:        float = 0.0      # 0.0 is standard for large-data pretraining
    rope_theta:     float = 10000.0  # RoPE base frequency
    norm_eps:       float = 1e-5     # RMSNorm epsilon
    tie_weights:    bool  = True     # share token embedding with output head
    grad_checkpoint:bool  = False    # trade compute for memory during training

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model must divide num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must divide num_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ── RMSNorm ─────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# ── Rotary Position Embedding ─────────────────────────────────────────────────
def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin tables for RoPE. Returns (seq_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, Dh)  cos/sin: (T, Dh)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ── Grouped-Query Self Attention ──────────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head    = cfg.num_heads
        self.n_kv_head = cfg.num_kv_heads
        self.head_dim  = cfg.head_dim
        self.n_rep     = self.n_head // self.n_kv_head
        self.dropout_p = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.d_model, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        # append to KV cache for incremental decoding
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # expand KV heads to match query heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Flash attention. is_causal only valid when q and k have equal length.
        is_causal = kv_cache is None or kv_cache[0] is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.resid_drop(self.o_proj(out))
        return out, new_cache


# ── SwiGLU Feed Forward ────────────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ── Transformer Block ──────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn  = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ff    = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), cos, sin, kv_cache)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_cache


# ── Main Model ─────────────────────────────────────────────────────────────────
class ManinmironLLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm_f    = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.token_emb.weight = self.lm_head.weight

        # rope cache (registered as buffers, rebuilt if seq grows)
        self._rope_len = 0
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.num_layers))

        print(f"ManinmironLLM | {self.count_params()/1e6:.1f}M params | "
              f"L{cfg.num_layers} d{cfg.d_model} h{cfg.num_heads}/kv{cfg.num_kv_heads} ctx{cfg.context_length}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def count_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_weights:
            n -= self.token_emb.weight.numel()
        return n

    def _ensure_rope(self, seq_len, device, dtype):
        if seq_len > self._rope_len or self.rope_cos.device != device:
            cos, sin = build_rope_cache(
                max(seq_len, self.cfg.context_length),
                self.cfg.head_dim, self.cfg.rope_theta, device, dtype,
            )
            self.rope_cos, self.rope_sin = cos, sin
            self._rope_len = cos.size(0)

    def forward(self, idx, targets=None, kv_caches=None, start_pos=0):
        B, T = idx.shape
        x = self.drop(self.token_emb(idx))

        self._ensure_rope(start_pos + T, idx.device, x.dtype)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]

        new_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            if self.cfg.grad_checkpoint and self.training:
                x, nc = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, cache_i, use_reentrant=False
                )
            else:
                x, nc = block(x, cos, sin, cache_i)
            if new_caches is not None:
                new_caches.append(nc)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss

        # inference: only compute logits for the last position to save memory
        logits = self.lm_head(x[:, [-1], :])
        return logits, new_caches

    # ── Optimizer with proper weight-decay groups ──
    def configure_optimizer(self, lr, weight_decay, betas, device_type):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    # ── Generation with KV-cache + sampling controls ──
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=50,
                 top_p=0.95, repetition_penalty=1.1, eos_token=None, stream_cb=None):
        self.eval()
        device = idx.device
        kv_caches = [(None, None) for _ in range(self.cfg.num_layers)]

        # prime the cache with the full prompt
        logits, kv_caches = self(idx[:, -self.cfg.context_length:], kv_caches=kv_caches, start_pos=0)
        pos = idx.size(1)
        generated = idx

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]

            # repetition penalty
            if repetition_penalty != 1.0:
                for tok in set(generated[0].tolist()):
                    logits[0, tok] /= repetition_penalty

            logits = logits / max(temperature, 1e-5)

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                to_remove = remove.scatter(1, sorted_idx, remove)
                logits[to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)

            if stream_cb is not None:
                stream_cb(next_tok.item())
            if eos_token is not None and next_tok.item() == eos_token:
                break

            # feed only the new token; cache handles the rest
            if pos >= self.cfg.context_length:
                # context full: re-prime (simple sliding window)
                kv_caches = [(None, None) for _ in range(self.cfg.num_layers)]
                logits, kv_caches = self(generated[:, -self.cfg.context_length:],
                                         kv_caches=kv_caches, start_pos=0)
            else:
                logits, kv_caches = self(next_tok, kv_caches=kv_caches, start_pos=pos)
                pos += 1

        return generated


# ── Self test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config(num_layers=2, d_model=128, num_heads=4, num_kv_heads=2,
                 d_ff=256, context_length=64, vocab_size=1000)
    model = ManinmironLLM(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, y)
    print(f"train logits {tuple(logits.shape)} | loss {loss.item():.4f}")

    prompt = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model.generate(prompt, max_new_tokens=20, top_k=10)
    print(f"generated {tuple(out.shape)}")
    print("Architecture test PASSED")
