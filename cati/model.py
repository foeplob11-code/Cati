"""Cati 언어 모델 — Llama 계열 디코더 (Flax linen).

설계 제약 하나가 나머지를 결정한다: **최종 산출물이 llama.cpp에서 돌아야 한다.**
그래서 자체 구조를 만들지 않고 HuggingFace LlamaForCausalLM과 가중치 호환되는
형태를 정확히 따른다. 그러면 학습 후 경로가 단순해진다.

    Flax 파라미터 → HF Llama safetensors → convert_hf_to_gguf.py → GGUF → M1

특히 RoPE는 HF의 rotate_half(앞뒤 절반 분할) 규약을 쓴다. llama.cpp는 교차(interleaved)
규약을 쓰지만 convert_hf_to_gguf.py 가 Q/K 가중치를 치환해 주므로, HF 규약을 따르는 쪽이
직접 변환기를 쓰는 것보다 안전하다. 여기를 바꾸면 배포 단계에서 조용히 깨진다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CatiConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    tie_embeddings: bool = True
    seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    name: str = "cati"

    @classmethod
    def load(cls, path: str | Path) -> "CatiConfig":
        raw = json.loads(Path(path).read_text())
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def __post_init__(self):
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads는 n_kv_heads의 배수여야 한다")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads*head_dim({self.n_heads*self.head_dim}) != d_model({self.d_model})")


# ---------------------------------------------------------------------------
# 구성 요소
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],), jnp.float32)
        # 정규화는 항상 fp32로 계산한다. bf16으로 하면 분산이 무너진다.
        f = x.astype(jnp.float32)
        f = f * jax.lax.rsqrt(jnp.mean(jnp.square(f), axis=-1, keepdims=True) + self.eps)
        return (f * scale).astype(x.dtype)


def rope_tables(head_dim: int, seq_len: int, theta: float):
    """(T, head_dim) 크기의 cos/sin. HF 규약대로 주파수를 두 번 이어붙인다."""
    inv = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv)   # (T, hd/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)                   # (T, hd)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x, cos, sin):
    """x: (B, T, H, D) · cos/sin: (T, D)"""
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep: int):
    """GQA: KV 헤드를 쿼리 헤드 수에 맞춰 반복한다. (B, T, Hkv, D) -> (B, T, Hkv*n_rep, D)"""
    if n_rep == 1:
        return x
    b, t, h, d = x.shape
    return jnp.broadcast_to(x[:, :, :, None, :], (b, t, h, n_rep, d)).reshape(b, t, h * n_rep, d)


class Attention(nn.Module):
    cfg: CatiConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, cos, sin, mask):
        c = self.cfg
        dense = lambda feats, name: nn.Dense(
            feats, use_bias=False, dtype=self.dtype, param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(0.02), name=name)

        b, t, _ = x.shape
        q = dense(c.n_heads * c.head_dim, "q_proj")(x).reshape(b, t, c.n_heads, c.head_dim)
        k = dense(c.n_kv_heads * c.head_dim, "k_proj")(x).reshape(b, t, c.n_kv_heads, c.head_dim)
        v = dense(c.n_kv_heads * c.head_dim, "v_proj")(x).reshape(b, t, c.n_kv_heads, c.head_dim)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        n_rep = c.n_heads // c.n_kv_heads
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)

        # (B, H, T, D)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(c.head_dim).astype(self.dtype)
        # 소프트맥스는 fp32에서. bf16으로 하면 긴 시퀀스에서 정밀도가 무너진다.
        scores = jnp.where(mask, scores.astype(jnp.float32), jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(scores, axis=-1).astype(self.dtype)

        out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, c.n_heads * c.head_dim)
        return dense(c.d_model, "o_proj")(out)


class MLP(nn.Module):
    cfg: CatiConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dense = lambda feats, name: nn.Dense(
            feats, use_bias=False, dtype=self.dtype, param_dtype=jnp.float32,
            kernel_init=nn.initializers.normal(0.02), name=name)
        gate = dense(self.cfg.d_ff, "gate_proj")(x)
        up = dense(self.cfg.d_ff, "up_proj")(x)
        return dense(self.cfg.d_model, "down_proj")(nn.silu(gate) * up)


class Block(nn.Module):
    cfg: CatiConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, cos, sin, mask):
        x = x + Attention(self.cfg, self.dtype, name="self_attn")(
            RMSNorm(self.cfg.norm_eps, name="input_layernorm")(x), cos, sin, mask)
        x = x + MLP(self.cfg, self.dtype, name="mlp")(
            RMSNorm(self.cfg.norm_eps, name="post_attention_layernorm")(x))
        return x


class CatiLM(nn.Module):
    cfg: CatiConfig
    dtype: jnp.dtype = jnp.float32
    # remat=True면 블록 활성값을 저장하지 않고 역전파에서 다시 계산한다.
    # 어텐션 점수 행렬이 (배치 × 헤드 × T × T) 라서 저장하면 TPU 메모리를 넘긴다.
    # 350M/2048토큰/디바이스당8 기준: 저장하면 층당 1GB, remat하면 층당 35MB.
    remat: bool = False
    # remat은 순전파를 한 번 더 계산하므로 약 30%의 추가 연산을 낸다.
    # "dots_no_batch"는 배치 차원이 없는 행렬곱(=Dense 층 출력)만 저장하고
    # 어텐션은 다시 계산한다 — 메모리가 허용되면 이쪽이 빠르다.
    # 350M은 여유가 빠듯해서 기본값은 안전한 "full"이다. MFU 실측 후 판단한다.
    remat_policy: str = "full"

    @nn.compact
    def __call__(self, tokens):
        c = self.cfg
        b, t = tokens.shape

        embed = self.param("embed_tokens",
                           nn.initializers.normal(0.02),
                           (c.vocab_size, c.d_model), jnp.float32)
        x = jnp.take(embed, tokens, axis=0).astype(self.dtype)

        cos, sin = rope_tables(c.head_dim, t, c.rope_theta)
        mask = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None, :, :]

        if self.remat:
            policies = {
                "full": None,
                "dots_no_batch": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
            }
            if self.remat_policy not in policies:
                raise ValueError(f"remat_policy는 {sorted(policies)} 중 하나여야 한다")
            block_cls = nn.remat(Block, policy=policies[self.remat_policy])
        else:
            block_cls = Block
        for i in range(c.n_layers):
            x = block_cls(c, self.dtype, name=f"layers_{i}")(x, cos, sin, mask)
        x = RMSNorm(c.norm_eps, name="norm")(x)

        if c.tie_embeddings:
            logits = jnp.einsum("btd,vd->btv", x, embed.astype(self.dtype))
        else:
            logits = nn.Dense(c.vocab_size, use_bias=False, dtype=self.dtype,
                              param_dtype=jnp.float32,
                              kernel_init=nn.initializers.normal(0.02),
                              name="lm_head")(x)
        return logits.astype(jnp.float32)      # 손실 계산은 fp32에서


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def init_params(cfg: CatiConfig, seed: int = 0, dtype=jnp.float32):
    model = CatiLM(cfg, dtype=dtype)
    tokens = jnp.zeros((1, min(8, cfg.seq_len)), dtype=jnp.int32)
    return model, model.init(jax.random.PRNGKey(seed), tokens)["params"]


def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def loss_fn(model, params, batch):
    """batch: (B, T+1) int32. 다음 토큰 예측 교차 엔트로피."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits = model.apply({"params": params}, inputs)
    logp = jax.nn.log_softmax(logits, axis=-1)
    tok_logp = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -jnp.mean(tok_logp)
