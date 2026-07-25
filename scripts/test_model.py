#!/usr/bin/env python3
"""모델 정확성 검증.

가장 중요한 검사는 [1]이다. 실제 Flax 모델의 파라미터 수가 budget.py의 계산과
정확히 일치해야 한다. 어긋나면 쿼터 계획 전체가 틀어진다.

    .venv/bin/python scripts/test_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cati.model import CatiConfig, CatiLM, count_params, init_params, loss_fn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from budget import count_params as budget_count  # noqa: E402

PASS, FAIL = "\033[32m통과\033[0m", "\033[31m실패\033[0m"
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def tiers():
    import json
    for p in sorted((ROOT / "configs").glob("tier*.json")):
        yield CatiConfig.load(p), json.loads(p.read_text())


def test_param_counts():
    print("\n[1] 파라미터 수가 budget.py 계산과 일치하는가")
    for cfg, raw in tiers():
        expected = budget_count(raw)["total"]
        _, params = init_params(cfg)
        actual = count_params(params)
        # RMSNorm scale 벡터는 budget.py에서 층당 2*d + 최종 d로 세었다
        check(f"{cfg.name}", actual == expected,
              f"실제 {actual:,} / 계산 {expected:,} (차 {actual-expected:+,})")
        del params


def test_shapes_and_dtypes():
    print("\n[2] 형상과 dtype")
    cfg = CatiConfig.load(ROOT / "configs" / "tier0_50m.json")
    model = CatiLM(cfg, dtype=jnp.bfloat16)
    _, params = init_params(cfg, dtype=jnp.bfloat16)
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = model.apply({"params": params}, tokens)
    check("로짓 형상 (B,T,V)", logits.shape == (2, 16, cfg.vocab_size), str(logits.shape))
    check("로짓은 fp32 (손실 계산용)", logits.dtype == jnp.float32, str(logits.dtype))
    leaves = jax.tree_util.tree_leaves(params)
    check("파라미터는 fp32 저장 (마스터 가중치)",
          all(x.dtype == jnp.float32 for x in leaves))
    check("임베딩 tied — lm_head 파라미터 없음", "lm_head" not in params)


def test_causality():
    """자기회귀 모델의 필수 성질: 미래 토큰이 과거 로짓에 영향을 주면 안 된다."""
    print("\n[3] 인과성 (미래를 보지 않는가)")
    cfg = CatiConfig.load(ROOT / "configs" / "tier0_50m.json")
    model = CatiLM(cfg)
    _, params = init_params(cfg)

    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.integers(0, cfg.vocab_size, (1, 12)), dtype=jnp.int32)
    b = a.at[0, 8].set((int(a[0, 8]) + 7919) % cfg.vocab_size)   # 8번 위치만 바꾼다

    la = model.apply({"params": params}, a)
    lb = model.apply({"params": params}, b)
    before = jnp.abs(la[:, :8] - lb[:, :8]).max()
    after = jnp.abs(la[:, 8:] - lb[:, 8:]).max()
    check("변경 지점 이전 로짓 불변", float(before) == 0.0, f"최대차 {float(before):.2e}")
    check("변경 지점 이후 로짓 변화", float(after) > 0, f"최대차 {float(after):.2e}")


def test_rope_convention():
    """HF rotate_half 규약을 따르는지 확인. 여기가 틀리면 GGUF 변환이 조용히 깨진다."""
    print("\n[4] RoPE 규약 (HF rotate_half)")
    from cati.model import apply_rope, rope_tables, rotate_half

    hd = 8
    cos, sin = rope_tables(hd, 4, 10000.0)
    check("cos/sin 형상 (T, head_dim)", cos.shape == (4, hd), str(cos.shape))
    check("주파수가 앞뒤로 복제됨 (HF 규약)",
          bool(jnp.allclose(cos[:, :hd // 2], cos[:, hd // 2:])))
    check("위치 0에서 회전 없음", bool(jnp.allclose(cos[0], 1.0)) and bool(jnp.allclose(sin[0], 0.0)))

    x = jnp.arange(hd, dtype=jnp.float32).reshape(1, 1, 1, hd)
    check("rotate_half가 절반씩 분할·부호반전",
          bool(jnp.allclose(rotate_half(x)[0, 0, 0], jnp.array([-4., -5., -6., -7., 0., 1., 2., 3.]))))

    # RoPE는 회전이므로 노름을 보존해야 한다
    v = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 2, hd))
    r = apply_rope(v, cos, sin)
    check("노름 보존 (순수 회전)",
          bool(jnp.allclose(jnp.linalg.norm(v, axis=-1), jnp.linalg.norm(r, axis=-1), atol=1e-5)))


def test_gqa():
    print("\n[5] GQA — KV 헤드 반복")
    from cati.model import repeat_kv

    x = jnp.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)
    r = repeat_kv(x, 3)
    check("형상 (B,T,Hkv*n_rep,D)", r.shape == (2, 3, 6, 4), str(r.shape))
    check("각 KV 헤드가 인접하게 반복됨",
          bool(jnp.array_equal(r[:, :, 0], r[:, :, 1])) and
          bool(jnp.array_equal(r[:, :, 1], r[:, :, 2])) and
          bool(jnp.array_equal(r[:, :, 3], r[:, :, 4])))
    check("n_rep=1은 항등", bool(jnp.array_equal(repeat_kv(x, 1), x)))


def test_learns():
    """작은 배치를 외우게 해서 최적화 경로가 실제로 동작하는지 본다."""
    print("\n[6] 학습이 되는가 (배치 암기)")
    cfg = CatiConfig.load(ROOT / "configs" / "tier0_50m.json")
    small = CatiConfig(**{**cfg.__dict__, "vocab_size": 512, "n_layers": 2, "seq_len": 32})
    model, params = init_params(small)

    rng = np.random.default_rng(0)
    batch = jnp.asarray(rng.integers(0, small.vocab_size, (4, 17)), dtype=jnp.int32)

    opt = optax.adamw(3e-3, b1=0.9, b2=0.95, weight_decay=0.0)
    state = opt.init(params)

    @jax.jit
    def step(params, state, batch):
        loss, grads = jax.value_and_grad(loss_fn, argnums=1)(model, params, batch)
        updates, state = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss

    losses = []
    for _ in range(40):
        params, state, loss = step(params, state, batch)
        losses.append(float(loss))

    baseline = float(jnp.log(small.vocab_size))
    check("초기 손실 ≈ ln(vocab)", abs(losses[0] - baseline) < 0.5,
          f"{losses[0]:.3f} vs {baseline:.3f}")
    check("손실 단조 감소", losses[-1] < losses[0] * 0.5,
          f"{losses[0]:.3f} → {losses[-1]:.3f}")
    check("발산·NaN 없음", all(np.isfinite(losses)))


def test_bf16_stability():
    print("\n[7] bf16 계산 경로")
    cfg = CatiConfig.load(ROOT / "configs" / "tier0_50m.json")
    small = CatiConfig(**{**cfg.__dict__, "vocab_size": 512, "n_layers": 2, "seq_len": 32})
    model = CatiLM(small, dtype=jnp.bfloat16)
    _, params = init_params(small, dtype=jnp.bfloat16)
    rng = np.random.default_rng(0)
    batch = jnp.asarray(rng.integers(0, small.vocab_size, (2, 17)), dtype=jnp.int32)
    loss = loss_fn(model, params, batch)
    check("bf16 순전파에서 유한한 손실", bool(jnp.isfinite(loss)), f"loss {float(loss):.3f}")

    g = jax.grad(loss_fn, argnums=1)(model, params, batch)
    finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))
    check("bf16 역전파 기울기 유한", finite)


def test_config_validation():
    print("\n[8] 설정 검증")
    base = CatiConfig.load(ROOT / "configs" / "tier2_200m.json").__dict__
    # 실제로 겪은 사고: 200M 설정에 n_heads=14/n_kv_heads=4 를 넣었다가 여기서 잡혔다.
    for bad, why in [({"n_kv_heads": 4}, "n_kv_heads가 n_heads(14)의 약수가 아님"),
                     ({"head_dim": 63}, "n_heads*head_dim != d_model")]:
        try:
            CatiConfig(**{**base, **bad})
            check(f"거부: {why}", False)
        except ValueError:
            check(f"거부: {why}", True)


def main():
    print("=" * 70)
    print(f"모델 검증  (JAX {jax.__version__}, {jax.devices()[0].platform})")
    print("=" * 70)
    test_param_counts()
    test_shapes_and_dtypes()
    test_causality()
    test_rope_convention()
    test_gqa()
    test_learns()
    test_bf16_stability()
    test_config_validation()

    ok = sum(_results)
    print("\n" + "=" * 70)
    print(f"{ok}/{len(_results)} 통과")
    print("=" * 70)
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
