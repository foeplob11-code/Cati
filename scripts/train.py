#!/usr/bin/env python3
"""Cati 사전학습.

9시간 세션 경계는 ResumableRun이 처리한다. 이 스크립트는 그냥 실행하면 되고,
체크포인트가 있으면 자동으로 이어서 학습한다.

    # Kaggle TPU
    python scripts/train.py --tier configs/tier0_50m.json

    # 로컬 스모크 (CPU, 작은 모델로 몇 스텝만)
    python scripts/train.py --tier configs/tier0_50m.json --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cati import ResumableRun, ResumableStream, SessionGuard, default_store
from cati.model import CatiConfig, CatiLM
from cati.packing import TokenPacker, prefetch
from cati.stream import HFSource

ROOT = Path(__file__).resolve().parent.parent
TPU_V3_8_PEAK = 420e12


# ---------------------------------------------------------------------------
# 준비
# ---------------------------------------------------------------------------
def build_stream(data_cfg: dict, phase: str, seed: int) -> ResumableStream:
    from cati.stream import usable_sources

    specs = data_cfg[phase]["sources"]
    cands = [HFSource(s["name"], s["repo"], s.get("config"), s.get("field", "text"),
                      repeat=s.get("repeat", True)) for s in specs]
    print("데이터 소스 확인")
    sources, weights, dropped = usable_sources(cands, [s["weight"] for s in specs])
    for s, w in zip(sources, weights):
        print(f"  {s.name:8s} w={w:.2f}  {s.repo}" + (f" [{s.config}]" if s.config else ""))
    if dropped:
        print(f"  ⚠️ {len(dropped)}개 소스를 못 썼다. 데이터 믹스가 계획과 다르다.")
    return ResumableStream(sources, weights, seed=seed)


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer
    if not path.exists():
        sys.exit(f"토크나이저가 없다: {path}\n"
                 "먼저 scripts/train_tokenizer.py 를 실행할 것. "
                 "전 티어가 같은 토크나이저를 공유해야 한다.")
    return Tokenizer.from_file(str(path))


def make_optimizer(cfg: CatiConfig, raw: dict, total_steps: int):
    warmup = max(1, int(total_steps * raw.get("warmup_ratio", 0.01)))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=raw["lr"],
        warmup_steps=warmup,
        decay_steps=total_steps,
        end_value=raw["lr"] * raw.get("min_lr_ratio", 0.1),
    )
    # weight decay는 2차원 이상(행렬)에만 적용한다. RMSNorm scale에 걸면 학습이 망가진다.
    wd_mask = lambda params: jax.tree_util.tree_map(lambda p: p.ndim >= 2, params)
    return optax.chain(
        optax.clip_by_global_norm(raw.get("grad_clip", 1.0)),
        optax.adamw(schedule, b1=raw.get("beta1", 0.9), b2=raw.get("beta2", 0.95),
                    weight_decay=raw.get("weight_decay", 0.1), mask=wd_mask),
    ), schedule, warmup


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--data", default="configs/data.json")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer/tokenizer.json")
    ap.add_argument("--ckpt", default=None, help="기본: artifacts/ckpt/<티어이름>")
    ap.add_argument("--phase", default="pretrain", choices=["pretrain", "anneal"])
    ap.add_argument("--session-hours", type=float, default=9.0)
    ap.add_argument("--publish-every", type=int, default=None,
                    help="N스텝마다 원격 저장소에 발행 (기본: save_every의 4배). "
                         "Colab은 예고 없이 끊겨 세션 끝 발행을 못 하므로 필요하다.")
    ap.add_argument("--quota-hours", type=float, default=160.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="CPU에서 작은 모델로 몇 스텝만")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-store", action="store_true", help="원격 발행 생략")
    args = ap.parse_args()

    raw = json.loads((ROOT / args.tier).read_text())
    cfg = CatiConfig.load(ROOT / args.tier)
    data_cfg = json.loads((ROOT / args.data).read_text())

    devices = jax.devices()
    n_dev = len(devices)

    # ---- 스모크 모드: 작게 줄인다 ------------------------------------
    if args.smoke:
        # n_kv_heads도 같이 줄여야 한다 — 350M은 4인데 n_heads를 2로 낮추면
        # "n_heads는 n_kv_heads의 배수" 조건이 깨진다.
        cfg = CatiConfig(**{**cfg.__dict__, "vocab_size": 4096, "n_layers": 2,
                            "d_model": 128, "n_heads": 2, "n_kv_heads": 1,
                            "head_dim": 64, "d_ff": 256, "seq_len": 64})
        raw = {**raw, "global_batch_tokens": 64 * 2 * n_dev,
               "micro_batch_per_device": 2}
        args.max_steps = args.max_steps or 6

    micro_per_dev = raw.get("micro_batch_per_device", 8)
    micro_total = micro_per_dev * n_dev
    seqs_per_step = raw["global_batch_tokens"] // cfg.seq_len
    if seqs_per_step % micro_total:
        sys.exit(f"global_batch_tokens/seq_len({seqs_per_step})이 "
                 f"micro_batch_per_device*devices({micro_total})의 배수가 아니다")
    accum = seqs_per_step // micro_total

    target_tokens = raw["train_tokens"] if args.phase == "pretrain" else raw["anneal_tokens"]
    tokens_per_step = raw["global_batch_tokens"]
    total_steps = args.max_steps or max(1, target_tokens // tokens_per_step)

    print("=" * 66)
    print(f"{cfg.name}  ·  {args.phase}")
    print("=" * 66)
    print(f"디바이스        {n_dev}x {devices[0].device_kind}")
    print(f"목표            {target_tokens/1e9:.2f}B 토큰 / {total_steps:,} 스텝")
    print(f"스텝당 토큰     {tokens_per_step:,}  (시퀀스 {seqs_per_step})")
    print(f"마이크로배치    디바이스당 {micro_per_dev} x {n_dev}대 = {micro_total}"
          f"  · 누적 {accum}회")
    remat_policy = raw.get("remat_policy", "full")
    print(f"remat           {'끔(스모크)' if args.smoke else f'켬 · 정책 {remat_policy}'}")

    # ---- 모델/옵티마이저 ---------------------------------------------
    model = CatiLM(cfg, dtype=jnp.bfloat16, remat=not args.smoke,
                   remat_policy=remat_policy)
    opt, schedule, warmup = make_optimizer(cfg, raw, total_steps)
    print(f"학습률          {raw['lr']:.1e} · 웜업 {warmup:,} 스텝")

    mesh = Mesh(np.asarray(devices), axis_names=("data",))
    shard_data = NamedSharding(mesh, P(None, "data", None))   # (accum, batch, T+1)
    replicate = NamedSharding(mesh, P())

    def compute_loss(params, batch):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model.apply({"params": params}, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    @partial(jax.jit, donate_argnums=(0, 1), out_shardings=(replicate, replicate,
                                                            replicate, replicate))
    def train_step(params, opt_state, micro):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)

        def accumulate(carry, mb):
            loss_sum, grad_sum = carry
            loss, grads = jax.value_and_grad(compute_loss)(params, mb)
            return (loss_sum + loss,
                    jax.tree_util.tree_map(jnp.add, grad_sum, grads)), None

        (loss, grads), _ = jax.lax.scan(accumulate, (jnp.float32(0.0), zeros), micro)
        n = micro.shape[0]
        loss = loss / n
        grads = jax.tree_util.tree_map(lambda g: g / n, grads)
        gnorm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, gnorm

    def init_fn():
        key = jax.random.PRNGKey(args.seed)
        dummy = jnp.zeros((1, min(8, cfg.seq_len)), dtype=jnp.int32)
        params = model.init(key, dummy)["params"]
        n = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"파라미터        {n:,} ({n/1e6:.1f}M)")
        return {"params": params, "opt_state": opt.init(params)}

    # ---- 데이터 -------------------------------------------------------
    tok = (load_tokenizer(ROOT / args.tokenizer) if not args.smoke
           else _smoke_tokenizer(cfg.vocab_size))
    eos = tok.token_to_id("<|endoftext|>")
    if eos is None:
        eos = 0
    stream = build_stream(data_cfg, args.phase, args.seed) if not args.smoke \
        else _smoke_stream(args.seed)
    packer = TokenPacker(stream, tok, cfg.seq_len, micro_total, eos_id=eos)

    # ---- 러너 ---------------------------------------------------------
    ckpt_root = Path(args.ckpt) if args.ckpt else ROOT / "artifacts" / "ckpt" / cfg.name
    n_params_est = raw.get("_params_estimate") or 0
    run = ResumableRun(
        cfg.name, ckpt_root,
        params=n_params_est or _estimate_params(raw),
        target_tokens=target_tokens,
        store=None if args.no_store or args.smoke else default_store(),
        guard=SessionGuard(limit_hours=args.session_hours),
        peak_flops=TPU_V3_8_PEAK, quota_hours_total=args.quota_hours,
        save_every=raw.get("save_every_steps", 200),
        log_every=raw.get("log_every_steps", 10),
        publish_every=(args.publish_every if args.publish_every is not None
                       else 4 * raw.get("save_every_steps", 200)),
    )

    state, step, tokens = run.start(init_fn, packer)
    params = jax.device_put(state["params"], replicate)
    opt_state = jax.device_put(state["opt_state"], replicate)
    print("=" * 66)

    started = time.monotonic()
    batch_iter = prefetch(packer.batches(), depth=4)
    pending: list[np.ndarray] = []
    last_state = packer.state_dict()

    while step < total_steps:
        t0 = time.monotonic()
        try:
            while len(pending) < accum:
                b, last_state = next(batch_iter)
                pending.append(b)
        except StopIteration:
            print("데이터 소진")
            break

        micro = jnp.asarray(np.stack(pending), dtype=jnp.int32)
        pending = []
        micro = jax.device_put(micro, shard_data)

        params, opt_state, loss, gnorm = train_step(params, opt_state, micro)
        loss = float(loss)
        step += 1
        tokens += tokens_per_step
        dt = time.monotonic() - t0

        if not np.isfinite(loss):
            print(f"\n손실이 발산했다 (step {step}, loss {loss}). 중단한다.")
            print("직전 체크포인트로 돌아가 학습률을 낮출 것.")
            break

        if step % run.log_every == 0 or step <= 3:
            r = run.budget.report(tokens, run.device_hours)
            print(f"step {step:>6}/{total_steps}  loss {loss:6.3f}  "
                  f"|g| {float(gnorm):6.3f}  lr {float(schedule(step)):.2e}  "
                  f"{dt:5.2f}s  {r['tokens_per_sec']:>8,.0f} tok/s  MFU {r['mfu']:4.1%}")

        # packer 상태는 실제로 학습에 쓴 배치까지만 반영한다 (프리페치 큐 제외)
        packer_snapshot = _Snapshot(last_state, packer)
        if not run.tick(step, tokens, dt, {"params": params, "opt_state": opt_state},
                        packer_snapshot, loss=loss, grad_norm=float(gnorm),
                        step_tokens=tokens_per_step):
            break

    run.finish({"params": params, "opt_state": opt_state}, step, tokens,
               _Snapshot(last_state, packer))

    tp = packer.throughput(time.monotonic() - started)
    print("\n" + "=" * 66)
    print("데이터 파이프라인")
    print(f"  토크나이징 대기 비중  {tp['fetch_fraction']:.1%}")
    print(f"  공급 속도             {tp['fetch_tokens_per_sec']:,.0f} tok/s")
    print(f"  판정                  {tp['verdict']}")
    print("=" * 66)


class _Snapshot:
    """러너에 넘길, 특정 시점의 packer 상태."""

    def __init__(self, state: dict, packer: TokenPacker):
        self._state = state
        self.docs_seen = packer.docs_seen
        self.epochs = packer.epochs

    def state_dict(self) -> dict:
        return self._state


def _estimate_params(raw: dict) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from budget import count_params
    return count_params(raw)["total"]


# ---------------------------------------------------------------------------
# 스모크용 대체물
# ---------------------------------------------------------------------------
def _smoke_tokenizer(vocab: int):
    class T:
        def encode(self, text):
            class E:
                ids = [(abs(hash(w)) % (vocab - 1)) + 1 for w in text.split()]
            return E()

        def token_to_id(self, _):
            return 0
    return T()


def _smoke_stream(seed: int):
    from cati.stream import ListSource
    items = [f"문서 {i} " + " ".join(f"단어{j}" for j in range(60)) for i in range(4000)]
    return ResumableStream([ListSource("smoke", items, repeat=True)], [1.0], seed=seed)


if __name__ == "__main__":
    main()
