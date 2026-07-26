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

# 디바이스 1개당 bf16 피크 TFLOPS. MFU를 엉뚱한 기준으로 재면 판단이 전부 틀어진다.
# (v3-8 값 420을 하드코딩해뒀다가 v5e-1에서 MFU를 2.1배 낮게 봤다)
DEVICE_PEAK_TFLOPS = {
    "TPU v5 lite": 197.0,   # v5e, 칩당 코어 1개
    "TPU v5e": 197.0,
    "TPU v5p": 459.0,
    "TPU v4": 137.5,        # 칩 275, 코어 2개
    "TPU v3": 52.5,         # v3-8 전체 420 / 8코어
    "TPU v2": 22.5,         # v2-8 전체 180 / 8코어
}
DEFAULT_PEAK_TFLOPS = 197.0


def device_peak_flops(devices) -> tuple[float, str]:
    """전체 피크 FLOPS와 근거 문자열."""
    kind = devices[0].device_kind
    for name, per_dev in DEVICE_PEAK_TFLOPS.items():
        if name.lower() in kind.lower():
            total = per_dev * len(devices) * 1e12
            return total, f"{kind} x{len(devices)} = {total/1e12:.0f} TFLOPS bf16"
    total = DEFAULT_PEAK_TFLOPS * len(devices) * 1e12
    return total, (f"{kind} x{len(devices)} — 피크값 미등록, "
                   f"{DEFAULT_PEAK_TFLOPS:.0f} TFLOPS/대로 가정")


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
def main(argv=None):
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
    ap.add_argument("--bench", type=int, default=0, metavar="N",
                    help="N스텝만 돌려 처리량/MFU를 재고 끝낸다. 체크포인트를 "
                         "저장하지 않아 본 학습에 영향이 없다. 73시간을 태우기 전에 "
                         "설정을 비교할 때 쓴다.")
    ap.add_argument("--set", action="append", default=[], metavar="키=값",
                    help="설정 임시 변경 (예: --set micro_batch_per_device=32 "
                         "--set seq_len=1024). 파일을 고치지 않는다.")
    args = ap.parse_args(argv)

    raw = json.loads((ROOT / args.tier).read_text())
    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set 은 키=값 형식이어야 한다: {item!r}")
        k, v = item.split("=", 1)
        if k in raw and isinstance(raw[k], bool):
            raw[k] = v.lower() in ("1", "true", "yes")
        elif k in raw:
            raw[k] = type(raw[k])(v)
        else:
            # 설정 파일에 없어도 코드 기본값이 있는 키(micro_batch_per_device 등)를
            # 실험할 수 있어야 한다. 타입은 값에서 추론한다.
            for cast in (int, float):
                try:
                    raw[k] = cast(v)
                    break
                except ValueError:
                    continue
            else:
                raw[k] = v
            print(f"[--set] {k} 은 설정 파일에 없던 키다 (기본값을 덮어쓴다)")
        print(f"[--set] {k} = {raw[k]!r}")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    cfg = CatiConfig(**{k: v for k, v in raw.items()
                        if k in CatiConfig.__dataclass_fields__})
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
        ok = [b for b in range(1, seqs_per_step + 1)
              if seqs_per_step % b == 0 and b % n_dev == 0 or (n_dev == 1 and seqs_per_step % b == 0)]
        sys.exit(f"micro_batch_per_device x 디바이스({micro_total})가 "
                 f"시퀀스 수({seqs_per_step})를 나누어떨어뜨리지 못한다.\n"
                 f"  쓸 수 있는 micro_batch_per_device: "
                 f"{sorted({b // n_dev for b in ok if b % n_dev == 0})}")
    accum = seqs_per_step // micro_total

    target_tokens = raw["train_tokens"] if args.phase == "pretrain" else raw["anneal_tokens"]
    tokens_per_step = raw["global_batch_tokens"]
    total_steps = args.max_steps or max(1, target_tokens // tokens_per_step)
    if args.bench:
        total_steps, args.no_store = args.bench, True
        print(f"[벤치] {args.bench}스텝만 돌리고 끝낸다. 체크포인트를 저장하지 않는다.")

    print("=" * 66)
    print(f"{cfg.name}  ·  {args.phase}")
    print("=" * 66)
    peak_flops, peak_why = device_peak_flops(devices)
    print(f"디바이스        {peak_why}")
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

    loss_chunk = raw.get("loss_chunk_tokens", 256)

    def compute_loss(params, batch):
        """어휘 투영과 교차엔트로피를 시퀀스 조각으로 나눠 계산한다.

        한 번에 하면 로짓이 (B, T, 49152) fp32 = 배치 32·길이 2048 기준 12.9GB다.
        v5e-1의 HBM 16GB를 혼자 다 먹어서 OOM이 난다. 조각내면 피크가
        (B, chunk, V) 로 줄고 메모리 트래픽도 함께 줄어 속도에도 유리하다.
        """
        inputs, targets = batch[:, :-1], batch[:, 1:]
        hidden = model.apply({"params": params}, inputs, return_hidden=True)
        b, t, _ = hidden.shape
        n = max(1, min(loss_chunk, t))
        if t % n:                      # 나누어떨어지지 않으면 통째로 (스모크 등)
            logits = model.head(params, hidden).astype(jnp.float32)
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        h = hidden.reshape(b, t // n, n, hidden.shape[-1])
        y = targets.reshape(b, t // n, n)

        def one(carry, xs):
            hc, yc = xs
            logits = model.head(params, hc).astype(jnp.float32)
            return carry + optax.softmax_cross_entropy_with_integer_labels(
                logits, yc).sum(), None

        # (chunks, B, n, d) 로 옮겨 scan 이 조각 단위로 돌게 한다
        total, _ = jax.lax.scan(one, jnp.float32(0.0),
                                (h.transpose(1, 0, 2, 3), y.transpose(1, 0, 2)))
        return total / (b * t)

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
    ckpt_root = (ROOT / "artifacts" / "bench" if args.bench
                 else Path(args.ckpt) if args.ckpt
                 else ROOT / "artifacts" / "ckpt" / cfg.name)
    if args.bench:
        import shutil as _sh
        _sh.rmtree(ckpt_root, ignore_errors=True)
    n_params_est = raw.get("_params_estimate") or 0
    run = ResumableRun(
        cfg.name, ckpt_root,
        params=n_params_est or _estimate_params(raw),
        target_tokens=target_tokens,
        store=None if args.no_store or args.smoke else default_store(),
        guard=SessionGuard(limit_hours=args.session_hours),
        peak_flops=peak_flops, quota_hours_total=args.quota_hours,
        save_every=0 if args.bench else raw.get("save_every_steps", 200),
        log_every=raw.get("log_every_steps", 10),
        publish_every=0 if args.bench else (
            args.publish_every if args.publish_every is not None
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

    if args.bench:
        r = run.budget.report(tokens, run.device_hours)
        print("\n" + "=" * 66)
        print(f"벤치 결과  {cfg.name}")
        print("=" * 66)
        print(f"  처리량        {r['tokens_per_sec']:,.0f} tok/s")
        print(f"  MFU           {r['mfu']:.1%}   (피크 {peak_flops/1e12:.0f} TFLOPS)")
        print(f"  스텝 시간     {sum(run.budget._times)/len(run.budget._times):.2f}s")
        print(f"  목표까지      {r['hours_needed']:.0f}시간 "
              f"({r['hours_needed']/20:.1f}주, 주 20h 기준)")
        print("=" * 66)
    else:
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
