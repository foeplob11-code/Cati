#!/usr/bin/env python3
"""체크포인트/재개 인프라 정확성 검증.

핵심 주장: **중단 후 재개한 런은 무중단 런과 비트 단위로 같아야 한다.**

이게 틀리면 손실 곡선은 멀쩡해 보이는데 실제로는 코퍼스 앞부분만 반복 학습하는
상태가 된다. 97시간을 태우고 나서야 알게 되는 종류의 버그라 여기서 잡는다.

    .venv/bin/python scripts/test_resume.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cati import CheckpointManager, ListSource, NumpyBackend, ResumableStream, SessionGuard
from cati.telemetry import BudgetTracker

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "artifacts" / "test_resume"

DIM = 16
BATCH = 4
TOTAL_STEPS = 60
SAVE_EVERY = 10
SEED = 1234

PASS, FAIL = "\033[32m통과\033[0m", "\033[31m실패\033[0m"
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# 장난감 학습 — 결정적이기만 하면 된다
# ---------------------------------------------------------------------------
def text_vec(text: str) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    return np.frombuffer(h[:DIM], dtype=np.uint8).astype(np.float64) / 255.0


def train_step(p: np.ndarray, m: np.ndarray, texts: list[str]):
    g = sum(text_vec(t) for t in texts) / len(texts)
    m = 0.9 * m + 0.1 * g
    p = p * 0.99 + 0.01 * m
    return p, m


def make_sources():
    """ko는 반복(다중 에폭), en/code는 1회성 — 고갈 처리까지 검증한다."""
    return [
        ListSource("ko", [f"한국어 문서 {i}" for i in range(120)], repeat=True),
        ListSource("en", [f"english document {i}" for i in range(90)], repeat=False),
        ListSource("code", [f"def f{i}(): pass" for i in range(40)], repeat=False),
    ]


def fresh_stream():
    return ResumableStream(make_sources(), [0.5, 0.35, 0.15], seed=SEED)


def init_state():
    rng = np.random.default_rng(SEED)
    return rng.standard_normal(DIM), np.zeros(DIM)


# ---------------------------------------------------------------------------
# 런 실행기
# ---------------------------------------------------------------------------
def run(root: Path, stop_at: int | None, log: list[str]):
    """체크포인트가 있으면 재개하고, stop_at 스텝에서 저장 후 중단한다."""
    mgr = CheckpointManager(root, backend=NumpyBackend(), keep_last=2)
    stream = fresh_stream()

    loaded = mgr.load_latest()
    if loaded is None:
        p, m = init_state()
        step, tokens = 0, 0
    else:
        arrays, meta = loaded
        p, m = arrays["p"], arrays["m"]
        step, tokens = meta["step"], meta["tokens"]
        stream.load_state_dict(meta["data"])

    it = iter(stream)
    while step < TOTAL_STEPS:
        texts = []
        for _ in range(BATCH):
            try:
                name, text = next(it)
            except StopIteration:
                break
            texts.append(text)
            log.append(f"{name}:{text}")
        if not texts:
            break

        p, m = train_step(p, m, texts)
        step += 1
        tokens += sum(len(t) for t in texts)

        hit_stop = stop_at is not None and step == stop_at
        if step % SAVE_EVERY == 0 or hit_stop or step == TOTAL_STEPS:
            mgr.save(step, {"p": p, "m": m},
                     {"tokens": tokens, "data": stream.state_dict(),
                      "epochs": stream.epochs})
        if hit_stop:
            return p, m, step, tokens, "중단"
    return p, m, step, tokens, "완료"


# ---------------------------------------------------------------------------
# 검사
# ---------------------------------------------------------------------------
def test_exact_resume():
    print("\n[1] 중단·재개가 무중단과 동일한가")
    shutil.rmtree(WORK, ignore_errors=True)

    log_a: list[str] = []
    pa, ma, sa, ta, _ = run(WORK / "uninterrupted", None, log_a)
    print(f"  무중단: {sa}스텝 {ta:,}자 {len(log_a)}문서")

    # 세션 3개로 쪼개서 같은 지점까지 간다
    log_b: list[str] = []
    root_b = WORK / "interrupted"
    for i, stop in enumerate([17, 34, 51, None], 1):
        _, _, sb, _, status = run(root_b, stop, log_b)
        print(f"  세션{i}: → {sb}스텝 ({status})")
    pb, mb, sb, tb, _ = run(root_b, None, log_b)

    check("최종 스텝 일치", sa == sb, f"{sa} vs {sb}")
    check("파라미터 비트 단위 일치", np.array_equal(pa, pb),
          f"최대차 {np.abs(pa-pb).max():.3e}")
    check("옵티마이저 상태 비트 단위 일치", np.array_equal(ma, mb))
    check("누적 토큰 일치", ta == tb, f"{ta:,} vs {tb:,}")
    check("문서 소비 순서 완전 일치", log_a == log_b,
          f"{len(log_a)} vs {len(log_b)}문서" +
          ("" if log_a == log_b else f", 첫 불일치 {_first_diff(log_a, log_b)}"))


def _first_diff(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def test_no_data_repeat():
    print("\n[2] 재개가 데이터를 되감지 않는가")
    log: list[str] = []
    shutil.rmtree(WORK / "norepeat", ignore_errors=True)
    for stop in [10, 20, 30, None]:
        run(WORK / "norepeat", stop, log)

    # ko는 repeat=True라 반복이 정상. en/code는 1회성이므로 중복이 있으면 안 된다.
    for name, size in [("en", 90), ("code", 40)]:
        docs = [d for d in log if d.startswith(f"{name}:")]
        check(f"{name} 소스 중복 소비 없음", len(docs) == len(set(docs)),
              f"{len(docs)}개 소비 / 고유 {len(set(docs))}개 / 전체 {size}개")


def test_atomic_save():
    print("\n[3] 반쯤 쓰인 체크포인트를 무시하는가")
    root = WORK / "atomic"
    shutil.rmtree(root, ignore_errors=True)
    mgr = CheckpointManager(root, backend=NumpyBackend())
    p, m = init_state()
    mgr.save(10, {"p": p, "m": m}, {"tokens": 1, "data": fresh_stream().state_dict()})

    # 저장 도중 세션이 죽은 상황을 흉내낸다
    (root / ".tmp_step_000000020").mkdir()
    (root / ".tmp_step_000000020" / "arrays.npz").write_bytes(b"garbage")

    latest = mgr.latest_dir()
    check("임시 디렉터리를 체크포인트로 오인하지 않음",
          latest is not None and latest.name == "step_000000010", str(latest))
    check("복원 성공", mgr.load_latest() is not None)


def test_retention():
    print("\n[4] 오래된 체크포인트를 정리하는가")
    root = WORK / "retention"
    shutil.rmtree(root, ignore_errors=True)
    mgr = CheckpointManager(root, backend=NumpyBackend(), keep_last=2)
    p, m = init_state()
    for s in range(1, 6):
        mgr.save(s, {"p": p, "m": m}, {"tokens": s, "data": fresh_stream().state_dict()})
    kept = sorted(d.name for d in root.glob("step_*"))
    check("최근 2개만 유지", kept == ["step_000000004", "step_000000005"], str(kept))


def test_session_guard():
    print("\n[5] 세션 가드 데드라인")
    g = SessionGuard(limit_hours=9.0, save_reserve_minutes=35, upload_reserve_minutes=10)
    check("정상종료 데드라인 8.42h", abs(g.soft_deadline - (9 * 3600 - 35 * 60)) < 1,
          f"{g.soft_deadline/3600:.2f}h")
    check("업로드 데드라인 8.83h", abs(g.hard_deadline - (9 * 3600 - 10 * 60)) < 1,
          f"{g.hard_deadline/3600:.2f}h")
    check("시작 직후엔 계속 진행", not g.should_stop)
    check("업로드 가능", g.can_upload)

    g.started_at = time.monotonic() - (8.5 * 3600)
    check("8.5h 경과 시 종료 신호", g.should_stop)
    check("8.5h 경과 시 업로드 여전히 가능", g.can_upload)
    g.started_at = time.monotonic() - (8.95 * 3600)
    check("8.95h 경과 시 업로드 포기", not g.can_upload)

    g2 = SessionGuard(limit_hours=9.0)
    g2.started_at = time.monotonic() - (8.0 * 3600)
    check("긴 스텝은 시작하지 않음", not g2.fits(3600))
    check("짧은 스텝은 진행", g2.fits(60))

    try:
        SessionGuard(limit_hours=9.0, save_reserve_minutes=5, upload_reserve_minutes=10)
        check("잘못된 예약시간 거부", False)
    except ValueError:
        check("잘못된 예약시간 거부", True)


def test_budget_tracker():
    print("\n[6] 쿼터 추정")
    # cati-350m: 343.5M 파라미터, 25B 토큰 목표
    bt = BudgetTracker(params=343_500_000, target_tokens=25_000_000_000,
                       peak_flops=420e12, quota_hours_total=160.0)
    # MFU 35% 가정 시 처리량: 0.35*420e12 / (6*343.5e6) = 71,325 tok/s
    for _ in range(10):
        bt.record(step_seconds=1.0, step_tokens=71_325)
    check("MFU 35% 복원", abs(bt.mfu - 0.35) < 0.01, f"{bt.mfu:.1%}")

    r = bt.report(tokens_done=0, device_hours_used=5.5)
    check("계획서의 97.4h를 재현", abs(r["hours_needed"] - 97.4) < 1.0,
          f"{r['hours_needed']:.1f}h")
    check("쿼터 안에 들어옴", r["fits"])

    # MFU가 절반으로 떨어지면 하향 권고가 나와야 한다
    slow = BudgetTracker(params=343_500_000, target_tokens=25_000_000_000,
                         quota_hours_total=160.0)
    for _ in range(10):
        slow.record(step_seconds=1.0, step_tokens=30_000)
    r2 = slow.report(tokens_done=0, device_hours_used=5.5)
    check("MFU 저하 시 초과 감지", not r2["fits"], f"{r2['hours_needed']:.0f}h 필요")
    check("실행 가능 토큰 수 제시", 0 < r2["feasible_tokens"] < 25_000_000_000,
          f"{r2['feasible_tokens']/1e9:.1f}B 토큰")
    print("\n" + "\n".join("    " + l for l in slow.format(0, 5.5).splitlines()))


def test_runner_across_sessions():
    """Kaggle의 실제 조건: 새 세션은 /kaggle/working 이 비어 있다.

    그래서 로컬 체크포인트가 없어도 원격 스토어에서 끌어와 이어갈 수 있어야 한다.
    """
    print("\n[7] 러너 — 작업 디렉터리가 비워진 상태에서 재개")
    from cati import LocalStore, ResumableRun

    base = WORK / "runner"
    shutil.rmtree(base, ignore_errors=True)
    local, remote = base / "working", base / "remote"

    # 기준: 무중단
    log_ref: list[str] = []
    p_ref, m_ref, _, _, _ = run(base / "ref", None, log_ref)

    log: list[str] = []
    arrays = None
    for session in range(1, 6):
        if local.exists():
            shutil.rmtree(local)        # ← Kaggle 새 세션: 작업 디렉터리가 비어 있다
        r = ResumableRun("toy", local, params=1_000_000, target_tokens=10**9,
                         store=LocalStore(remote), backend=NumpyBackend(),
                         guard=SessionGuard(limit_hours=9.0),
                         save_every=5, log_every=0)
        stream = fresh_stream()
        arrays, step, tokens = r.start(lambda: dict(zip(("p", "m"), init_state())), stream)
        it = iter(stream)

        while step < TOTAL_STEPS:
            texts = []
            for _ in range(BATCH):
                try:
                    name, text = next(it)
                except StopIteration:
                    break
                texts.append(text)
                log.append(f"{name}:{text}")
            if not texts:
                break
            arrays["p"], arrays["m"] = train_step(arrays["p"], arrays["m"], texts)
            step += 1
            tokens += sum(len(t) for t in texts)

            # 13스텝마다 세션이 죽는 상황을 만든다
            if step % 13 == 0:
                r.guard.started_at = time.monotonic() - r.guard.soft_deadline - 1
            if not r.tick(step, tokens, 0.01, arrays, stream, step_tokens=sum(len(t) for t in texts)):
                break
        r.finish(arrays, step, tokens, stream)
        if step >= TOTAL_STEPS:
            break

    check("작업 디렉터리를 비워도 재개 성공", step == TOTAL_STEPS, f"{step}스텝")
    check("무중단 결과와 비트 단위 일치", np.array_equal(arrays["p"], p_ref),
          f"최대차 {np.abs(arrays['p']-p_ref).max():.3e}")
    check("문서 소비 순서 일치", log == log_ref, f"{len(log)} vs {len(log_ref)}문서")
    check("지표 로그가 체크포인트와 함께 보존됨",
          any((d / "metrics.jsonl").exists() for d in remote.glob("step_*")))


def test_optimizer_pytree_structure():
    """회귀 테스트.

    옵티마이저 상태는 dict가 아니라 NamedTuple 트리다 (optax ScaleByAdamState 등).
    배열만 저장/복원하면 구조가 dict로 뭉개져서 재개 직후
    `'dict' object has no attribute 'mu'` 로 죽는다.
    장난감 옵티마이저(그냥 배열)로는 이 버그가 안 잡힌다.
    """
    print("\n[8] 실제 옵티마이저 상태의 pytree 구조 보존")
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError:
        print("  (JAX 미설치 — 건너뜀)")
        return

    root = WORK / "pytree"
    shutil.rmtree(root, ignore_errors=True)

    params = {"w": jnp.ones((4, 4)), "norm": {"scale": jnp.ones((4,))}}
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(1e-3, b1=0.9, b2=0.95, weight_decay=0.1))
    opt_state = opt.init(params)
    state = {"params": params, "opt_state": opt_state}

    # 한 스텝 밟아서 상태에 값이 들어가게 한다
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 0.1), params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    state = {"params": params, "opt_state": opt_state}

    mgr = CheckpointManager(root, backend=NumpyBackend())
    mgr.save(1, state, {"tokens": 0, "data": fresh_stream().state_dict()})

    # target 없이 복원하면 구조가 뭉개진다 — 그게 원래 버그
    no_target, _ = mgr.load_latest()
    check("target 없으면 구조가 dict로 뭉개짐 (버그 재현)",
          not isinstance(no_target["opt_state"], tuple))

    # target을 주면 원래 구조로 돌아온다
    fresh = {"params": jax.tree_util.tree_map(jnp.zeros_like, params),
             "opt_state": opt.init(jax.tree_util.tree_map(jnp.zeros_like, params))}
    restored, _ = mgr.load_latest(target=fresh)

    check("옵티마이저 상태 타입 복원",
          type(restored["opt_state"]) is type(opt_state),
          type(restored["opt_state"]).__name__)
    check("Adam 모먼트 값 일치",
          bool(jnp.allclose(restored["opt_state"][1][0].mu["w"], opt_state[1][0].mu["w"])))
    check("파라미터 값 일치", bool(jnp.allclose(restored["params"]["w"], params["w"])))

    # 진짜 검증: 복원한 상태로 옵티마이저가 돌아가는가
    try:
        opt.update(grads, restored["opt_state"], restored["params"])
        check("복원한 상태로 opt.update() 동작", True)
    except Exception as e:
        check("복원한 상태로 opt.update() 동작", False, f"{type(e).__name__}: {e}")

    # 구조가 다르면 조용히 통과하지 말고 실패해야 한다
    try:
        mgr.load_latest(target={"params": {"w": jnp.ones((4, 4))}})
        check("구조 불일치 시 명확한 오류", False, "조용히 통과했다")
    except KeyError:
        check("구조 불일치 시 명확한 오류", True)


def main():
    print("=" * 68)
    print("체크포인트/재개 인프라 검증")
    print("=" * 68)
    test_exact_resume()
    test_no_data_repeat()
    test_atomic_save()
    test_retention()
    test_session_guard()
    test_budget_tracker()
    test_runner_across_sessions()
    test_optimizer_pytree_structure()

    ok = sum(_results)
    print("\n" + "=" * 68)
    print(f"{ok}/{len(_results)} 통과")
    print("=" * 68)
    shutil.rmtree(WORK, ignore_errors=True)
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
