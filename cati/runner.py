"""세션 오케스트레이션.

체크포인트·세션가드·데이터스트림·로그·쿼터추정을 하나로 묶는다.
학습 스크립트는 스텝 함수만 제공하면 되고, 9시간 세션 경계는 여기서 처리한다.

    run = ResumableRun("cati-50m", root, params=47_700_000, target_tokens=2_000_000_000)
    arrays, step, tokens = run.start(init_fn, stream)

    while True:
        t0 = time.monotonic()
        arrays, loss, n = train_step(arrays, next(batches))
        step, tokens = step + 1, tokens + n
        if not run.tick(step, tokens, time.monotonic() - t0, arrays, stream, loss=loss):
            break

    run.finish(arrays, step, tokens, stream)
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable

from cati.checkpoint import CheckpointManager, auto_backend
from cati.session import SessionGuard
from cati.telemetry import BudgetTracker, MetricLog

STOP_TARGET = "목표 토큰 도달"
STOP_SESSION = "세션 시간 만료"


class ResumableRun:
    def __init__(
        self,
        name: str,
        root: Path | str,
        params: int,
        target_tokens: int,
        *,
        store=None,
        guard: SessionGuard | None = None,
        backend=None,
        peak_flops: float = 420e12,
        quota_hours_total: float = 160.0,
        save_every: int = 500,
        log_every: int = 20,
        keep_last: int = 2,
        publish_every: int = 0,
    ):
        self.name = name
        self.root = Path(root)
        self.guard = guard or SessionGuard()
        self.mgr = CheckpointManager(self.root, store=store,
                                     backend=backend or auto_backend(),
                                     keep_last=keep_last)
        self.budget = BudgetTracker(params=params, target_tokens=target_tokens,
                                    peak_flops=peak_flops,
                                    quota_hours_total=quota_hours_total)
        self.target_tokens = target_tokens
        self.save_every = save_every
        self.log_every = log_every
        # Colab 무료는 예고 없이 끊긴다 — finish()가 안 돌 수 있으므로 주기적으로 발행한다.
        # 스토어가 Google Drive 같은 로컬 마운트면 복사라서 싸다.
        # Kaggle Dataset처럼 업로드가 비싼 경우엔 0으로 두고 세션 끝에만 올린다.
        self.publish_every = publish_every

        self.log = MetricLog(self.root / "metrics.jsonl")
        self._stream = None
        self._base_device_hours = 0.0
        self._session_index = 0
        self.stop_reason: str | None = None

    # ---- 수명주기 ------------------------------------------------------
    def start(self, init_fn: Callable[[], dict], stream) -> tuple[dict, int, int]:
        """체크포인트가 있으면 복원하고, 없으면 init_fn()으로 새로 시작한다.

        init_fn()은 항상 호출한다. 새 런에서는 초기 상태로 쓰고, 재개할 때는
        pytree 구조를 되살리는 target으로 쓴다 — 옵티마이저 상태가 NamedTuple
        트리라서 배열만 복원하면 구조가 dict로 뭉개진다.
        """
        self._stream = stream
        target = init_fn()
        loaded = self.mgr.load_latest(target=target)

        if loaded is None:
            arrays, step, tokens = target, 0, 0
            print(f"[{self.name}] 새 런 시작")
        else:
            arrays, meta = loaded
            step, tokens = meta["step"], meta["tokens"]
            self._base_device_hours = meta.get("device_hours_used", 0.0)
            self._session_index = meta.get("session_index", 0)
            if stream is not None and "data" in meta:
                stream.load_state_dict(meta["data"])
            print(f"[{self.name}] 재개: {step:,}스텝 / {tokens/1e9:.3f}B 토큰 / "
                  f"누적 {self._base_device_hours:.2f} 디바이스-시간")

        self._session_index += 1
        print(f"[{self.name}] 세션 #{self._session_index} · {self.guard.summary()}")
        self.log.log(event="session_start", session=self._session_index,
                     step=step, tokens=tokens,
                     device_hours=self._base_device_hours)
        return arrays, step, tokens

    @property
    def device_hours(self) -> float:
        """이전 세션까지의 누적 + 이번 세션 경과."""
        return self._base_device_hours + self.guard.elapsed / 3600

    def tick(self, step: int, tokens: int, step_seconds: float,
             arrays: dict, stream=None, **metrics) -> bool:
        """스텝마다 호출한다. 계속 진행하면 True, 세션을 끝내야 하면 False."""
        self.budget.record(step_seconds, metrics.pop("step_tokens", 0) or 0)

        if self.log_every and step % self.log_every == 0:
            r = self.budget.report(tokens, self.device_hours)
            self.log.log(event="step", step=step, tokens=tokens,
                         step_seconds=step_seconds, mfu=r["mfu"],
                         tokens_per_sec=r["tokens_per_sec"], **metrics)

        done = tokens >= self.target_tokens
        expired = self.guard.should_stop
        due = self.save_every and step % self.save_every == 0

        if due or done or expired:
            self._save(step, tokens, arrays, stream or self._stream)
            if (self.publish_every and not done and not expired
                    and step % self.publish_every == 0 and self.mgr.store is not None):
                ok = self.mgr.publish(f"{self.name} step {step} (중간)")
                self.log.log(event="publish", step=step, ok=ok)

        if done:
            self.stop_reason = STOP_TARGET
            return False
        if expired:
            self.stop_reason = STOP_SESSION
            return False
        return True

    def finish(self, arrays: dict, step: int, tokens: int, stream=None) -> None:
        """세션 종료. 마지막 저장 후 원격에 발행한다."""
        self._save(step, tokens, arrays, stream or self._stream)
        reason = self.stop_reason or "루프 종료"
        print(f"\n[{self.name}] {reason} · {step:,}스텝 / {tokens/1e9:.3f}B 토큰")
        print(self.budget.format(tokens, self.device_hours))

        if self.mgr.store is None:
            print(f"[{self.name}] 로컬 저장만 (원격 스토어 없음): {self.root}")
        elif self.guard.can_upload:
            print(f"[{self.name}] 체크포인트 발행 중...")
            ok = self.mgr.publish(f"{self.name} step {step} · {tokens/1e9:.2f}B tok")
            print(f"[{self.name}] 발행 {'완료' if ok else '실패 — 로컬 체크포인트 확인'}")
        else:
            # 업로드하다 세션이 죽으면 반쯤 올라간 데이터셋이 남는다. 그럴 바엔 포기한다.
            print(f"[{self.name}] ⚠️ 업로드 시간 부족 — 건너뜀. "
                  f"노트북 출력 {self.root} 에서 회수할 것")

        self.log.log(event="session_end", session=self._session_index,
                     step=step, tokens=tokens, reason=reason,
                     device_hours=self.device_hours)
        self.log.close()

    # ---- 내부 ----------------------------------------------------------
    def _save(self, step: int, tokens: int, arrays: dict, stream) -> None:
        meta = {
            "tokens": tokens,
            "device_hours_used": self.device_hours,
            "session_index": self._session_index,
            "target_tokens": self.target_tokens,
        }
        if stream is not None:
            meta["data"] = stream.state_dict()
            meta["docs_seen"] = stream.docs_seen
            meta["epochs"] = stream.epochs
        d = self.mgr.save(step, arrays, meta)
        # 지표 로그도 체크포인트와 함께 발행되도록 복사한다.
        if self.log.path.exists():
            shutil.copy2(self.log.path, d / "metrics.jsonl")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.log.close()
