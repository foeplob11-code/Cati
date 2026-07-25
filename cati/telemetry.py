"""로그와 쿼터 추정.

Kaggle 세션은 예고 없이 죽는다. 표준 출력에만 찍힌 손실 곡선은 같이 사라진다.
그래서 모든 지표를 **매 스텝 flush 하는 JSONL**로 남기고, 체크포인트와 함께 발행한다.

BudgetTracker는 계획서의 "남은 쿼터로 완주 가능한지 상시 추정"을 구현한다.
실측 MFU가 가정(35%)보다 낮으면 목표 토큰 수를 얼마로 낮춰야 하는지 바로 알려준다.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path


class MetricLog:
    """JSONL 지표 로그. 매 쓰기마다 flush 해서 세션이 죽어도 남는다."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)      # 라인 버퍼링

    def log(self, **fields) -> None:
        fields.setdefault("wall", time.time())
        self._fh.write(json.dumps(fields, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@dataclass
class BudgetTracker:
    """처리량·MFU를 측정하고 남은 쿼터로 완주 가능한지 추정한다.

    params: 모델 전체 파라미터 수 (FLOPs = 6 * params * tokens)
    target_tokens: 이 런의 목표 학습 토큰 수
    peak_flops: 가속기 피크 (TPU v3-8 bf16 = 420e12)
    quota_hours_total / quota_hours_per_week: 남은 예산
    """

    params: int
    target_tokens: int
    peak_flops: float = 420e12
    quota_hours_total: float = 160.0
    quota_hours_per_week: float = 20.0
    window: int = 50

    def __post_init__(self):
        self._times: deque[float] = deque(maxlen=self.window)
        self._toks: deque[int] = deque(maxlen=self.window)

    def record(self, step_seconds: float, step_tokens: int) -> None:
        self._times.append(step_seconds)
        self._toks.append(step_tokens)

    @property
    def tokens_per_sec(self) -> float:
        t = sum(self._times)
        return sum(self._toks) / t if t > 0 else 0.0

    @property
    def mfu(self) -> float:
        """실효 계산 활용률. 6ND / (피크 × 시간)."""
        return 6 * self.params * self.tokens_per_sec / self.peak_flops

    def report(self, tokens_done: int, device_hours_used: float) -> dict:
        tps = self.tokens_per_sec
        remaining_tokens = max(0, self.target_tokens - tokens_done)
        hours_needed = (remaining_tokens / tps / 3600) if tps > 0 else float("inf")
        hours_left = self.quota_hours_total - device_hours_used
        total_projected = device_hours_used + hours_needed

        r = {
            "progress": tokens_done / self.target_tokens if self.target_tokens else 0.0,
            "tokens_per_sec": tps,
            "mfu": self.mfu,
            "device_hours_used": device_hours_used,
            "hours_needed": hours_needed,
            "hours_left": hours_left,
            "total_projected": total_projected,
            "weeks_needed": hours_needed / self.quota_hours_per_week,
            "fits": hours_needed <= hours_left,
        }
        if not r["fits"] and tps > 0:
            # 남은 쿼터에 맞추려면 목표를 얼마로 낮춰야 하는가
            r["feasible_tokens"] = int(tokens_done + hours_left * 3600 * tps)
        return r

    def format(self, tokens_done: int, device_hours_used: float) -> str:
        r = self.report(tokens_done, device_hours_used)
        lines = [
            f"진행 {r['progress']:6.1%}  "
            f"{tokens_done/1e9:6.2f}B / {self.target_tokens/1e9:.1f}B 토큰",
            f"처리량 {r['tokens_per_sec']:,.0f} tok/s   MFU {r['mfu']:5.1%}",
            f"쿼터 {r['device_hours_used']:.1f}h 사용 / "
            f"{r['hours_needed']:.1f}h 더 필요 / {r['hours_left']:.1f}h 남음  "
            f"({r['weeks_needed']:.1f}주)",
        ]
        if r["fits"]:
            margin = r["hours_left"] - r["hours_needed"]
            lines.append(f"→ 완주 가능. 버퍼 {margin:.1f}h")
        else:
            over = r["hours_needed"] - r["hours_left"]
            lines.append(
                f"→ ⚠️ 쿼터 {over:.1f}h 초과. "
                f"목표를 {r['feasible_tokens']/1e9:.1f}B 토큰으로 낮출 것")
        return "\n".join(lines)
