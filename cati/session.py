"""Kaggle 9시간 세션 가드.

Kaggle 노트북은 정해진 시간에 **경고 없이** 종료된다. 종료 순간에 저장 중이면
체크포인트가 깨지고, 저장을 안 했으면 그 세션의 계산이 통째로 날아간다.

그래서 실제 한계보다 앞선 두 개의 데드라인을 둔다.

    ├──────────── 학습 ────────────┤ 저장 ┤ 업로드 ┤   여유   ┤
    0h                          7.6h        8.4h    8.5h    9.0h
                                 ↑           ↑              ↑
                            soft_deadline  hard_deadline  Kaggle 강제 종료

- soft_deadline: 여기를 넘으면 다음 체크포인트 저장 후 정상 종료한다.
- hard_deadline: 여기를 넘으면 업로드를 포기하고 로컬 저장만 한다.
  (로컬 /kaggle/working 은 노트북 출력으로 남으므로 최악의 경우 수동 회수 가능)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SessionGuard:
    """세션 벽시계 예산을 관리한다.

    limit_hours: Kaggle이 세션을 죽이는 시각 (TPU 9h, GPU 12h)
    save_reserve_minutes: 체크포인트 저장 + 업로드에 남겨둘 시간
    upload_reserve_minutes: 업로드에만 남겨둘 시간 (저장보다 훨씬 오래 걸린다)
    """

    limit_hours: float = 9.0
    save_reserve_minutes: float = 35.0
    upload_reserve_minutes: float = 10.0
    started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        if self.save_reserve_minutes <= self.upload_reserve_minutes:
            raise ValueError("save_reserve는 upload_reserve보다 커야 한다")
        limit = self.limit_hours * 3600
        if self.save_reserve_minutes * 60 >= limit:
            raise ValueError("save_reserve가 세션 한계보다 크다")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def limit_seconds(self) -> float:
        return self.limit_hours * 3600

    @property
    def soft_deadline(self) -> float:
        return self.limit_seconds - self.save_reserve_minutes * 60

    @property
    def hard_deadline(self) -> float:
        return self.limit_seconds - self.upload_reserve_minutes * 60

    @property
    def remaining(self) -> float:
        """정상 종료까지 남은 초."""
        return max(0.0, self.soft_deadline - self.elapsed)

    @property
    def should_stop(self) -> bool:
        """True면 다음 체크포인트를 저장하고 세션을 끝낸다."""
        return self.elapsed >= self.soft_deadline

    @property
    def can_upload(self) -> bool:
        """업로드를 시작해도 안전한가."""
        return self.elapsed < self.hard_deadline

    def fits(self, seconds: float) -> bool:
        """작업 하나를 더 돌릴 시간이 남았는가 (스텝 시간 예측에 쓴다)."""
        return self.elapsed + seconds < self.soft_deadline

    def summary(self) -> str:
        h = lambda s: f"{s/3600:.2f}h"
        return (f"경과 {h(self.elapsed)} / 정상종료 {h(self.soft_deadline)} "
                f"/ 강제종료 {h(self.limit_seconds)} · 남음 {h(self.remaining)}")
