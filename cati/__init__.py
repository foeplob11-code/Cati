"""Cati 학습 인프라.

Kaggle 무료 티어의 9시간 세션 위에서 97시간짜리 런을 돌리기 위한 부품들.
"""

from cati.checkpoint import CheckpointManager, NumpyBackend, OrbaxBackend, auto_backend
from cati.runner import ResumableRun
from cati.session import SessionGuard
from cati.store import KaggleDatasetStore, LocalStore, default_store
from cati.stream import HFSource, ListSource, ResumableStream
from cati.telemetry import BudgetTracker, MetricLog

__all__ = [
    "CheckpointManager", "NumpyBackend", "OrbaxBackend", "auto_backend",
    "ResumableRun",
    "SessionGuard",
    "KaggleDatasetStore", "LocalStore", "default_store",
    "HFSource", "ListSource", "ResumableStream",
    "BudgetTracker", "MetricLog",
]
