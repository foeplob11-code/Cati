"""문서 스트림 → 고정 길이 학습 배치.

두 가지를 반드시 지켜야 한다.

1. **남은 토큰 버퍼도 체크포인트에 저장한다.**
   문서 중간에서 세션이 끊기는 건 정상이다. 버퍼를 안 저장하면 재개할 때
   그 문서의 앞부분이 버려지고, 데이터 위치와 실제 학습량이 어긋난다.

2. **상태를 배치와 함께 내보낸다.**
   프리페치(미리 읽기)를 하면 큐에 든 배치는 아직 학습에 안 쓰였다.
   생산 시점 상태를 그대로 저장하면 "안 배운 데이터를 배웠다"고 기록된다.
   그래서 각 배치에 그 시점의 상태를 붙여서 내보낸다.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Iterator

import numpy as np


class TokenPacker:
    """(소스, 텍스트) 스트림을 (batch_size, seq_len+1) int32 배치로 만든다.

    seq_len+1 인 이유: 다음 토큰 예측이라 inputs=[:-1], targets=[1:] 로 쪼갠다.
    """

    def __init__(self, stream, tokenizer, seq_len: int, batch_size: int,
                 eos_id: int, dtype=np.int32):
        self.stream = stream
        self.tok = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eos_id = eos_id
        self.dtype = dtype

        self._buf: list[int] = []
        self._it: Iterator[tuple[str, str]] | None = None

        self.tokens_produced = 0
        self.batches_produced = 0
        # 스트리밍이 TPU 계산을 먹여살릴 수 있는지 판단하는 근거.
        self.fetch_seconds = 0.0

    # 러너가 스트림처럼 다룰 수 있도록 위임한다.
    @property
    def docs_seen(self) -> int:
        return self.stream.docs_seen

    @property
    def epochs(self) -> dict:
        return self.stream.epochs

    @property
    def row_len(self) -> int:
        return self.seq_len + 1

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.seq_len      # 예측 대상 토큰 수

    def _need(self) -> int:
        return self.batch_size * self.row_len

    def batches(self) -> Iterator[tuple[np.ndarray, dict]]:
        """(배치, 그 시점의 재개 상태) 를 내보낸다."""
        if self._it is None:
            self._it = iter(self.stream)

        while True:
            t0 = time.monotonic()
            while len(self._buf) < self._need():
                try:
                    _, text = next(self._it)
                except StopIteration:
                    self.fetch_seconds += time.monotonic() - t0
                    return
                self._buf.extend(self.tok.encode(text).ids)
                self._buf.append(self.eos_id)
            self.fetch_seconds += time.monotonic() - t0

            flat = np.asarray(self._buf[:self._need()], dtype=self.dtype)
            self._buf = self._buf[self._need():]
            batch = flat.reshape(self.batch_size, self.row_len)

            self.tokens_produced += self.tokens_per_batch
            self.batches_produced += 1
            yield batch, self.state_dict()

    # ---- 재개 --------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "stream": self.stream.state_dict(),
            "buffer": list(self._buf),          # ← 이걸 빼먹으면 데이터가 어긋난다
            "tokens_produced": self.tokens_produced,
            "batches_produced": self.batches_produced,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("seq_len") != self.seq_len:
            raise ValueError(
                f"seq_len 불일치: 체크포인트 {state.get('seq_len')} / 현재 {self.seq_len}. "
                "시퀀스 길이를 바꾸려면 어닐링 단계에서 명시적으로 전환할 것.")
        self.stream.load_state_dict(state["stream"])
        self._buf = list(state["buffer"])
        self.tokens_produced = state["tokens_produced"]
        self.batches_produced = state["batches_produced"]
        self._it = None                          # 스트림 위치가 바뀌었으니 다시 만든다

    # ---- 진단 --------------------------------------------------------
    def throughput(self, total_seconds: float) -> dict:
        """스트리밍이 병목인지 판단한다.

        fetch_fraction 이 크면 TPU가 데이터를 기다리며 놀고 있다는 뜻이고,
        그때는 미리 토큰화해서 Kaggle Dataset에 저장하는 방식으로 바꿔야 한다.
        """
        frac = self.fetch_seconds / total_seconds if total_seconds > 0 else 0.0
        return {
            "tokens_produced": self.tokens_produced,
            "fetch_seconds": self.fetch_seconds,
            "fetch_fraction": frac,
            "fetch_tokens_per_sec": (self.tokens_produced / self.fetch_seconds
                                     if self.fetch_seconds > 0 else 0.0),
            "verdict": ("스트리밍 충분" if frac < 0.05 else
                        "주의 — 사전 토큰화 검토" if frac < 0.15 else
                        "병목 — 사전 토큰화 필요"),
        }


def prefetch(batches: Iterator[tuple[np.ndarray, dict]], depth: int = 4):
    """배치를 백그라운드 스레드에서 미리 만들어 둔다.

    토크나이징(CPU)과 학습(TPU)을 겹치게 해서 TPU가 데이터를 기다리지 않게 한다.
    상태가 배치와 함께 오므로, 큐에 남은 배치는 체크포인트에 반영되지 않는다 —
    즉 세션이 끊겨도 "안 배운 데이터를 배웠다"고 기록되지 않는다.
    """
    q: queue.Queue = queue.Queue(maxsize=depth)
    sentinel = object()

    def worker():
        try:
            for item in batches:
                q.put(item)
        except Exception as e:            # 스레드에서 죽으면 조용히 멈추므로 전달한다
            q.put(e)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item
