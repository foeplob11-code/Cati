"""재개 가능한 데이터 스트림.

체크포인트에서 가장 틀리기 쉬운 부분이다. 파라미터와 옵티마이저만 복원하고
데이터 위치를 복원하지 않으면, 재개할 때마다 코퍼스 앞부분을 다시 먹는다.
97시간짜리 런이 9시간마다 끊긴다면 앞쪽 데이터를 11번 반복 학습하고
뒤쪽은 한 번도 못 보게 된다. **손실 곡선만 봐서는 이 버그가 보이지 않는다.**

복원해야 하는 상태는 네 가지다.
  1. 각 소스별 소비 위치
  2. 소스 선택 RNG 상태 (어떤 순서로 섞였는지)
  3. 고갈된 소스 목록 — 이게 없으면 재개 시 가중치 분포가 달라진다
  4. 누적 문서/토큰 수
"""
from __future__ import annotations

import random
from typing import Iterator, Protocol


class StreamSource(Protocol):
    """재개 가능한 단일 데이터 소스."""

    name: str
    repeat: bool

    def iterator(self) -> Iterator[str]: ...
    def reset(self) -> None: ...
    def probe(self) -> tuple[bool, str]: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...


def usable_sources(sources: list, weights: list[float], verbose: bool = True):
    """접근 가능한 소스만 남기고 가중치를 재정규화한다.

    공개 데이터셋은 예고 없이 gated로 바뀌거나 이름이 바뀐다.
    (bigcode/the-stack-smol 이 2026-07에 그렇게 됐다)
    그때 5주짜리 런이 죽으면 안 되므로, 못 쓰는 소스는 건너뛰고 남은 것으로 계속한다.
    """
    keep_s, keep_w, dropped = [], [], []
    for s, w in zip(sources, weights):
        ok, why = s.probe()
        if ok:
            keep_s.append(s)
            keep_w.append(w)
        else:
            dropped.append((s.name, why))

    if not keep_s:
        raise RuntimeError("쓸 수 있는 데이터 소스가 하나도 없다:\n  " +
                           "\n  ".join(f"{n}: {w}" for n, w in dropped))

    total = sum(keep_w)
    keep_w = [w / total for w in keep_w]

    if verbose:
        for name, why in dropped:
            print(f"  [건너뜀] {name}: {why}")
        if dropped:
            print(f"  가중치 재정규화: " +
                  ", ".join(f"{s.name} {w:.2f}" for s, w in zip(keep_s, keep_w)))
    return keep_s, keep_w, dropped


class ListSource:
    """테스트용 결정적 소스. 재개 정확성 검증에 쓴다."""

    def __init__(self, name: str, items: list[str], repeat: bool = False):
        self.name = name
        self.repeat = repeat
        self._items = items
        self._pos = 0

    def iterator(self) -> Iterator[str]:
        while self._pos < len(self._items):
            item = self._items[self._pos]
            self._pos += 1
            yield item

    def reset(self) -> None:
        self._pos = 0

    def probe(self) -> tuple[bool, str]:
        return bool(self._items), "" if self._items else "빈 소스"

    def state_dict(self) -> dict:
        return {"pos": self._pos}

    def load_state_dict(self, state: dict) -> None:
        self._pos = state["pos"]


class HFSource:
    """HuggingFace 스트리밍 데이터셋 래퍼.

    datasets 2.20+ 의 IterableDataset.state_dict()/load_state_dict() 를 쓴다.
    그게 없으면 건너뛰기로 대체하는데, 재개마다 앞부분을 다시 읽어야 해서
    세션이 늘어날수록 비용이 커진다.

    ⚠️ W1에 네이티브 경로가 실제로 동작하는지 반드시 확인할 것.
       (state_dict() 존재 여부가 아니라, 재개 후 문서가 이어지는지를 확인)
    """

    def __init__(self, name: str, repo: str, config: str | None = None,
                 field: str = "text", min_chars: int = 200,
                 split: str = "train", repeat: bool = True):
        self.name = name
        self.repeat = repeat
        self.repo, self.config, self.field = repo, config, field
        self.min_chars, self.split = min_chars, split
        self._ds = None
        self._skip = 0
        self._native = None

    def _dataset(self):
        if self._ds is None:
            from datasets import load_dataset
            self._ds = load_dataset(self.repo, self.config,
                                    split=self.split, streaming=True)
            self._native = hasattr(self._ds, "state_dict")
            if not self._native:
                print(f"  [경고] {self.repo}: 네이티브 재개 미지원 → 건너뛰기로 대체. "
                      "재개마다 앞부분을 다시 읽는다.")
        return self._ds

    def iterator(self) -> Iterator[str]:
        ds = self._dataset()
        skipped = 0
        for row in ds:
            # 네이티브 상태 복원을 못 쓰는 경우에만 앞부분을 흘려보낸다.
            if skipped < self._skip:
                skipped += 1
                continue
            text = row.get(self.field) or ""
            if len(text) < self.min_chars:
                continue
            self._skip += 1
            yield text

    def reset(self) -> None:
        self._skip = 0
        self._ds = None

    def probe(self) -> tuple[bool, str]:
        """실제로 열어본다. gated·삭제·개명을 여기서 잡는다."""
        try:
            self._dataset()
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"

    def state_dict(self) -> dict:
        ds = self._dataset()
        if self._native:
            return {"native": ds.state_dict(), "skip": self._skip}
        return {"skip": self._skip}

    def load_state_dict(self, state: dict) -> None:
        ds = self._dataset()
        self._skip = state.get("skip", 0)
        if "native" in state and self._native:
            ds.load_state_dict(state["native"])
            self._skip = 0


class ResumableStream:
    """가중치에 따라 여러 소스를 섞는, 재개 가능한 스트림.

    repeat=True 인 소스는 고갈되면 처음부터 다시 읽는다 (다중 에폭).
    repeat=False 인 소스는 고갈되면 제외되고, 그 사실이 체크포인트에 남는다.
    """

    def __init__(self, sources: list[StreamSource], weights: list[float], seed: int = 0):
        if len(sources) != len(weights):
            raise ValueError("sources와 weights 길이가 다르다")
        if not sources:
            raise ValueError("소스가 비어 있다")
        self.sources = sources
        self.weights = list(weights)
        self.seed = seed
        self._rng = random.Random(seed)
        self._iters: dict[str, Iterator[str]] = {}
        self._exhausted: set[str] = set()
        self.epochs: dict[str, int] = {s.name: 0 for s in sources}
        self.docs_seen = 0
        self.chars_seen = 0

    def _iter_for(self, src: StreamSource) -> Iterator[str]:
        if src.name not in self._iters:
            self._iters[src.name] = src.iterator()
        return self._iters[src.name]

    def __iter__(self) -> Iterator[tuple[str, str]]:
        """(소스이름, 텍스트) 를 내보낸다."""
        while True:
            live = [(s, w) for s, w in zip(self.sources, self.weights)
                    if s.name not in self._exhausted]
            if not live:
                return
            (src,) = self._rng.choices([s for s, _ in live],
                                       weights=[w for _, w in live], k=1)
            try:
                text = next(self._iter_for(src))
            except StopIteration:
                if src.repeat:
                    # 다음 에폭. RNG는 건드리지 않는다.
                    src.reset()
                    self._iters.pop(src.name, None)
                    self.epochs[src.name] += 1
                    continue
                self._exhausted.add(src.name)
                continue
            self.docs_seen += 1
            self.chars_seen += len(text)
            yield src.name, text

    def state_dict(self) -> dict:
        version, internal, gauss = self._rng.getstate()
        return {
            "seed": self.seed,
            "docs_seen": self.docs_seen,
            "chars_seen": self.chars_seen,
            "exhausted": sorted(self._exhausted),
            "epochs": dict(self.epochs),
            # random.Random 상태는 튜플이라 JSON을 위해 리스트로 바꾼다.
            "rng": {"version": version, "internal": list(internal), "gauss": gauss},
            "sources": {s.name: s.state_dict() for s in self.sources},
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed = state["seed"]
        self.docs_seen = state["docs_seen"]
        self.chars_seen = state["chars_seen"]
        self._exhausted = set(state.get("exhausted", []))
        self.epochs = dict(state.get("epochs", {s.name: 0 for s in self.sources}))
        r = state["rng"]
        self._rng.setstate((r["version"], tuple(r["internal"]), r["gauss"]))
        for s in self.sources:
            if s.name in state["sources"]:
                s.load_state_dict(state["sources"][s.name])
        # 소스 위치가 바뀌었으므로 이터레이터를 다시 만든다.
        self._iters = {}
