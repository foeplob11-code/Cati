"""체크포인트 저장/복원.

원칙 네 가지.

1. **원자적 쓰기** — 임시 디렉터리에 다 쓴 뒤 rename 한다. Kaggle이 저장 도중에
   세션을 죽여도 반쯤 쓰인 체크포인트가 남지 않는다.
2. **로컬 저장과 원격 발행을 분리** — 저장은 자주(싸다), 발행은 세션 끝에 한 번(수 GB 업로드).
3. **배열과 메타데이터를 분리** — 메타(step/데이터위치/RNG)는 JSON이라
   체크포인트를 열지 않고도 사람이 읽고 고칠 수 있다.
4. **구조는 target에서 복원한다** ← 아래 참고

pytree 구조 문제
----------------
옵티마이저 상태는 dict가 아니라 NamedTuple 트리다 (optax의 ScaleByAdamState 등).
배열만 저장하면 구조 정보가 사라져서, 복원할 때 dict로 되돌아오고
`opt.update()` 가 `'dict' object has no attribute 'mu'` 로 죽는다.

그래서 백엔드는 **평평한 {경로: 배열}** 만 다루고, 구조는 호출자가 주는
target(갓 초기화한 같은 모양의 트리)에서 가져온다. 백엔드 두 개가 구조를
각자 처리하지 않으므로 경로가 하나뿐이다.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

META_NAME = "meta.json"
LATEST = "LATEST"


# --------------------------------------------------------------------------
# pytree ↔ 평평한 dict
# --------------------------------------------------------------------------
def _key_str(k) -> str:
    # jax의 키 종류마다 속성 이름이 다르다: DictKey.key / GetAttrKey.name / SequenceKey.idx
    for attr in ("key", "name", "idx"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def _dict_flatten(tree: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_dict_flatten(v, key))
        else:
            out[key] = v
    return out


def _dict_unflatten(flat: dict[str, Any]) -> dict:
    out: dict = {}
    for key, v in flat.items():
        parts = key.split("/")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def tree_to_flat(tree) -> dict[str, Any]:
    """임의의 pytree를 {경로문자열: 배열} 로 만든다."""
    try:
        import jax
    except ImportError:
        return _dict_flatten(tree)
    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {"/".join(_key_str(k) for k in path): leaf for path, leaf in pairs}


def flat_to_tree(flat: dict[str, Any], target=None):
    """평평한 dict를 target과 같은 구조로 되돌린다.

    target이 없으면 dict 트리로만 복원한다 (구조가 dict뿐일 때만 정확하다).
    """
    if target is None:
        return _dict_unflatten(flat)

    import jax
    pairs, treedef = jax.tree_util.tree_flatten_with_path(target)
    leaves, used = [], set()
    for path, _ in pairs:
        key = "/".join(_key_str(k) for k in path)
        if key not in flat:
            raise KeyError(f"체크포인트에 '{key}' 가 없다. "
                           "모델/옵티마이저 설정이 저장 시점과 다르다.")
        used.add(key)
        leaves.append(flat[key])

    # 양방향으로 엄격하게 본다. target이 체크포인트의 부분집합이면 조용히
    # 일부만 복원되는데, 옵티마이저 상태가 통째로 빠지는 사고가 여기서 난다.
    extra = sorted(set(flat) - used)
    if extra:
        raise KeyError(
            f"체크포인트에 target이 받지 않는 항목이 {len(extra)}개 있다: "
            f"{extra[:4]}{' ...' if len(extra) > 4 else ''}. "
            "모델/옵티마이저 설정이 저장 시점과 다르다.")
    return jax.tree_util.tree_unflatten(treedef, leaves)


# --------------------------------------------------------------------------
# 배열 백엔드 — 평평한 dict만 다룬다
# --------------------------------------------------------------------------
class NumpyBackend:
    """npz 백엔드. 로컬 테스트와 JAX 없는 환경에서 쓴다."""

    name = "numpy"

    def save(self, path: Path, flat: dict[str, Any]) -> None:
        import numpy as np
        np.savez(path / "arrays.npz", **{k: np.asarray(v) for k, v in flat.items()})

    def load(self, path: Path, target_flat: dict | None = None) -> dict[str, Any]:
        import numpy as np
        with np.load(path / "arrays.npz", allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


class OrbaxBackend:
    """JAX/TPU 백엔드. 샤딩된 배열을 그대로 저장한다."""

    name = "orbax"

    def __init__(self):
        import orbax.checkpoint as ocp
        self._ocp = ocp

    def save(self, path: Path, flat: dict[str, Any]) -> None:
        ckptr = self._ocp.StandardCheckpointer()
        ckptr.save(path.absolute() / "arrays", flat)
        ckptr.wait_until_finished()

    def load(self, path: Path, target_flat: dict | None = None) -> dict[str, Any]:
        ckptr = self._ocp.StandardCheckpointer()
        d = path.absolute() / "arrays"
        if target_flat is None:
            # target 없이 복원하면 Orbax가 샤딩/토폴로지를 모른다고 경고한다.
            # 배열이 device 0 에 몰릴 수 있어 다중 디바이스에서 비효율적이다.
            return ckptr.restore(d)
        return ckptr.restore(d, target_flat)


def auto_backend():
    """JAX가 있으면 Orbax, 없으면 numpy."""
    try:
        return OrbaxBackend()
    except Exception:
        return NumpyBackend()


# --------------------------------------------------------------------------
# 매니저
# --------------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, root: Path | str, store=None, backend=None, keep_last: int = 2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.backend = backend or auto_backend()
        self.keep_last = max(1, keep_last)
        self._last_saved: Path | None = None

    # ---- 조회 ----------------------------------------------------------
    def _step_dirs(self, root: Path | None = None) -> list[Path]:
        root = root or self.root
        return sorted(d for d in root.glob("step_*") if (d / META_NAME).exists())

    def latest_dir(self, root: Path | None = None) -> Path | None:
        dirs = self._step_dirs(root)
        return dirs[-1] if dirs else None

    # ---- 저장 ----------------------------------------------------------
    def save(self, step: int, arrays: dict, meta: dict) -> Path:
        tmp = self.root / f".tmp_step_{step:09d}"
        final = self.root / f"step_{step:09d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        full_meta = {
            "step": step,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": self.backend.name,
            **meta,
        }
        self.backend.save(tmp, tree_to_flat(arrays))
        (tmp / META_NAME).write_text(json.dumps(full_meta, indent=2, ensure_ascii=False))

        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)                       # ← 여기서부터 유효한 체크포인트
        (self.root / LATEST).write_text(final.name)
        self._last_saved = final
        self._prune()
        return final

    def _prune(self) -> None:
        for d in self._step_dirs()[:-self.keep_last]:
            shutil.rmtree(d, ignore_errors=True)

    # ---- 복원 ----------------------------------------------------------
    def load_latest(self, target=None) -> tuple[dict, dict] | None:
        """(arrays, meta) 또는 None(새 런).

        target: 갓 초기화한 같은 모양의 트리. 옵티마이저 상태 같은 NamedTuple
                구조를 되살리려면 반드시 넘겨야 한다.
        """
        d = self.latest_dir()
        if d is None and self.store is not None:
            fetched = self.store.fetch_latest(self.root / "_fetched")
            if fetched is not None:
                d = self.latest_dir(Path(fetched)) or (
                    Path(fetched) if (Path(fetched) / META_NAME).exists() else None)
        if d is None:
            return None

        meta = json.loads((d / META_NAME).read_text())
        if meta.get("backend") != self.backend.name:
            raise RuntimeError(
                f"체크포인트 백엔드 불일치: 저장 {meta.get('backend')} / 현재 {self.backend.name}. "
                "TPU와 로컬 체크포인트를 섞지 말 것.")
        # target을 평평하게 넘겨 백엔드가 형상·dtype·샤딩을 알고 복원하게 한다.
        target_flat = tree_to_flat(target) if target is not None else None
        return flat_to_tree(self.backend.load(d, target_flat), target), meta

    # ---- 발행 ----------------------------------------------------------
    def publish(self, message: str | None = None) -> bool:
        """세션 종료 시 한 번만 호출한다 (업로드가 비싸다)."""
        if self.store is None:
            return False
        d = self._last_saved or self.latest_dir()
        if d is None:
            return False
        return self.store.publish(d, message or f"cati {d.name}")
