"""체크포인트 영속화 백엔드.

Kaggle에서 `/kaggle/working` 은 세션이 끝나면 노트북 출력으로 남지만,
다음 세션이 그걸 자동으로 집어오지는 않는다. 세션 간 인계는
**Kaggle Dataset** 을 거치는 것이 확실하다.

읽기는 마운트된 `/kaggle/input/<slug>` 를 우선한다 (다운로드 없이 즉시 접근).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

KAGGLE_INPUT = Path("/kaggle/input")


class CheckpointStore(Protocol):
    def publish(self, local_dir: Path, message: str) -> bool: ...
    def fetch_latest(self, dest: Path) -> Path | None: ...


class LocalStore:
    """로컬 디렉터리 백엔드. 테스트와 Kaggle 외 환경에서 쓴다."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def publish(self, local_dir: Path, message: str) -> bool:
        self.root.mkdir(parents=True, exist_ok=True)
        dest = self.root / Path(local_dir).name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local_dir, dest)
        (self.root / "LATEST").write_text(dest.name)
        return True

    def fetch_latest(self, dest: Path) -> Path | None:
        marker = self.root / "LATEST"
        if not marker.exists():
            return None
        src = self.root / marker.read_text().strip()
        if not src.exists():
            return None
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return dest


class KaggleDatasetStore:
    """Kaggle Dataset을 세션 간 인계 매체로 쓴다.

    `~/.kaggle/kaggle.json` 또는 KAGGLE_USERNAME/KAGGLE_KEY 환경변수가 필요하다.
    Kaggle 노트북 안에서는 Add-ons > Secrets 로 주입한다.

    ⚠️ 350M 체크포인트는 4.5GB다. 업로드가 수 분~수십 분 걸리므로
       스텝마다 publish 하지 말고 **세션 종료 시 한 번만** 호출한다.
    """

    def __init__(self, owner: str, slug: str, dry_run: bool = False):
        self.owner, self.slug = owner, slug
        self.ref = f"{owner}/{slug}"
        self.dry_run = dry_run

    # ---- 내부 ----------------------------------------------------------
    def _run(self, args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        if self.dry_run:
            print("  [dry-run]", " ".join(args))
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=3600)

    def _write_metadata(self, d: Path) -> None:
        meta = d / "dataset-metadata.json"
        if not meta.exists():
            meta.write_text(json.dumps({
                "title": self.slug,
                "id": self.ref,
                "licenses": [{"name": "CC0-1.0"}],
            }, indent=2))

    def _exists(self) -> bool:
        r = self._run(["kaggle", "datasets", "status", self.ref])
        return r.returncode == 0

    # ---- 공개 API ------------------------------------------------------
    def publish(self, local_dir: Path, message: str) -> bool:
        local_dir = Path(local_dir)
        self._write_metadata(local_dir)
        args = (["kaggle", "datasets", "version", "-p", str(local_dir),
                 "-m", message, "--dir-mode", "zip"] if self._exists() else
                ["kaggle", "datasets", "create", "-p", str(local_dir), "--dir-mode", "zip"])
        r = self._run(args)
        if r.returncode != 0:
            print(f"  [경고] Kaggle 업로드 실패: {r.stderr.strip()[:300]}")
            print("  로컬 체크포인트는 남아 있다. 노트북 출력에서 회수할 것.")
            return False
        return True

    def fetch_latest(self, dest: Path) -> Path | None:
        # 1) 마운트된 입력이 있으면 그대로 쓴다 (다운로드 불필요)
        mounted = KAGGLE_INPUT / self.slug
        if mounted.exists():
            print(f"  마운트된 데이터셋 사용: {mounted}")
            return mounted

        # 2) 없으면 API로 내려받는다
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        r = self._run(["kaggle", "datasets", "download", "-d", self.ref,
                       "-p", str(dest), "--unzip"])
        if r.returncode != 0:
            print(f"  이전 체크포인트 없음 (새 런으로 시작): {r.stderr.strip()[:200]}")
            return None
        return dest


class HFHubStore:
    """Hugging Face Hub을 세션 간 인계 매체로 쓴다.

    Google Drive 대안. 이쪽이 나은 점:
      · 파일 크기 제한이 없다 (Drive 무료는 15GB 전체 한도)
      · 최종 모델은 어차피 HF에 올려야 GGUF로 변환하므로 가는 길에 있다
      · 토큰이 문자열이라 파일 다운로드가 필요 없다

    토큰: hf.co/settings/tokens 에서 write 권한으로 만든다.
    Colab이면 좌측 🔑(Secrets)에 HF_TOKEN 으로 넣으면 자동으로 잡힌다.

    체크포인트는 항상 같은 경로(checkpoint/)에 덮어쓴다. 스텝별로 쌓으면
    저장소가 계속 커진다.
    """

    PATH_IN_REPO = "checkpoint"

    def __init__(self, repo_id: str, token: str | None = None, private: bool = True):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self._ensured = False

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def _ensure_repo(self) -> None:
        if self._ensured:
            return
        self._api().create_repo(self.repo_id, private=self.private,
                                exist_ok=True, repo_type="model")
        self._ensured = True

    def publish(self, local_dir: Path, message: str) -> bool:
        try:
            self._ensure_repo()
            self._api().upload_folder(
                folder_path=str(local_dir),
                path_in_repo=self.PATH_IN_REPO,
                repo_id=self.repo_id,
                commit_message=message,
                # 이전 체크포인트의 잔여 파일을 남기지 않는다
                delete_patterns="*",
            )
            return True
        except Exception as e:
            print(f"  [경고] HF 업로드 실패: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
            print("  로컬 체크포인트는 남아 있다.")
            return False

    def fetch_latest(self, dest: Path) -> Path | None:
        from huggingface_hub import snapshot_download
        try:
            got = snapshot_download(
                repo_id=self.repo_id, repo_type="model", token=self.token,
                local_dir=str(dest), allow_patterns=f"{self.PATH_IN_REPO}/**")
        except Exception as e:
            print(f"  이전 체크포인트 없음 (새 런으로 시작): "
                  f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")
            return None
        d = Path(got) / self.PATH_IN_REPO
        return d if (d / "meta.json").exists() else None


def default_store(dry_run: bool = False) -> CheckpointStore:
    """환경변수를 보고 스토어를 고른다.

    CATI_HF_REPO      → Hugging Face Hub  (예: "사용자명/cati-ckpt-200m")
    KAGGLE_USERNAME   → Kaggle Dataset
    CATI_CKPT_STORE   → 로컬 디렉터리 (Google Drive 마운트 경로 등)
    """
    hf_repo = os.environ.get("CATI_HF_REPO")
    if hf_repo:
        return HFHubStore(hf_repo)

    owner = os.environ.get("KAGGLE_USERNAME")
    if owner:
        slug = os.environ.get("CATI_CKPT_DATASET", "cati-checkpoints")
        return KaggleDatasetStore(owner, slug, dry_run=dry_run)

    return LocalStore(Path(os.environ.get("CATI_CKPT_STORE", "artifacts/store")))
