#!/usr/bin/env python3
"""Kaggle에서 토크나이저 학습을 명령 한 개로 돌린다. (선택 사항)

**학습 본체는 Colab에서 한다** (notebooks/cati_colab.ipynb).
Kaggle에는 TPU 옵션이 없어서 200M 학습에 111시간이 걸린다.

이 스크립트는 TPU가 필요 없는 토크나이저만 Kaggle에서 만든다.
귀한 Colab 시간 30분을 아끼려는 목적이고, 건너뛰어도 된다 —
Colab 노트북이 토크나이저가 없으면 알아서 만든다.

    ./tokenizer            토크나이저 학습 시작
    ./tokenizer status     진행 상황
    ./tokenizer log        로그 보기
    ./tokenizer get        완성된 tokenizer.json 가져오기
    ./tokenizer watch      끝날 때까지 지켜보기
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "cati_tokenizer.ipynb"
STAGE = ROOT / "artifacts" / "kaggle_push"
OUTDIR = ROOT / "artifacts" / "kaggle_out"
CREDS = Path.home() / ".kaggle" / "kaggle.json"

SLUG = "cati-tokenizer"
TITLE = "Cati Tokenizer"

SETUP_HELP = f"""
Kaggle 인증 파일이 없습니다. 한 번만 해두면 됩니다.

  1. https://www.kaggle.com/settings  접속
  2. API 항목 → [Create New Token] → kaggle.json 다운로드
  3. 아래를 터미널에 붙여넣기

     mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

안 되면 그냥 건너뛰세요. Colab 노트북이 토크나이저를 알아서 만듭니다.

(파일 위치: {CREDS})
"""


def kaggle_bin() -> str:
    local = ROOT / ".venv" / "bin" / "kaggle"
    if local.exists():
        return str(local)
    found = shutil.which("kaggle")
    if found:
        return found
    sys.exit("kaggle CLI가 없습니다:  .venv/bin/pip install kaggle")


def username() -> str:
    if not CREDS.exists():
        sys.exit(SETUP_HELP)
    try:
        name = json.loads(CREDS.read_text()).get("username")
    except json.JSONDecodeError:
        sys.exit(f"{CREDS} 를 읽을 수 없습니다. 다시 다운로드하세요.")
    if not name:
        sys.exit(f"{CREDS} 에 username이 없습니다. 다시 다운로드하세요.")
    return name


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([kaggle_bin(), *args], capture_output=True, text=True)


def clean(text: str) -> str:
    """kaggle CLI가 뱉는 무해한 경고를 걷어낸다."""
    skip = ("NotOpenSSLWarning", "warnings.warn", "urllib3/__init__")
    return "\n".join(l for l in text.splitlines()
                     if l.strip() and not any(s in l for s in skip))


def stage(user: str) -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(NB, STAGE / NB.name)
    (STAGE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,      # 토크나이저는 CPU 작업이다
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    return STAGE


def cmd_push() -> int:
    user = username()
    print(f"토크나이저 학습 (약 30분) → {user}/{SLUG}\n")
    r = run(["kernels", "push", "-p", str(stage(user))])
    print(clean(r.stdout + r.stderr) or "(출력 없음)")
    if r.returncode != 0:
        print("\n올리기 실패. 위 메시지를 확인하세요.")
        return r.returncode
    print(f"\n실행 시작됨.  https://www.kaggle.com/code/{user}/{SLUG}")
    print("  진행 확인:  ./tokenizer status")
    print("  결과 받기:  ./tokenizer get")
    return 0


def cmd_status() -> int:
    user = username()
    r = run(["kernels", "status", f"{user}/{SLUG}"])
    print(clean(r.stdout + r.stderr) or "(상태 없음)")
    return r.returncode


def _fetch_output() -> Path | None:
    user = username()
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True)
    r = run(["kernels", "output", f"{user}/{SLUG}", "-p", str(OUTDIR)])
    if r.returncode != 0:
        print(clean(r.stdout + r.stderr))
        print("\n아직 출력이 없습니다. ./tokenizer status 로 확인하세요.")
        return None
    return OUTDIR


def cmd_log(tail: int = 60) -> int:
    if _fetch_output() is None:
        return 1
    logs = sorted(OUTDIR.rglob("*.log")) + sorted(OUTDIR.rglob("*.txt"))
    if not logs:
        print(f"출력 파일: {[p.name for p in OUTDIR.rglob('*') if p.is_file()][:10]}")
        return 0
    lines = logs[0].read_text(errors="replace").splitlines()
    print(f"─── {logs[0].name} (마지막 {tail}줄) ───")
    print("\n".join(lines[-tail:]))
    return 0


def cmd_get() -> int:
    """완성된 토크나이저를 저장소로 가져온다."""
    if _fetch_output() is None:
        return 1
    hits = list(OUTDIR.rglob("tokenizer.json"))
    if not hits:
        print("tokenizer.json 이 없습니다. 아직 학습 중일 수 있습니다.")
        return 1
    dst = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hits[0], dst)
    print(f"받았습니다: {dst.relative_to(ROOT)}  ({dst.stat().st_size/1e6:.2f} MB)")
    print("\n이걸 커밋해두면 Colab이 코드와 함께 받아갑니다:")
    print("  git add -f artifacts/tokenizer/tokenizer.json")
    print("  git commit -m '토크나이저 확정' && git push")
    return 0


def cmd_watch(interval: int = 120) -> int:
    user = username()
    print(f"{user}/{SLUG} 를 {interval//60}분마다 확인합니다. Ctrl+C 로 중단.\n")
    try:
        while True:
            r = run(["kernels", "status", f"{user}/{SLUG}"])
            line = clean(r.stdout + r.stderr).replace("\n", " ")
            print(f"[{time.strftime('%H:%M:%S')}] {line}")
            if any(w in line.lower() for w in ("complete", "error", "cancel")):
                print("\n종료됨.  ./tokenizer get  으로 결과 받기")
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n중단. Kaggle에서는 계속 돌고 있습니다.")
        return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return cmd_push()
    a = args[0]
    if a == "status":
        return cmd_status()
    if a == "log":
        return cmd_log(int(args[1]) if len(args) > 1 else 60)
    if a == "get":
        return cmd_get()
    if a == "watch":
        return cmd_watch()
    if a in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    print(f"모르는 명령: {a}\n{__doc__}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
