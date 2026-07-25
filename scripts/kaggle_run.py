#!/usr/bin/env python3
"""Kaggle 학습을 명령 한 개로 실행한다.

브라우저를 열지 않는다. Kaggle API가 노트북을 올리고 실행까지 시킨다.

    ./train            학습 시작 / 이어하기
    ./train 2          STEP 을 2로 바꿔서 실행 (1=50M 2=100M 3=350M)
    ./train status     진행 상황
    ./train log        로그 보기
    ./train watch      끝날 때까지 지켜보기
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "cati_train.ipynb"
STAGE = ROOT / "artifacts" / "kaggle_push"
OUTDIR = ROOT / "artifacts" / "kaggle_out"
STEP_FILE = ROOT / "artifacts" / ".kaggle_step"
CREDS = Path.home() / ".kaggle" / "kaggle.json"

SLUG = "cati-train"
TITLE = "Cati Train"
TIER_NAMES = {1: "50M (약 1시간)", 2: "100M (약 4.5시간)", 3: "350M (5주, 세션 11번)"}

SETUP_HELP = f"""
Kaggle 인증 파일이 없습니다. 한 번만 해두면 됩니다.

  1. https://www.kaggle.com/settings  접속
  2. API 항목 → [Create New Token] → kaggle.json 다운로드
  3. 아래 두 줄을 터미널에 붙여넣기

     mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json

  그 다음 다시 ./train 를 실행하세요.

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


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([kaggle_bin(), *args], capture_output=True, text=True, **kw)


def clean(text: str) -> str:
    """kaggle CLI가 뱉는 무해한 경고를 걷어낸다."""
    skip = ("NotOpenSSLWarning", "warnings.warn", "urllib3/__init__")
    return "\n".join(l for l in text.splitlines()
                     if l.strip() and not any(s in l for s in skip))


# ---------------------------------------------------------------------------
def read_step() -> int:
    if STEP_FILE.exists():
        try:
            return max(1, min(3, int(STEP_FILE.read_text().strip())))
        except ValueError:
            pass
    return 1


def patch_notebook(step: int) -> dict:
    """노트북의 STEP 값을 바꿔서 반환한다 (원본은 건드리지 않는다)."""
    nb = json.loads(NB.read_text())
    patched = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if line.lstrip().startswith("STEP"):
                _, _, tail = line.partition("#")
                cell["source"][i] = f"STEP = {step}" + (f"   # {tail.strip()}"
                                                        if tail.strip() else "") + "\n"
                patched = True
                break
        if patched:
            break
    if not patched:
        sys.exit("노트북에서 STEP 줄을 찾지 못했습니다. "
                 "python3 scripts/make_notebooks.py 로 다시 만드세요.")
    return nb


def stage(step: int, user: str) -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    (STAGE / NB.name).write_text(
        json.dumps(patch_notebook(step), indent=1, ensure_ascii=False) + "\n")
    (STAGE / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2))
    return STAGE


# ---------------------------------------------------------------------------
def cmd_push(step: int | None) -> int:
    user = username()
    step = step or read_step()
    print(f"STEP {step} — {TIER_NAMES[step]}")
    print(f"대상  {user}/{SLUG}\n")

    d = stage(step, user)
    r = run(["kernels", "push", "-p", str(d)])
    out = clean(r.stdout + r.stderr)

    if r.returncode != 0 and "tpu" in out.lower():
        # 일부 API 버전은 enable_tpu를 모른다. 그 경우 한 번만 UI에서 설정하면 유지된다.
        print("TPU 설정이 API로 안 먹습니다. 가속기 없이 올립니다.")
        meta = json.loads((d / "kernel-metadata.json").read_text())
        meta.pop("enable_tpu", None)
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        r = run(["kernels", "push", "-p", str(d)])
        out = clean(r.stdout + r.stderr)
        print(f"\n⚠️ 노트북 페이지에서 Accelerator를 'TPU VM v3-8'로 한 번만 설정하세요.\n"
              f"   https://www.kaggle.com/code/{user}/{SLUG}/edit\n"
              f"   한 번 설정하면 이후 ./train 에서 유지됩니다.")

    print(out or "(출력 없음)")
    if r.returncode != 0:
        print("\n올리기 실패. 위 메시지를 확인하세요.")
        return r.returncode

    STEP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STEP_FILE.write_text(str(step))
    print(f"\n실행 시작됨.  https://www.kaggle.com/code/{user}/{SLUG}")
    print("  진행 확인:  ./train status")
    print("  로그 보기:  ./train log")
    return 0


def cmd_status() -> int:
    user = username()
    r = run(["kernels", "status", f"{user}/{SLUG}"])
    print(clean(r.stdout + r.stderr) or "(상태 없음)")
    return r.returncode


def cmd_log(tail: int = 60) -> int:
    user = username()
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True)
    r = run(["kernels", "output", f"{user}/{SLUG}", "-p", str(OUTDIR)])
    if r.returncode != 0:
        print(clean(r.stdout + r.stderr))
        print("\n아직 출력이 없습니다. ./train status 로 실행 중인지 확인하세요.")
        return r.returncode

    logs = sorted(OUTDIR.rglob("*.log")) + sorted(OUTDIR.rglob("*.txt"))
    if not logs:
        print(f"출력 파일: {[p.name for p in OUTDIR.rglob('*') if p.is_file()][:10]}")
        return 0
    text = logs[0].read_text(errors="replace").splitlines()
    print(f"─── {logs[0].name} (마지막 {tail}줄) ───")
    print("\n".join(text[-tail:]))
    return 0


def cmd_watch(interval: int = 300) -> int:
    user = username()
    print(f"{user}/{SLUG} 를 {interval//60}분마다 확인합니다. Ctrl+C 로 중단.\n")
    try:
        while True:
            r = run(["kernels", "status", f"{user}/{SLUG}"])
            line = clean(r.stdout + r.stderr).replace("\n", " ")
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] {line}")
            if any(w in line.lower() for w in ("complete", "error", "cancel")):
                print("\n종료됨.  ./train log  로 결과 확인")
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n중단. Kaggle에서는 계속 돌고 있습니다.")
        return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        return cmd_push(None)
    a = args[0]
    if a in ("1", "2", "3"):
        return cmd_push(int(a))
    if a == "status":
        return cmd_status()
    if a == "log":
        return cmd_log(int(args[1]) if len(args) > 1 else 60)
    if a == "watch":
        return cmd_watch()
    if a in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    print(f"모르는 명령: {a}\n{__doc__}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
