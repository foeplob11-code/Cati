#!/usr/bin/env python3
"""Kaggle 노트북 생성기.

.ipynb 를 손으로 쓰면 JSON 이스케이프 때문에 금방 깨진다. 여기서 만든다.

    python3 scripts/make_notebooks.py   →  notebooks/cati_train.ipynb

설계 목표는 **사람 손을 줄이는 것**이다. 5주 동안 세션을 13번 돌려야 하므로
매 세션이 "Save & Run All 누르기" 하나로 끝나야 한다. 그래서
  · 셀을 3개로 줄였다 (준비 / 학습 / 결과)
  · 고칠 것은 맨 위 숫자 하나뿐이다 (STEP = 1 → 2 → 3)
  · 없는 것(토크나이저·체크포인트)은 알아서 만들거나 찾는다
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
GITHUB_URL = "https://github.com/foeplob11-code/Cati.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


PREPARE = f'''
# ══ 준비 ══ 코드·패키지·토크나이저를 알아서 챙긴다. 볼 것 없다.
import os, shutil, socket, subprocess, sys
from pathlib import Path

TIERS = ["configs/tier0_50m.json", "configs/tier1_100m.json", "configs/tier2_350m.json"]
SHORT = ["50m", "100m", "350m"]
TIER, NAME = TIERS[STEP - 1], SHORT[STEP - 1]

# ── 코드 ────────────────────────────────────────────────────────
CATI = Path("/kaggle/working/Cati")
if CATI.exists():
    subprocess.run(["git", "-C", str(CATI), "pull", "-q"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1",
                    "{GITHUB_URL}", str(CATI)], check=True)
os.chdir(CATI)
sys.path.insert(0, str(CATI))

# ── 패키지 ──────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tokenizers>=0.22", "datasets>=3.0", "kaggle"], check=False)

# ── 환경 ────────────────────────────────────────────────────────
import jax
devs = jax.devices()
IS_TPU = devs[0].platform == "tpu"
print(f"디바이스   {{len(devs)}}개 · {{devs[0].device_kind}}"
      f"{{'' if IS_TPU else '  ← TPU 아님'}}")
if not IS_TPU:
    print("           토크나이저는 이대로 만들 수 있다 (오히려 TPU 쿼터를 아낀다).")
    print("           학습은 건너뛴다 — 끝나면 Settings에서 TPU VM v3-8 로 바꾸고 다시 실행.")
try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
    print("인터넷     OK")
except OSError:
    raise SystemExit("인터넷이 꺼져 있다 → Settings → Internet → On")

# ── Kaggle 저장소 (세션 간 인계용) ──────────────────────────────
# 티어마다 별도 데이터셋을 쓴다. 하나로 합치면 티어를 바꿀 때
# 남의 체크포인트를 집어와서 구조 불일치로 죽는다.
STORE = False
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = s.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = s.get_secret("KAGGLE_KEY")
    os.environ["CATI_CKPT_DATASET"] = f"cati-ckpt-{{NAME}}"
    STORE = True
    print(f"저장소     {{os.environ['KAGGLE_USERNAME']}}/cati-ckpt-{{NAME}}")
except Exception:
    print("저장소     없음 (로컬 저장만) — 350M 전에 Secrets를 넣을 것")

# ── 토크나이저: 없으면 만들고 있으면 재사용 ─────────────────────
# 전 티어가 같은 토크나이저를 공유해야 사다리 실험을 비교할 수 있다.
TOK = Path("artifacts/tokenizer/tokenizer.json")
TOK.parent.mkdir(parents=True, exist_ok=True)


def _find():
    if TOK.exists():
        return TOK, "저장소"
    hits = list(Path("/kaggle/input").rglob("tokenizer.json"))
    if hits:
        return hits[0], "입력 마운트"
    if STORE:
        from cati.store import KaggleDatasetStore
        got = KaggleDatasetStore(os.environ["KAGGLE_USERNAME"],
                                 "cati-tokenizer").fetch_latest(Path("/kaggle/working/_tok"))
        if got and (Path(got) / "tokenizer.json").exists():
            return Path(got) / "tokenizer.json", "Kaggle Dataset"
    return None, None


found, how = _find()
if found is None:
    print("토크나이저 없음 → 새로 학습 (20~40분, 처음 한 번만)\\n")
    subprocess.run([sys.executable, "scripts/train_tokenizer.py",
                    "train", "--docs", str(TOKENIZER_DOCS)], check=True)
    assert TOK.exists(), "학습이 끝났는데 파일이 없다 — 위 출력 확인"
    if STORE:
        from cati.store import KaggleDatasetStore
        stage = Path("/kaggle/working/_tok_pub")
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy(TOK, stage / "tokenizer.json")
        KaggleDatasetStore(os.environ["KAGGLE_USERNAME"],
                           "cati-tokenizer").publish(stage, "cati tokenizer")
else:
    if found != TOK:
        shutil.copy(found, TOK)
    print(f"토크나이저 재사용 ({{how}})")

from tokenizers import Tokenizer
_t = Tokenizer.from_file(str(TOK))
_p = "고양이는 창가에 앉아 오래 밖을 바라보았다."
_r = len(_p) / len(_t.encode(_p).ids)
print(f"           vocab {{_t.get_vocab_size():,}} · 한국어 {{_r:.2f}} 글자/토큰 "
      f"({{'통과' if _r >= 2.0 else '미달 — 알려주세요'}})")
print(f"\\n준비 완료 → {{NAME.upper()}} 학습으로 넘어갑니다")
'''


TRAIN = '''
# ══ 학습 ══ 오래 걸린다. 창 닫아도 백그라운드에서 계속 돈다.
#
#  loss  10.8 에서 시작해 내려가야 한다
#  MFU   35% 근처면 계획대로. 25% 미만이면 알려주세요
import subprocess, sys

if not IS_TPU:
    # T4/P100은 bf16 하드웨어가 없고, 350M을 돌리면 596시간이 걸린다.
    # 세션 하나를 헛되게 태우지 않도록 여기서 멈춘다.
    print("가속기가 TPU가 아니라 학습을 건너뜁니다.\\n")
    print("토크나이저는 위에서 만들어졌습니다. 이제 이것만 하면 됩니다:")
    print("  1. 우측 Settings → Accelerator → 'TPU VM v3-8'")
    print("  2. Save Version → Save & Run All")
    print("\\n(TPU 쿼터를 아꼈습니다 — 토크나이저를 GPU/CPU에서 만들었으니까요)")
else:
    cmd = [sys.executable, "scripts/train.py", "--tier", TIER,
           "--session-hours", str(SESSION_HOURS)]
    if not STORE:
        cmd.append("--no-store")
    subprocess.run(cmd, check=False)
'''


RESULT = '''
# ══ 결과 ══ 다음에 뭘 할지 알려준다.
import json, shutil
from pathlib import Path

steps = sorted(Path("artifacts/ckpt").glob("*/step_*"))
if not IS_TPU:
    print("┌─────────────────────────────────────────────┐")
    print("│  토크나이저 완료                             │")
    print("│  → Settings에서 TPU VM v3-8 로 바꾸고         │")
    print("│    Save & Run All 을 다시 누르세요            │")
    print("└─────────────────────────────────────────────┘")
elif not steps:
    print("체크포인트가 없다 — 위 학습 셀 출력을 확인할 것")
else:
    latest = steps[-1]
    m = json.loads((latest / "meta.json").read_text())
    pct = m["tokens"] / m["target_tokens"]
    print(f"{latest.parent.name}   {m['step']:,}스텝   "
          f"{m['tokens']/1e9:.2f}B / {m['target_tokens']/1e9:.0f}B 토큰   {pct:.0%}")
    print(f"누적 TPU {m.get('device_hours_used', 0):.1f}시간 · 세션 #{m.get('session_index', 1)}")

    if not STORE:
        dst = Path("/kaggle/working/ckpt") / latest.parent.name / latest.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copytree(latest, dst)

    print()
    if pct < 1.0:
        print("┌─────────────────────────────────────────────┐")
        print("│  아직 진행 중                                │")
        print("│  → Save & Run All 을 다시 누르세요            │")
        print("└─────────────────────────────────────────────┘")
    elif STEP < 3:
        print("┌─────────────────────────────────────────────┐")
        print(f"│  {latest.parent.name} 완료                            │")
        print(f"│  → 맨 위 STEP 을 {STEP + 1} 로 바꾸고 다시 실행       │")
        print("└─────────────────────────────────────────────┘")
    else:
        print("┌─────────────────────────────────────────────┐")
        print("│  사전학습 전부 완료                          │")
        print("│  → 다음은 도서 어닐링 (문체 학습)            │")
        print("└─────────────────────────────────────────────┘")
'''


def build() -> dict:
    return notebook([
        md('''
# Cati 학습

### 처음 한 번만
1. 우측 **Settings** → Accelerator **TPU VM v3-8**, Internet **On**
2. 우측 상단 **Save Version → Save & Run All** → 창 닫기

### 그 다음부터
**Save & Run All 만 다시 누르세요.** 알아서 이어집니다.

맨 아래 셀이 다음에 뭘 할지 알려줍니다.
'''),
        code('''
STEP = 1               # 1=50M(1시간)   2=100M(4.5시간)   3=350M(5주)

SESSION_HOURS = 9.0    # Kaggle TPU 세션 한계
TOKENIZER_DOCS = 400_000
'''),
        code(PREPARE),
        code(TRAIN),
        code(RESULT),
    ])


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    OUT.mkdir(exist_ok=True)
    for old in ("01_tokenizer.ipynb", "02_train.ipynb"):
        p = OUT / old
        if p.exists():
            p.unlink()
            print(f"삭제: notebooks/{old}")

    nb = build()
    path = OUT / "cati_train.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"{path.relative_to(ROOT)}  (셀 {len(nb['cells'])}개 / 코드 {n_code}개)")


if __name__ == "__main__":
    main()
