#!/usr/bin/env python3
"""Kaggle 노트북 생성기.

.ipynb 를 손으로 쓰면 JSON 이스케이프 때문에 금방 깨진다. 여기서 만든다.

    python3 scripts/make_notebooks.py

만들어지는 것:
    notebooks/cati_train.ipynb   토크나이저 + 학습을 한 노트북에서 처리

설계 목표는 **사람 손을 줄이는 것**이다. 5주 동안 11번 세션을 돌려야 하므로,
매 세션이 "Save & Run All 누르기" 하나로 끝나야 한다.
그래서 노트북을 하나로 합치고, 없는 것(토크나이저·체크포인트)은 알아서 만들거나 찾는다.
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


def build() -> dict:
    return notebook([
        md('''
# Cati 학습

## 처음 한 번만
1. 우측 **Settings** → Accelerator **TPU VM v3-8** · Internet **On**
2. 우측 상단 **Save Version → Save & Run All** → Save → 창 닫기

## 그 다음부터
**Save & Run All 만 다시 누르면 됩니다.** 체크포인트에서 자동으로 이어집니다.

토크나이저가 없으면 알아서 만들고, 있으면 건너뜁니다. 붙일 것도 고를 것도 없습니다.

## 순서
아래 `TIER` 를 이 순서로 바꿔가며 돌립니다.

| TIER | 시간 | 세션 |
|---|---|---|
| `tier0_50m` | 1.1h | 1번 |
| `tier1_100m` | 4.4h | 1번 |
| `tier2_350m` | 97.4h | **11번** (5주) |

50M → 100M 을 먼저 끝내세요. 97시간을 태우기 전에 파이프라인을 검증하고,
스케일링 법칙으로 350M 설정이 맞는지 확인하는 단계입니다.

## 350M 전에 한 번 해둘 것
세션 11번을 이어가려면 체크포인트가 세션 밖에서 살아남아야 합니다.

1. `kaggle.com/settings` → API → **Create New Token** → `kaggle.json` 받기
2. 이 노트북 → **Add-ons → Secrets** → 2개 추가
   - `KAGGLE_USERNAME` (파일 안의 username)
   - `KAGGLE_KEY` (파일 안의 key)

50M·100M은 한 세션에 끝나므로 없어도 됩니다.
'''),

        md("## 설정 — 여기만 고칩니다"),
        code('''
TIER = "configs/tier0_50m.json"     # tier0_50m → tier1_100m → tier2_350m
SESSION_HOURS = 9.0                 # Kaggle TPU 세션 한계
TOKENIZER_DOCS = 400_000            # 토크나이저 학습 문서 수 (처음 한 번만 씀)

CKPT_DATASET = "cati-checkpoints"   # 체크포인트를 넣어둘 Kaggle Dataset
TOK_DATASET = "cati-tokenizer"      # 토크나이저를 넣어둘 Kaggle Dataset
'''),

        md("## 1. 코드 가져오기"),
        code(f'''
GITHUB_URL = "{GITHUB_URL}"
DATASET_DIR = "/kaggle/input/cati-code"   # GitHub 대신 Dataset으로 올린 경우

import os, shutil, subprocess, sys
from pathlib import Path

CATI = Path("/kaggle/working/Cati")
if GITHUB_URL:
    if CATI.exists():
        subprocess.run(["git", "-C", str(CATI), "pull", "-q"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "1", GITHUB_URL, str(CATI)], check=True)
elif Path(DATASET_DIR).exists():
    if not CATI.exists():
        shutil.copytree(DATASET_DIR, CATI)    # Dataset은 읽기 전용이라 복사한다
else:
    raise SystemExit("코드를 못 찾았다. GITHUB_URL을 채우거나 "
                     "Cati 폴더를 Dataset 'cati-code'로 올릴 것")

os.chdir(CATI)
sys.path.insert(0, str(CATI))
rev = subprocess.run(["git", "-C", str(CATI), "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout.strip()
print("코드:", CATI, "·", rev or "(git 정보 없음)")
'''),

        md("## 2. 패키지"),
        code('''
!pip install -q "tokenizers>=0.22" "datasets>=3.0" kaggle 2>&1 | tail -2
import datasets, tokenizers
print("tokenizers", tokenizers.__version__, "· datasets", datasets.__version__)
'''),

        md('''## 3. 환경 확인

디바이스가 **8개**로 나와야 합니다. 1개(cpu)면 Accelerator 설정이 안 된 겁니다.'''),
        code('''
import socket
import jax

devs = jax.devices()
print(f"디바이스 {len(devs)}개 · {devs[0].device_kind} · {devs[0].platform}")
if devs[0].platform != "tpu":
    print("⚠️ TPU가 아니다. Settings → Accelerator → TPU VM v3-8 으로 바꾸고 세션 재시작.")
    print("   (토크나이저만 만들 목적이면 CPU로 돌려서 TPU 쿼터를 아낄 수 있다)")

try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
    print("인터넷 OK")
except OSError as e:
    raise SystemExit(f"인터넷이 꺼져 있다: {e}\\n"
                     "Settings → Internet → On 으로 켜고 다시 실행")
'''),

        md('''## 4. Kaggle 저장소 연결

Secrets를 넣었으면 체크포인트와 토크나이저가 세션 밖에서 살아남습니다.
없으면 로컬 저장만 하고, 세션이 끝나면 Output을 다음 세션 Input으로 붙여야 합니다.'''),
        code('''
import os

STORE = False
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = s.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = s.get_secret("KAGGLE_KEY")
    os.environ["CATI_CKPT_DATASET"] = CKPT_DATASET
    STORE = True
    print(f"저장소 연결: {os.environ['KAGGLE_USERNAME']}/{CKPT_DATASET}")
except Exception as e:
    print(f"Secrets 없음 ({type(e).__name__}) — 로컬 저장만 한다")
    print("50M·100M은 한 세션에 끝나므로 문제없다. 350M 전에는 넣을 것.")
'''),

        md('''## 5. 토크나이저 — 없으면 만들고, 있으면 건너뜀

**전 티어가 같은 토크나이저를 공유해야 합니다.** 중간에 바뀌면 사다리 실험을
비교할 수 없게 되므로, 한 번 만든 뒤로는 계속 재사용합니다.

처음 실행이면 20~40분 걸립니다. 두 번째부터는 몇 초입니다.'''),
        code('''
from pathlib import Path
import shutil, subprocess, sys

TOK = Path("artifacts/tokenizer/tokenizer.json")
TOK.parent.mkdir(parents=True, exist_ok=True)


def locate():
    """이미 만들어둔 토크나이저를 찾는다."""
    if TOK.exists():
        return TOK, "저장소에 커밋됨"
    hits = [p for p in Path("/kaggle/input").rglob("tokenizer.json")]
    if hits:
        return hits[0], f"입력 마운트 ({hits[0].parent.name})"
    if STORE:
        from cati.store import KaggleDatasetStore
        st = KaggleDatasetStore(os.environ["KAGGLE_USERNAME"], TOK_DATASET)
        got = st.fetch_latest(Path("/kaggle/working/_tok"))
        if got and (Path(got) / "tokenizer.json").exists():
            return Path(got) / "tokenizer.json", f"Kaggle Dataset ({TOK_DATASET})"
    return None, None


found, how = locate()

if found is None:
    print("토크나이저가 없다 → 새로 학습한다 (20~40분, 처음 한 번만)\\n")
    subprocess.run([sys.executable, "scripts/train_tokenizer.py", "train",
                    "--docs", str(TOKENIZER_DOCS)], check=True)
    assert TOK.exists(), "학습이 끝났는데 파일이 없다 — 위 출력을 확인할 것"
    if STORE:
        from cati.store import KaggleDatasetStore
        stage = Path("/kaggle/working/_tok_pub")
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy(TOK, stage / "tokenizer.json")
        ok = KaggleDatasetStore(os.environ["KAGGLE_USERNAME"], TOK_DATASET).publish(
            stage, "cati tokenizer")
        print(f"토크나이저 발행 {'완료' if ok else '실패 — 다음 세션에 다시 만든다'}")
else:
    if found != TOK:
        shutil.copy(found, TOK)
    print(f"토크나이저 재사용: {how}")

from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(TOK))
probe = "고양이는 창가에 앉아 오래 밖을 바라보았다."
ratio = len(probe) / len(tok.encode(probe).ids)
print(f"\\nvocab {tok.get_vocab_size():,} · 한국어 압축률 {ratio:.2f} 글자/토큰")
print(f"기준선: SmolLM2(같은 vocab) 0.47 · Qwen3(vocab 3배) 1.39 · 목표 2.0 이상")
print("판정:", "통과 ✅" if ratio >= 2.0 else "미달 ⚠️ — 한국어 믹스 비중을 올릴 것")
'''),

        md("## 6. 예산 확인"),
        code('''
!python scripts/budget.py
'''),

        md('''## 7. 학습

여기부터 오래 걸립니다. `Save & Run All` 로 돌렸으면 창을 닫아도 됩니다.

로그에서 볼 것:
- **loss** — `10.8`(=ln 49152)에서 시작해 내려가야 합니다
- **MFU** — 계산 활용률. 35% 근처면 계획대로. **25% 미만이면 토큰 수를 낮춰야 합니다**
- **|g|** — 기울기 크기. 갑자기 튀면 불안정 신호입니다'''),
        code('''
import subprocess, sys

cmd = [sys.executable, "scripts/train.py", "--tier", TIER,
       "--session-hours", str(SESSION_HOURS)]
if not STORE:
    cmd.append("--no-store")
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=False)
'''),

        md("## 8. 결과 보존"),
        code('''
import json, shutil
from pathlib import Path

steps = sorted(Path("artifacts/ckpt").glob("*/step_*"))
if not steps:
    print("체크포인트가 없다 — 위 학습 셀 출력을 확인할 것")
else:
    latest = steps[-1]
    meta = json.loads((latest / "meta.json").read_text())
    print(f"티어      {latest.parent.name}")
    print(f"스텝      {meta['step']:,}")
    print(f"토큰      {meta['tokens']/1e9:.3f}B / {meta['target_tokens']/1e9:.1f}B "
          f"({meta['tokens']/meta['target_tokens']:.1%})")
    print(f"누적 TPU  {meta.get('device_hours_used', 0):.2f}시간")
    print(f"세션      #{meta.get('session_index', 1)}")

    if not STORE:
        dst = Path("/kaggle/working/ckpt") / latest.parent.name / latest.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copytree(latest, dst)
        print(f"\\n로컬 보존: {dst}")
        print("다음 세션에서 이 노트북 Output을 Input으로 추가할 것")

    done = meta["tokens"] >= meta["target_tokens"]
    print("\\n" + ("이 티어 완료 ✅ — 위 TIER를 다음 것으로 바꿔서 실행"
                   if done else
                   "아직 진행 중 — Save & Run All 을 다시 누르면 이어집니다"))
'''),
    ])


def main():
    OUT.mkdir(exist_ok=True)
    # 예전 2개 노트북은 하나로 합쳐졌다
    for old in ("01_tokenizer.ipynb", "02_train.ipynb"):
        p = OUT / old
        if p.exists():
            p.unlink()
            print(f"삭제: notebooks/{old}")

    nb = build()
    path = OUT / "cati_train.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"{path.relative_to(ROOT)}  ({len(nb['cells'])}셀 / 코드 {n_code}개)")


if __name__ == "__main__":
    main()
