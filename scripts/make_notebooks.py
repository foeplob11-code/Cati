#!/usr/bin/env python3
"""Kaggle 노트북 생성기.

.ipynb 를 손으로 쓰면 JSON 이스케이프 때문에 금방 깨진다. 여기서 만든다.

    python3 scripts/make_notebooks.py

만들어지는 것:
    notebooks/01_tokenizer.ipynb   CPU 세션 — 토크나이저 학습 (TPU 쿼터 안 씀)
    notebooks/02_train.ipynb       TPU 세션 — 사전학습
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"


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


# ---------------------------------------------------------------------------
# 공통 셀
# ---------------------------------------------------------------------------
SETUP = '''
# ── 코드 가져오기 ────────────────────────────────────────────────
# 두 가지 방법 중 하나. GITHUB_URL을 채우면 그쪽을 쓴다.
GITHUB_URL = ""          # 예: "https://github.com/사용자명/Cati.git"
DATASET_DIR = "/kaggle/input/cati-code"   # 코드를 Dataset으로 올린 경우

import os, subprocess, sys, shutil
from pathlib import Path

if GITHUB_URL:
    if not Path("/kaggle/working/Cati").exists():
        subprocess.run(["git", "clone", "--depth", "1", GITHUB_URL,
                        "/kaggle/working/Cati"], check=True)
    CATI = Path("/kaggle/working/Cati")
elif Path(DATASET_DIR).exists():
    # Dataset은 읽기 전용이라 작업 디렉터리로 복사한다
    CATI = Path("/kaggle/working/Cati")
    if not CATI.exists():
        shutil.copytree(DATASET_DIR, CATI)
else:
    raise SystemExit(
        "코드를 못 찾았다. 둘 중 하나를 하세요:\\n"
        "  (a) GitHub에 올리고 위 GITHUB_URL 채우기\\n"
        "  (b) Cati 폴더를 Kaggle Dataset 'cati-code'로 업로드하고 이 노트북에 추가하기")

os.chdir(CATI)
sys.path.insert(0, str(CATI))
print("코드:", CATI)
print("파일:", sorted(p.name for p in CATI.iterdir() if not p.name.startswith(".")))
'''

INSTALL = '''
# ── 패키지 ───────────────────────────────────────────────────────
!pip install -q "tokenizers>=0.22" "datasets>=3.0" 2>&1 | tail -2
import tokenizers, datasets
print("tokenizers", tokenizers.__version__, "· datasets", datasets.__version__)
'''


# ---------------------------------------------------------------------------
# 노트북 1 — 토크나이저 (CPU)
# ---------------------------------------------------------------------------
def tokenizer_notebook() -> dict:
    return notebook([
        md('''
# Cati 01 — 토크나이저 학습

**가속기: 없음 (CPU)** ← 오른쪽 Settings에서 Accelerator를 **None**으로 두세요.
CPU 세션은 TPU 쿼터를 쓰지 않습니다. 20시간을 아끼는 겁니다.

## 이 노트북이 하는 일
1. FineWeb2 한국어 스트리밍이 실제로 되는지 확인
2. 끊고 이어받기가 정확한지 확인
3. 토크나이저 학습 (vocab 49,152)
4. 한국어 압축률 측정 — 기성 토크나이저와 비교

## 왜 중요한가
계산량은 **토큰** 단위로 붙습니다. 한국어 압축률이 0.47(SmolLM2급)에서 2.5로
올라가면 같은 GPU 시간에 **한국어 원문이 5.3배** 들어갑니다.

## ⚠️ 한 번만 하고 다시는 바꾸지 않습니다
전 티어(50M/100M/350M)가 같은 토크나이저를 공유해야 실험 결과를 비교할 수 있습니다.

## 끝나고 할 일
`Save Version` → 다음 노트북에서 이 노트북의 **Output**을 Input으로 추가합니다.
'''),
        code(SETUP),
        code(INSTALL),
        md("## 1. 인터넷 확인\n\nSettings → Internet 을 **On** 으로 해야 합니다."),
        code('''
import socket
try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
    print("인터넷 OK")
except OSError as e:
    raise SystemExit(f"인터넷이 꺼져 있다: {e}\\n"
                     "오른쪽 Settings → Internet → On 으로 켜고 다시 실행")
'''),
        md('''## 2. 스트리밍 + 이어받기 검증

여기가 유일하게 검증 안 된 부분이었습니다. `datasets` 의 네이티브 재개가
동작하지 않으면 세션마다 앞부분을 다시 읽어야 해서 5주 런에서 비용이 커집니다.'''),
        code('''
from cati.stream import HFSource, ResumableStream

src = HFSource("ko", "HuggingFaceFW/fineweb-2", "kor_Hang", "text")
st = ResumableStream([src], [1.0], seed=0)
it = iter(st)

print("연결 중...")
head = [next(it)[1] for _ in range(4)]
print(f"네이티브 재개 지원: {src._native}")
for i, t in enumerate(head):
    print(f"  {i}: {t[:70].replace(chr(10), ' ')}...")

state = st.state_dict()
after = [next(it)[1][:60] for _ in range(3)]

st2 = ResumableStream([HFSource("ko", "HuggingFaceFW/fineweb-2", "kor_Hang", "text")],
                      [1.0], seed=0)
st2.load_state_dict(state)
resumed = [next(iter(st2))[1][:60] for _ in range(1)]

ok = after[0] == resumed[0]
print(f"\\n이어받기: {'정확 ✅' if ok else '어긋남 ⚠️ — 건너뛰기 경로로 동작한다'}")
if not ok:
    print("  → 치명적이지는 않지만, 사전 토큰화 방식 전환을 검토할 것")
'''),
        md('''## 3. 토크나이저 학습

문서 40만 개로 학습합니다. vocab 49,152에는 충분합니다.
RAM이 남으면 `--docs` 를 올려도 되지만 효과는 크지 않습니다.

10~30분 걸립니다.'''),
        code('''
!python scripts/train_tokenizer.py train --docs 400000
'''),
        md("## 4. 압축률 측정 — 목표를 넘겼는지"),
        code('''
!pip install -q transformers 2>&1 | tail -1
!python scripts/train_tokenizer.py measure --baseline Qwen/Qwen3-1.7B
'''),
        md('''## 5. 결과 저장

`/kaggle/working` 에 두면 `Save Version` 할 때 노트북 Output으로 보존됩니다.
다음 노트북에서 이걸 Input으로 붙입니다.'''),
        code('''
import shutil
from pathlib import Path

src = Path("artifacts/tokenizer/tokenizer.json")
dst = Path("/kaggle/working/tokenizer.json")
assert src.exists(), "토크나이저가 만들어지지 않았다 — 위 셀 출력을 확인할 것"
shutil.copy(src, dst)
print(f"저장: {dst}  ({dst.stat().st_size/1e6:.2f} MB)")
print("\\n다음: 우측 상단 Save Version → Save & Run All")
print("     그 다음 02_train 노트북에서 이 노트북의 Output을 Input으로 추가")
'''),
    ])


# ---------------------------------------------------------------------------
# 노트북 2 — 학습 (TPU)
# ---------------------------------------------------------------------------
def train_notebook() -> dict:
    return notebook([
        md('''
# Cati 02 — 사전학습

**가속기: TPU VM v3-8** ← 오른쪽 Settings에서 반드시 설정하세요.

## 실행 방법
1. Settings → Accelerator → **TPU VM v3-8**
2. Settings → Internet → **On**
3. Input에 **01_tokenizer 노트북의 Output** 추가
4. 아래 `TIER` 를 고르고
5. 우측 상단 **Save Version → Save & Run All (Commit)**

4번이 중요합니다. `Save & Run All` 로 돌리면 **브라우저를 닫아도 백그라운드에서
9시간을 다 씁니다.** 화면을 보고 있을 필요가 없습니다.

## 순서
| 티어 | 시간 | 세션 |
|---|---|---|
| `tier0_50m` | 1.1h | 1번 |
| `tier1_100m` | 4.4h | 1번 |
| `tier2_350m` | 97.4h | **11번** (5주) |

50M → 100M 을 먼저 끝내세요. 97시간을 태우기 전에 파이프라인을 검증하고,
스케일링 법칙으로 350M 설정이 맞는지 확인하는 단계입니다.

## 세션이 끝나면
그냥 **다시 Save & Run All** 하면 됩니다. 체크포인트에서 자동으로 이어집니다.
'''),
        code('''
# ── 여기만 고치면 됩니다 ─────────────────────────────────────────
TIER = "configs/tier0_50m.json"     # tier0_50m → tier1_100m → tier2_350m
SESSION_HOURS = 9.0                 # Kaggle TPU 세션 한계
CKPT_DATASET = "cati-checkpoints"   # 체크포인트를 저장할 Kaggle Dataset 이름
'''),
        code(SETUP),
        code(INSTALL),
        md('''## 1. TPU 확인

디바이스가 8개로 나와야 합니다. 1개(CPU)로 나오면 Accelerator 설정이 안 된 겁니다.'''),
        code('''
import jax
devs = jax.devices()
print(f"디바이스 {len(devs)}개 · {devs[0].device_kind} · {devs[0].platform}")
if devs[0].platform != "tpu":
    raise SystemExit("TPU가 아니다. Settings → Accelerator → TPU VM v3-8 으로 바꾸고 "
                     "세션을 재시작할 것")
print("bf16 지원:", jax.numpy.bfloat16(1.0).dtype)
'''),
        md('''## 2. 토크나이저 연결

01 노트북의 Output을 Input으로 추가했다면 `/kaggle/input/.../tokenizer.json` 에 있습니다.'''),
        code('''
import shutil
from pathlib import Path

found = list(Path("/kaggle/input").rglob("tokenizer.json"))
if not found:
    raise SystemExit(
        "토크나이저를 못 찾았다.\\n"
        "  → 우측 Input 패널 → Add Input → Notebook Output → 01_tokenizer 선택\\n"
        "  (아직 01을 안 돌렸으면 그것부터 실행)")

dst = Path("artifacts/tokenizer/tokenizer.json")
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(found[0], dst)

from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(dst))
probe = "고양이는 창가에 앉아 오래 밖을 바라보았다."
n = len(tok.encode(probe).ids)
print(f"토크나이저 OK · vocab {tok.get_vocab_size():,}")
print(f"한국어 압축률 {len(probe)/n:.2f} 글자/토큰  (기준: SmolLM2 0.47 / Qwen3 1.39)")
'''),
        md('''## 3. 체크포인트 저장소 연결

**11번의 세션을 이어가려면 이게 필요합니다.** 세션이 끝나면 `/kaggle/working` 은
사라지므로, 체크포인트를 Kaggle Dataset에 올려둬야 다음 세션이 집어옵니다.

### 준비 (한 번만)
1. `kaggle.com/settings` → API → **Create New Token** → `kaggle.json` 다운로드
2. 이 노트북 → Add-ons → **Secrets** → 두 개 추가
   - `KAGGLE_USERNAME` : kaggle.json 의 username
   - `KAGGLE_KEY` : kaggle.json 의 key

건너뛰어도 학습은 됩니다. 다만 세션마다 이전 Output을 Input으로 직접 붙여야 합니다.'''),
        code('''
import os
try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = s.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = s.get_secret("KAGGLE_KEY")
    os.environ["CATI_CKPT_DATASET"] = CKPT_DATASET
    !pip install -q kaggle 2>&1 | tail -1
    print(f"체크포인트 저장소: {os.environ['KAGGLE_USERNAME']}/{CKPT_DATASET}")
    STORE = True
except Exception as e:
    print(f"Secrets 없음 ({type(e).__name__}) — 로컬 저장만 한다")
    print("세션이 끝나면 Output을 다음 세션의 Input으로 직접 추가할 것")
    STORE = False
'''),
        md('''## 4. 예산 확인

시작 전에 이 티어가 쿼터에 들어오는지 봅니다.'''),
        code('''
!python scripts/budget.py
'''),
        md('''## 5. 학습

여기부터 오래 걸립니다. `Save & Run All` 로 돌렸으면 창을 닫아도 됩니다.

로그에서 볼 것:
- **loss** — 내려가야 합니다. ln(49152)=10.8 에서 시작합니다
- **MFU** — 계산 활용률. 35% 근처면 계획대로입니다. 25% 미만이면 토큰 수를 낮춰야 합니다
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
        md('''## 6. 결과 보존

Secrets를 안 쓰는 경우 이 셀이 체크포인트를 Output으로 남깁니다.'''),
        code('''
import shutil
from pathlib import Path

src = sorted(Path("artifacts/ckpt").glob("*/step_*"))
if src:
    latest = src[-1]
    dst = Path("/kaggle/working/ckpt") / latest.parent.name / latest.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copytree(latest, dst)
    size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
    print(f"보존: {dst}  ({size/1e9:.2f} GB)")
else:
    print("체크포인트가 없다 — 위 학습 셀 출력을 확인할 것")

print("\\n다음 세션: 이 노트북을 다시 Save & Run All 하면 이어서 학습합니다.")
'''),
    ])


def main():
    OUT.mkdir(exist_ok=True)
    for name, nb in [("01_tokenizer", tokenizer_notebook()),
                     ("02_train", train_notebook())]:
        path = OUT / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"{path.relative_to(ROOT)}  ({len(nb['cells'])}셀 / 코드 {n_code}개)")


if __name__ == "__main__":
    main()
