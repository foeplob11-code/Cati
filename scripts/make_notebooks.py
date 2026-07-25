#!/usr/bin/env python3
"""노트북 생성기.

.ipynb 를 손으로 쓰면 JSON 이스케이프 때문에 금방 깨진다. 여기서 만든다.

    python3 scripts/make_notebooks.py

만들어지는 것
    notebooks/cati_colab.ipynb      학습 — Colab TPU v5e-1 (메인)
    notebooks/cati_tokenizer.ipynb  토크나이저 — Kaggle (선택, Colab 시간 절약용)

역할이 갈리는 이유: Kaggle에는 TPU 옵션이 없다 (T4/P100뿐). T4로 200M을 돌리면
111시간이 걸리고 bf16도 없어 fp16 loss scaling을 따로 써야 한다. 반면 Colab의
v5e-1은 bf16 네이티브라 지금 코드가 그대로 돈다. 그래서 학습은 Colab에서 한다.
Kaggle은 TPU가 필요 없는 토크나이저 학습에만 쓴다.
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


# ===========================================================================
# Colab — 학습 (메인)
# ===========================================================================
COLAB_PREPARE = f'''
# ══ 준비 ══ Drive 연결 · 코드 · 토크나이저를 알아서 챙긴다.
import os, shutil, subprocess, sys
from pathlib import Path

TIERS = ["configs/tier0_50m.json", "configs/tier1_100m.json", "configs/tier2_200m.json"]
TIER = TIERS[STEP - 1]

# ── TPU 확인 ────────────────────────────────────────────────────
import jax
devs = jax.devices()
IS_TPU = devs[0].platform == "tpu"
print(f"디바이스   {{len(devs)}}개 · {{devs[0].device_kind}}")
if not IS_TPU:
    print("           ⚠️ TPU가 아니다 → 런타임 → 런타임 유형 변경 → TPU v5e-1")
    print("           토크나이저는 이대로 만들 수 있다. 학습은 건너뛴다.")

# ── Google Drive ────────────────────────────────────────────────
# 무료 Colab은 예고 없이 끊기고 /content 는 사라진다.
# 체크포인트가 세션 밖에서 살아남는 유일한 길이다.
from google.colab import drive
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/cati")
DRIVE.mkdir(parents=True, exist_ok=True)
print(f"Drive      {{DRIVE}}")

# ── 코드 ────────────────────────────────────────────────────────
CATI = Path("/content/Cati")
if CATI.exists():
    subprocess.run(["git", "-C", str(CATI), "pull", "-q"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1",
                    "{GITHUB_URL}", str(CATI)], check=True)
os.chdir(CATI)
sys.path.insert(0, str(CATI))
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tokenizers>=0.22", "datasets>=3.0", "flax", "optax",
                "orbax-checkpoint"], check=False)

# ── 토크나이저: Drive에 있으면 재사용, 없으면 만들어 Drive에 둔다 ──
# 전 티어가 같은 토크나이저를 공유해야 사다리 실험을 비교할 수 있다.
TOK = Path("artifacts/tokenizer/tokenizer.json")
TOK.parent.mkdir(parents=True, exist_ok=True)
DRIVE_TOK = DRIVE / "tokenizer.json"

if DRIVE_TOK.exists():
    shutil.copy(DRIVE_TOK, TOK)
    print("토크나이저 Drive에서 재사용")
elif TOK.exists():
    shutil.copy(TOK, DRIVE_TOK)
    print("토크나이저 저장소 커밋본 사용")
else:
    print("토크나이저 없음 → 새로 학습 (20~40분, 처음 한 번만)\\n")
    subprocess.run([sys.executable, "scripts/train_tokenizer.py",
                    "train", "--docs", str(TOKENIZER_DOCS)], check=True)
    assert TOK.exists(), "학습이 끝났는데 파일이 없다 — 위 출력 확인"
    shutil.copy(TOK, DRIVE_TOK)
    print(f"\\n토크나이저를 Drive에 저장: {{DRIVE_TOK}}")

from tokenizers import Tokenizer
_t = Tokenizer.from_file(str(TOK))
_p = "고양이는 창가에 앉아 오래 밖을 바라보았다."
_r = len(_p) / len(_t.encode(_p).ids)
print(f"           vocab {{_t.get_vocab_size():,}} · 한국어 {{_r:.2f}} 글자/토큰 "
      f"({{'통과' if _r >= 2.0 else '미달 — 알려주세요'}})")
print("\\n준비 완료")
'''

COLAB_TRAIN = '''
# ══ 학습 ══
#  ⚠️ 이 탭을 닫지 마세요. 무료 Colab은 백그라운드 실행이 안 됩니다.
#     노트북 절전도 꺼두세요. 끊기면 마지막 체크포인트부터 이어집니다.
#
#  loss  10.8(=ln 49152) 에서 시작해 내려가야 한다
#  MFU   35% 근처면 계획대로. 25% 미만이면 알려주세요
import os, subprocess, sys

if not IS_TPU:
    print("TPU가 아니라 학습을 건너뜁니다.")
    print("런타임 → 런타임 유형 변경 → TPU v5e-1 로 바꾸고 다시 실행하세요.")
else:
    # 체크포인트는 /content 에 쓰고(빠름) Drive로 발행한다(살아남음).
    os.environ["CATI_CKPT_STORE"] = str(DRIVE / "ckpt")
    subprocess.run([sys.executable, "scripts/train.py", "--tier", TIER,
                    "--session-hours", str(SESSION_HOURS),
                    "--ckpt", "/content/ckpt",
                    "--quota-hours", "160"], check=False)
'''

COLAB_RESULT = '''
# ══ 결과 ══
import json
from pathlib import Path

steps = (sorted(Path("/content/ckpt").glob("step_*")) or
         sorted((DRIVE / "ckpt").glob("step_*")))
if not IS_TPU:
    print("토크나이저 완료 → 런타임을 TPU v5e-1 로 바꾸고 다시 실행하세요")
elif not steps:
    print("체크포인트가 없다 — 위 학습 셀 출력을 확인할 것")
else:
    m = json.loads((steps[-1] / "meta.json").read_text())
    pct = m["tokens"] / m["target_tokens"]
    print(f"{m['step']:,}스텝   {m['tokens']/1e9:.2f}B / "
          f"{m['target_tokens']/1e9:.0f}B 토큰   {pct:.1%}")
    print(f"누적 {m.get('device_hours_used', 0):.1f}시간 · 세션 #{m.get('session_index', 1)}")
    print(f"Drive:  {DRIVE / 'ckpt'}")
    print()
    if pct < 1.0:
        print("아직 진행 중  →  이 노트북을 다시 실행하세요 (이어집니다)")
    elif STEP < 3:
        print(f"이 티어 완료  →  맨 위 STEP 을 {STEP + 1} 로 바꾸고 다시 실행")
    else:
        print("사전학습 완료  →  다음은 도서 어닐링 (문체 학습)")
'''


def build_colab() -> dict:
    return notebook([
        md('''
# Cati 학습 — Colab TPU v5e-1

### 처음 한 번만
1. **런타임 → 런타임 유형 변경 → TPU v5e-1**
2. 위에서부터 셀을 순서대로 실행 (`Shift+Enter`)
3. Drive 연결 권한 허용 — 체크포인트를 여기 저장합니다

### 그 다음부터
**다시 실행하면 이어집니다.** 짧은 세션을 여러 번 돌려도 진행이 누적됩니다.

### ⚠️ 무료 Colab의 제약 두 가지
- **탭을 닫으면 멈춥니다.** 백그라운드 실행은 유료 기능입니다. 절전도 꺼두세요.
- **사용량 제한이 유동적입니다.** 갑자기 끊길 수 있습니다.

끊겨도 200스텝마다 Drive에 저장되므로 잃는 건 몇 분치입니다.

### 순서
| STEP | 모델 | v5e-1 시간 | 과학습 배수 |
|---|---|---|---|
| 1 | 50M | 2.3h | 42× |
| 2 | 100M | 9.5h | 41× |
| 3 | **200M** | **72h** | **75×** |

50M → 100M 을 먼저 끝내세요. 72시간을 태우기 전에 파이프라인을 검증하고
스케일링 법칙으로 200M 설정이 맞는지 확인하는 단계입니다.
'''),
        code('''
STEP = 1               # 1=50M   2=100M   3=200M

SESSION_HOURS = 3.5    # 무료 Colab 세션은 보통 3~4시간. 짧게 잡아 자주 저장한다
TOKENIZER_DOCS = 400_000
'''),
        code(COLAB_PREPARE),
        code(COLAB_TRAIN),
        code(COLAB_RESULT),
    ])


# ===========================================================================
# Kaggle — 토크나이저 전용 (선택)
# ===========================================================================
KAGGLE_TOK = f'''
# ══ 토크나이저 학습 ══
import os, shutil, socket, subprocess, sys
from pathlib import Path

CATI = Path("/kaggle/working/Cati")
if CATI.exists():
    subprocess.run(["git", "-C", str(CATI), "pull", "-q"], check=False)
else:
    subprocess.run(["git", "clone", "--depth", "1",
                    "{GITHUB_URL}", str(CATI)], check=True)
os.chdir(CATI)
sys.path.insert(0, str(CATI))
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tokenizers>=0.22", "datasets>=3.0"], check=False)

try:
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError:
    raise SystemExit("인터넷이 꺼져 있다 → Settings → Internet → On")

TOK = Path("artifacts/tokenizer/tokenizer.json")
TOK.parent.mkdir(parents=True, exist_ok=True)
if not TOK.exists():
    subprocess.run([sys.executable, "scripts/train_tokenizer.py",
                    "train", "--docs", str(TOKENIZER_DOCS)], check=True)

from tokenizers import Tokenizer
_t = Tokenizer.from_file(str(TOK))
_p = "고양이는 창가에 앉아 오래 밖을 바라보았다."
_r = len(_p) / len(_t.encode(_p).ids)
print(f"\\nvocab {{_t.get_vocab_size():,}} · 한국어 {{_r:.2f}} 글자/토큰 "
      f"({{'통과' if _r >= 2.0 else '미달 — 알려주세요'}})")
print("기준선: SmolLM2(같은 vocab) 0.47 · Qwen3(vocab 3배) 1.39 · 목표 2.0 이상")

out = Path("/kaggle/working/tokenizer.json")
shutil.copy(TOK, out)
print(f"\\n저장: {{out}}  ({{out.stat().st_size/1e6:.2f}} MB)")
print("\\n다음: 이 파일을 내려받아 Google Drive의 MyDrive/cati/ 에 넣으세요.")
print("      그러면 Colab 노트북이 바로 씁니다.")
'''


def build_tokenizer_nb() -> dict:
    return notebook([
        md('''
# Cati 토크나이저 — Kaggle

**학습은 Colab에서 합니다** (`cati_colab.ipynb`). Kaggle에는 TPU 옵션이 없어서요.
이 노트북은 토크나이저만 만듭니다 — 귀한 Colab 시간을 30분 아끼려고요.

건너뛰어도 됩니다. Colab 노트북이 토크나이저가 없으면 알아서 만듭니다.

### 하는 법
1. Settings → Accelerator **None** · Internet **On**
2. **Save Version → Save & Run All** → 창 닫기
3. 30분 뒤 Output의 `tokenizer.json` 을 내려받아
   Google Drive의 `MyDrive/cati/` 에 넣기
'''),
        code('TOKENIZER_DOCS = 400_000'),
        code(KAGGLE_TOK),
    ])


def main():
    OUT.mkdir(exist_ok=True)
    for old in ("01_tokenizer.ipynb", "02_train.ipynb", "cati_train.ipynb"):
        p = OUT / old
        if p.exists():
            p.unlink()
            print(f"삭제: notebooks/{old}")

    for name, nb in [("cati_colab", build_colab()),
                     ("cati_tokenizer", build_tokenizer_nb())]:
        path = OUT / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"{path.relative_to(ROOT)}  (셀 {len(nb['cells'])}개 / 코드 {n_code}개)")


if __name__ == "__main__":
    main()
