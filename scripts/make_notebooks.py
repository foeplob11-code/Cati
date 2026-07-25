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
# ══ 준비 ══ 저장소 연결 · 코드 · 토크나이저를 알아서 챙긴다.
import os, shutil, subprocess, sys, time
from pathlib import Path

# 세션 시작 시각. 토크나이저 학습에 40분을 쓰면 학습에 남는 시간도 그만큼 줄어든다.
# 이걸 학습 셀에 넘겨줘야 세션 가드가 실제 남은 시간을 안다.
NB_START = time.monotonic()

# 설정 셀을 건너뛰고 이 셀부터 실행해도 죽지 않게 기본값을 채운다.
# (Colab에서는 '런타임 → 모두 실행'을 쓰는 게 안전하다)
if "STEP" not in globals():
    STEP, SESSION_HOURS, TOKENIZER_DOCS = 1, 3.5, 400_000
    STORAGE, HF_USER = "drive", ""
    print("설정 셀을 건너뛰어 기본값을 씁니다 (STEP=1)\\n")

TIERS = ["configs/tier0_50m.json", "configs/tier1_100m.json", "configs/tier2_200m.json"]
if not 1 <= STEP <= 3:
    raise SystemExit(f"STEP은 1, 2, 3 중 하나여야 합니다 (지금 {{STEP}})")
TIER = TIERS[STEP - 1]

# ── TPU 확인 ────────────────────────────────────────────────────
import jax
devs = jax.devices()
IS_TPU = devs[0].platform == "tpu"
print(f"디바이스   {{len(devs)}}개 · {{devs[0].device_kind}}")
if not IS_TPU:
    print("           ⚠️ TPU가 아니다 → 런타임 → 런타임 유형 변경 → TPU v5e-1")
    print("           토크나이저는 이대로 만들 수 있다. 학습은 건너뛴다.")

# ── 체크포인트 저장소 ───────────────────────────────────────────
# 무료 Colab은 예고 없이 끊기고 /content 는 사라진다.
# 체크포인트를 세션 밖에 두지 않으면 73시간짜리 학습을 끝낼 수 없다.
TOK_HOME = None          # 토크나이저를 둘 곳 (Drive면 폴더, HF면 None)

if STORAGE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    TOK_HOME = Path("/content/drive/MyDrive/cati")
    TOK_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["CATI_CKPT_STORE"] = str(TOK_HOME / "ckpt")
    print(f"저장소     Google Drive · {{TOK_HOME}}")

elif STORAGE == "hf":
    # 토큰: hf.co/settings/tokens 에서 write 권한으로 만들고
    #       좌측 🔑(보안 비밀)에 HF_TOKEN 으로 넣으세요.
    from google.colab import userdata
    try:
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception as e:
        raise SystemExit(
            f"HF_TOKEN을 못 찾았습니다 ({{type(e).__name__}}).\\n"
            "  1. hf.co/settings/tokens → New token → Write 권한으로 생성\\n"
            "  2. Colab 좌측 🔑(보안 비밀) → 이름 HF_TOKEN, 값 붙여넣기, 액세스 켜기\\n"
            "  또는 맨 위 STORAGE 를 'drive' 로 바꾸세요.")
    if not HF_USER:
        raise SystemExit("맨 위 HF_USER 에 Hugging Face 사용자명을 넣으세요.")
    os.environ["CATI_HF_REPO"] = f"{{HF_USER}}/cati-ckpt"
    print(f"저장소     Hugging Face · {{os.environ['CATI_HF_REPO']}} (비공개)")

else:
    raise SystemExit("STORAGE 는 'drive' 또는 'hf' 여야 합니다")

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

# ── 토크나이저: 저장소에 있으면 재사용, 없으면 만들어 보관한다 ──
# 전 티어가 같은 토크나이저를 공유해야 사다리 실험을 비교할 수 있다.
TOK = Path("artifacts/tokenizer/tokenizer.json")
TOK.parent.mkdir(parents=True, exist_ok=True)
SAVED = (TOK_HOME / "tokenizer.json") if TOK_HOME else None

if SAVED is not None and SAVED.exists():
    shutil.copy(SAVED, TOK)
    print("토크나이저 저장소에서 재사용")
elif TOK.exists():
    print("토크나이저 git 커밋본 사용")
    if SAVED is not None:
        shutil.copy(TOK, SAVED)
else:
    print("토크나이저 없음 → 새로 학습 (20~40분, 처음 한 번만)\\n")
    subprocess.run([sys.executable, "scripts/train_tokenizer.py",
                    "train", "--docs", str(TOKENIZER_DOCS)], check=True)
    assert TOK.exists(), "학습이 끝났는데 파일이 없다 — 위 출력 확인"
    if SAVED is not None:
        shutil.copy(TOK, SAVED)
        print(f"\\n토크나이저 보관: {{SAVED}}")
    else:
        print("\\n⚠️ 이 토크나이저는 세션과 함께 사라집니다.")
        print("   git에 커밋해두면 다음부터 다시 만들지 않습니다:")
        print("   !cd /content/Cati && git add -f artifacts/tokenizer/tokenizer.json")

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
import os, subprocess, sys, time

if not IS_TPU:
    print("TPU가 아니라 학습을 건너뜁니다.")
    print("런타임 → 런타임 유형 변경 → TPU v5e-1 로 바꾸고 다시 실행하세요.")
else:
    # 준비 단계(토크나이저 등)에 쓴 시간을 빼야 세션 가드가 실제 남은 시간을 안다.
    used = (time.monotonic() - NB_START) / 3600
    left = SESSION_HOURS - used
    print(f"준비에 {used*60:.0f}분 사용 · 학습에 {left:.2f}시간 배정\\n")
    if left < 0.2:
        print("남은 시간이 너무 적습니다. 이 노트북을 다시 실행하세요 —")
        print("토크나이저는 이미 만들어졌으니 다음엔 바로 학습으로 갑니다.")
    else:
        # 체크포인트는 /content 에 쓰고(빠름) 저장소로 발행한다(살아남음).
        # 저장소는 준비 셀이 환경변수로 정해뒀다.
        subprocess.run([sys.executable, "scripts/train.py", "--tier", TIER,
                        "--session-hours", f"{left:.3f}",
                        "--ckpt", "/content/ckpt",
                        "--quota-hours", "160"], check=False)
'''

COLAB_RESULT = '''
# ══ 결과 ══
import json, os
from pathlib import Path

steps = sorted(Path("/content/ckpt").glob("step_*"))
if not steps and TOK_HOME:
    steps = sorted((TOK_HOME / "ckpt").glob("step_*"))
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
    print(f"저장소: {os.environ.get('CATI_HF_REPO') or os.environ.get('CATI_CKPT_STORE')}")
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
2. **런타임 → 모두 실행** ← 셀을 하나씩 누르지 마세요. 순서가 어긋납니다
3. Drive 연결 권한 허용 (기본 설정) — 체크포인트를 여기 저장합니다

### 체크포인트를 Drive에 두기 싫으면
맨 위 `STORAGE = "hf"` 로 바꾸고 `HF_USER` 에 Hugging Face 사용자명을 넣으세요.
토큰은 hf.co/settings/tokens 에서 **Write** 권한으로 만들어 Colab 좌측
🔑(보안 비밀)에 `HF_TOKEN` 이름으로 넣습니다. 파일 다운로드가 필요 없습니다.

### 그 다음부터
**런타임 → 모두 실행** 을 다시 누르면 이어집니다.
짧은 세션을 여러 번 돌려도 진행이 누적됩니다.

### ⚠️ 무료 Colab의 제약 두 가지
- **탭을 닫으면 멈춥니다.** 백그라운드 실행은 유료 기능입니다. 절전도 꺼두세요.
- **사용량 제한이 유동적입니다.** 갑자기 끊길 수 있습니다.

끊겨도 200스텝마다 로컬 저장, 800스텝마다 저장소로 발행하므로
잃는 건 최대 1시간치입니다.

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

STORAGE = "drive"      # 체크포인트를 어디에 둘까: "drive" 또는 "hf"
HF_USER = ""           # STORAGE="hf" 일 때만. Hugging Face 사용자명

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


def verify(nb: dict, name: str) -> list[str]:
    """생성한 노트북의 코드 셀이 파이썬으로 파싱되는지 확인한다.

    이걸 안 해서 깨진 노트북을 커밋한 적이 있다. 원인은 f-string 안의 `\\n` —
    생성기 소스에서 `\\n` 이라고 쓰면 생성물에는 실제 줄바꿈이 들어가
    문자열 리터럴이 끊긴다. 반드시 `\\\\n` 으로 써야 한다.
    """
    problems = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(l.lstrip().startswith(("!", "%")) for l in c["source"]):
            continue          # 셸/매직 명령은 파이썬 문법이 아니다
        try:
            compile(src, f"{name}:cell{i}", "exec")
        except SyntaxError as e:
            problems.append(f"{name} cell{i} line {e.lineno}: {e.msg}")
    return problems


def main():
    OUT.mkdir(exist_ok=True)
    for old in ("01_tokenizer.ipynb", "02_train.ipynb", "cati_train.ipynb"):
        p = OUT / old
        if p.exists():
            p.unlink()
            print(f"삭제: notebooks/{old}")

    built, problems = [], []
    for name, nb in [("cati_colab", build_colab()),
                     ("cati_tokenizer", build_tokenizer_nb())]:
        problems += verify(nb, name)
        built.append((name, nb))

    if problems:
        print("생성 중단 — 코드 셀에 문법 오류가 있다:")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)

    for name, nb in built:
        path = OUT / f"{name}.ipynb"
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"{path.relative_to(ROOT)}  (셀 {len(nb['cells'])}개 / 코드 {n_code}개 · 문법 OK)")


if __name__ == "__main__":
    main()
