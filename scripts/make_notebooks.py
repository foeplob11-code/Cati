#!/usr/bin/env python3
"""노트북 생성기.

.ipynb 를 손으로 쓰면 JSON 이스케이프 때문에 금방 깨진다. 여기서 만든다.

    python3 scripts/make_notebooks.py

만들어지는 것
    notebooks/cati_kaggle.ipynb     학습 — Kaggle GPU T4 x2 (주력)
    notebooks/cati_colab.ipynb      학습 — Colab TPU v5e-1 (보너스)
    notebooks/cati_tokenizer.ipynb  토크나이저만 — Kaggle CPU (선택)

두 곳을 쓰는 이유. Colab v5e-1은 빠르지만(69 TFLOPS) 무료 한도가 주 1~2시간에
불과하고 백그라운드 실행도 안 된다. Kaggle T4 x2는 느리지만(26~32 TFLOPS)
주 30시간이 보장되고 창을 닫아도 돈다 — 총 계산량이 10~30배 많다.
그래서 Kaggle이 주력, Colab은 한도가 회복될 때 얹는 보너스다.

체크포인트를 HF Hub에 두면 양쪽이 같은 학습을 이어받는다. 파라미터가 fp32로
저장되므로 bf16(Colab)과 fp16(Kaggle) 사이를 오가도 안전하다.
단 동시에 돌리면 서로 덮어쓴다.
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
os.environ["PYTHONUNBUFFERED"] = "1"


def run_script(name, args):
    """스크립트를 **이 프로세스 안에서** 실행한다. 자식 프로세스로 띄우지 않는다.

    이유 두 가지.
      1. TPU는 한 프로세스만 점유할 수 있다. 준비 셀이 jax.devices() 로
         TPU를 확인하는 순간 노트북이 TPU를 잡으므로, 자식 프로세스는
         "The TPU is already in use by process with pid ..." 로 죽는다.
      2. ipykernel이 sys.stdout 을 파이썬 레벨에서 갈아치우기 때문에,
         자식 프로세스의 출력은 노트북 셀에 안 보인다.

    같은 프로세스에서 돌리면 둘 다 해결된다.
    """
    import importlib
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    mod = importlib.import_module(name)
    importlib.reload(mod)        # git pull 로 갱신됐을 수 있다
    return mod.main(args)

# 설정 셀을 건너뛰고 이 셀부터 실행해도 죽지 않게 기본값을 채운다.
# (Colab에서는 '런타임 → 모두 실행'을 쓰는 게 안전하다)
if "STEP" not in globals():
    STEP, SESSION_HOURS, TOKENIZER_DOCS = 1, 3.5, 400_000
    STORAGE, HF_USER = "hf", ""
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
    run_script("train_tokenizer", ["train", "--docs", str(TOKENIZER_DOCS)])
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
        run_script("train", ["--tier", TIER,
                             "--session-hours", f"{left:.3f}",
                             "--ckpt", "/content/ckpt",
                             "--quota-hours", "160"])
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
3. 맨 위 `HF_USER` 에 Hugging Face 사용자명 입력
4. Colab 좌측 **🔑(보안 비밀)** 에 `HF_TOKEN` 추가
   (hf.co/settings/tokens → New token → **Write** 권한)

### 왜 HF Hub인가
**Kaggle 노트북과 같은 체크포인트를 씁니다.** 어느 쪽에서 이어받든 상관없습니다.
Google Drive는 Kaggle에서 못 읽어서 안 됩니다.
Colab 전용으로 쓰려면 `STORAGE = "drive"` 로 바꾸면 됩니다.

⚠️ **두 곳에서 동시에 돌리지 마세요.** 서로 덮어씁니다.

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

STORAGE = "hf"         # "hf" = Kaggle과 체크포인트 공유 (권장) · "drive" = Colab 전용
HF_USER = ""           # ← Hugging Face 사용자명

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


def run_script(name, args):
    """스크립트를 이 프로세스 안에서 실행한다 (자식 프로세스 출력은 셀에 안 보인다)."""
    import importlib
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    mod = importlib.import_module(name)
    importlib.reload(mod)
    return mod.main(args)

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
    run_script("train_tokenizer", ["train", "--docs", str(TOKENIZER_DOCS)])

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



# ===========================================================================
# Kaggle — 학습 (GPU T4 x2, 주력)
# ===========================================================================
KAGGLE_TRAIN = f'''
# ══ 준비 ══
import os, shutil, subprocess, sys, time
from pathlib import Path

NB_START = time.monotonic()

if "STEP" not in globals():
    STEP, SESSION_HOURS, TOKENIZER_DOCS, HF_USER = 1, 11.0, 400_000, ""

TIERS = ["configs/tier0_50m.json", "configs/tier1_100m.json", "configs/tier2_200m.json"]
if not 1 <= STEP <= 3:
    raise SystemExit(f"STEP은 1, 2, 3 중 하나여야 합니다 (지금 {{STEP}})")
TIER = TIERS[STEP - 1]


def run_script(name, args):
    """스크립트를 이 프로세스 안에서 실행한다 (자식 프로세스는 GPU/로그 문제가 생긴다)."""
    import importlib
    sys.path.insert(0, str(Path.cwd() / "scripts"))
    mod = importlib.import_module(name)
    importlib.reload(mod)
    return mod.main(args)


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
# Kaggle 이미지에는 JAX가 CPU 버전으로 깔려 있다. CUDA 버전으로 갈아끼운다.
print("패키지 설치 중 (3~5분)...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "jax[cuda12]", "flax", "optax", "orbax-checkpoint",
                "tokenizers>=0.22", "datasets>=3.0", "huggingface_hub>=0.27"],
               check=False)

import jax
devs = jax.devices()
print(f"디바이스   {{len(devs)}}개 · {{devs[0].device_kind}} · {{devs[0].platform}}")
if devs[0].platform != "gpu":
    print("           ⚠️ GPU가 아니다 → Settings → Accelerator → GPU T4 x2")

# ── HF Hub: Colab과 공용 저장소 ─────────────────────────────────
# Kaggle과 Colab이 같은 체크포인트를 주고받으려면 양쪽에서 접근 가능한 곳이어야 한다.
# Google Drive는 Kaggle에서 못 읽으므로 HF Hub를 쓴다.
from kaggle_secrets import UserSecretsClient
try:
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    raise SystemExit(
        f"HF_TOKEN을 못 찾았습니다 ({{type(e).__name__}}).\\n"
        "  1. huggingface.co/settings/tokens → New token → Write 권한\\n"
        "  2. 이 노트북 → Add-ons → Secrets → 이름 HF_TOKEN, 값 붙여넣기")
if not HF_USER:
    raise SystemExit("맨 위 HF_USER 에 Hugging Face 사용자명을 넣으세요.")
os.environ["CATI_HF_REPO"] = f"{{HF_USER}}/cati-ckpt"
print(f"저장소     {{os.environ['CATI_HF_REPO']}} (비공개)")

# ── 토크나이저 ──────────────────────────────────────────────────
TOK = Path("artifacts/tokenizer/tokenizer.json")
TOK.parent.mkdir(parents=True, exist_ok=True)
if not TOK.exists():
    from huggingface_hub import hf_hub_download
    try:
        got = hf_hub_download(f"{{HF_USER}}/cati-ckpt", "tokenizer.json",
                              repo_type="model")
        shutil.copy(got, TOK)
        print("토크나이저 HF에서 재사용")
    except Exception:
        print("토크나이저 없음 → 새로 학습 (20~40분, 처음 한 번만)\\n")
        run_script("train_tokenizer", ["train", "--docs", str(TOKENIZER_DOCS)])
        from huggingface_hub import HfApi
        HfApi().create_repo(f"{{HF_USER}}/cati-ckpt", private=True,
                            exist_ok=True, repo_type="model")
        HfApi().upload_file(path_or_fileobj=str(TOK), path_in_repo="tokenizer.json",
                            repo_id=f"{{HF_USER}}/cati-ckpt", repo_type="model")
        print("토크나이저를 HF에 올렸습니다 (Colab에서도 같은 걸 씁니다)")
else:
    print("토크나이저 저장소 커밋본 사용")

from tokenizers import Tokenizer
_t = Tokenizer.from_file(str(TOK))
_p = "고양이는 창가에 앉아 오래 밖을 바라보았다."
_r = len(_p) / len(_t.encode(_p).ids)
print(f"           vocab {{_t.get_vocab_size():,}} · 한국어 {{_r:.2f}} 글자/토큰")
print("\\n준비 완료")
'''

KAGGLE_TRAIN_RUN = '''
# ══ 학습 ══ Save & Run All 로 돌리면 창을 닫아도 12시간을 다 씁니다.
#
#  loss   10.8 에서 시작해 내려가야 한다
#  scale  fp16 손실 스케일링 배율. 자주 반토막나면 학습률이 높은 것
#  버림   넘쳐서 버린 스텝 수. 전체의 몇 % 를 넘으면 문제
import time

used = (time.monotonic() - NB_START) / 3600
left = SESSION_HOURS - used
print(f"준비에 {used*60:.0f}분 사용 · 학습에 {left:.2f}시간 배정\\n")
if left < 0.2:
    print("남은 시간이 너무 적습니다. 다시 실행하세요.")
else:
    run_script("train", ["--tier", TIER,
                         "--session-hours", f"{left:.3f}",
                         "--ckpt", "/kaggle/working/ckpt",
                         "--quota-hours", "240"])
'''

KAGGLE_RESULT = '''
# ══ 결과 ══
import json
from pathlib import Path

steps = sorted(Path("/kaggle/working/ckpt").glob("step_*"))
if not steps:
    print("체크포인트가 없다 — 위 학습 셀 출력을 확인할 것")
else:
    m = json.loads((steps[-1] / "meta.json").read_text())
    pct = m["tokens"] / m["target_tokens"]
    print(f"{m['step']:,}스텝   {m['tokens']/1e9:.2f}B / "
          f"{m['target_tokens']/1e9:.0f}B 토큰   {pct:.1%}")
    print(f"누적 {m.get('device_hours_used', 0):.1f}시간 · 세션 #{m.get('session_index', 1)}")
    print()
    if pct < 1.0:
        print("아직 진행 중  →  Save & Run All 을 다시 누르세요 (이어집니다)")
        print("            또는 Colab 노트북에서 이어받아도 됩니다")
    elif STEP < 3:
        print(f"이 티어 완료  →  맨 위 STEP 을 {STEP + 1} 로 바꾸고 다시 실행")
    else:
        print("사전학습 완료  →  다음은 도서 어닐링 (문체 학습)")
'''


def build_kaggle_train() -> dict:
    return notebook([
        md('''
# Cati 학습 — Kaggle GPU T4 x2 (주력)

주 30시간이 **보장**되고 `Save & Run All` 로 돌리면 창을 닫아도 계속 돕니다.
무료 Colab보다 느리지만 총 계산량은 10~30배 많습니다.

### 처음 한 번만
1. **Settings → Accelerator → GPU T4 x2** · Internet **On**
2. **Add-ons → Secrets** 에 `HF_TOKEN` 추가
   (huggingface.co/settings/tokens → New token → **Write** 권한)
3. 아래 `HF_USER` 에 Hugging Face 사용자명 입력
4. **Save Version → Save & Run All** → 창 닫기

### 그 다음부터
**Save & Run All** 만 다시 누르면 이어집니다.

체크포인트가 HF Hub에 올라가므로 **Colab에서 이어받아도 됩니다.**
단 동시에 돌리지는 마세요 — 서로 덮어씁니다.

### T4는 bf16이 없습니다
fp16 + 동적 손실 스케일링으로 돕니다. 로그의 `scale` 과 `버림` 을 보세요.
버린 스텝이 전체의 20%를 넘으면 자동으로 멈추고 알려줍니다.

### 순서
| STEP | 모델 | T4x2 시간 |
|---|---|---|
| 1 | 50M | 약 5h |
| 2 | 100M | 약 22h |
| 3 | **200M** | **약 170h** |
'''),
        code('''
STEP = 1               # 1=50M   2=100M   3=200M
HF_USER = ""           # ← Hugging Face 사용자명

SESSION_HOURS = 11.0   # Kaggle GPU 세션 한계 12시간, 여유 1시간
TOKENIZER_DOCS = 400_000
'''),
        code(KAGGLE_TRAIN),
        code(KAGGLE_TRAIN_RUN),
        code(KAGGLE_RESULT),
    ])


def main():
    OUT.mkdir(exist_ok=True)
    for old in ("01_tokenizer.ipynb", "02_train.ipynb", "cati_train.ipynb"):
        p = OUT / old
        if p.exists():
            p.unlink()
            print(f"삭제: notebooks/{old}")

    built, problems = [], []
    for name, nb in [("cati_kaggle", build_kaggle_train()),
                     ("cati_colab", build_colab()),
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
