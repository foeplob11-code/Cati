# Cati 🐈

오프라인으로 구동하는 에이전틱 AI 비서. 모델은 **처음부터 학습**한다.

- **모델**: cati-200m (201M dense) — Colab 무료 TPU v5e-1 로 15B 토큰 사전학습
- **배포**: MacBook M1, 2.5GB 단일 앱, 네트워크 불필요
- **성격**: MCP 도구를 호출하는 에이전트 · 글쓰기 특화
- **마스코트**: 하얀 고양이

전체 계획·근거·일정은 **[PLAN.md](PLAN.md)** 참조.

---

## 구조

```
configs/          # 티어 설정 (사다리) + 토크나이저 설정 — 전부 확정본
  tokenizer.json    vocab 49152, NFC, 바이트 레벨 BPE
  tier0_50m.json    파이프라인 검증용
  tier1_100m.json   제품의 의도 분류 라우터
  tier2_200m.json   최종 출시 모델
  alt/350m_tpu.json TPU 확보 시 복귀용
  data.json         데이터 소스 — 여기만 보면 된다
tokenizer         # ← Kaggle 토크나이저 실행 (선택)
cati/             # 학습 인프라
  model.py          Llama 계열 디코더 (Flax) · HF 가중치 호환
  session.py        9시간 세션 가드 (정상종료/업로드 데드라인)
  stream.py         재개 가능한 데이터 스트림 ← 가장 틀리기 쉬운 부분
  packing.py        문서 → 고정 길이 배치 (남은 토큰 버퍼도 재개 대상)
  checkpoint.py     원자적 저장 + 배열 백엔드 (Orbax / numpy)
  store.py          Kaggle Dataset 영속화
  telemetry.py      JSONL 로그 + MFU/쿼터 추정
  runner.py         위 전부를 묶은 세션 오케스트레이터
scripts/
  train.py            사전학습
  budget.py           파라미터/계산량/쿼터 계산기 — 설정 바꿀 때마다 돌린다
  train_tokenizer.py  토크나이저 학습 및 압축률 측정
  kaggle_run.py       ./tokenizer 의 알맹이
  make_notebooks.py   노트북 생성기
  test_resume.py      재개 정확성 검증 (35 검사)
  test_model.py       모델 검증 (24 검사)
notebooks/
  cati_colab.ipynb      학습 (Colab TPU v5e-1)
  cati_tokenizer.ipynb  토크나이저 (Kaggle, 선택)
artifacts/        # 학습 산출물 (git 제외)
app/              # Tauri 앱 (제품 트랙, 미착수)
```

## 체크포인트/재개

무료 Colab은 예고 없이 끊기고 `/content` 는 사라진다. 73시간짜리 200M 런은
수십 번 끊긴다. 그래서 재개는 **정확해야** 한다.

```python
from cati import ResumableRun, ResumableStream, HFSource, default_store

run = ResumableRun("cati-200m", "/content/ckpt",
                   params=201_000_000, target_tokens=15_000_000_000,
                   store=default_store(), save_every=200, publish_every=200)

stream = ResumableStream([HFSource("ko", "HuggingFaceFW/fineweb-2", "kor_Hang"),
                          HFSource("en", "HuggingFaceFW/fineweb-edu", "sample-10BT")],
                         weights=[0.4, 0.45], seed=0)

arrays, step, tokens = run.start(init_fn, stream)     # 복원 또는 새 시작
while True:
    t0 = time.monotonic()
    arrays, loss, n = train_step(arrays, next(batches))
    step, tokens = step + 1, tokens + n
    if not run.tick(step, tokens, time.monotonic() - t0, arrays, stream,
                    loss=loss, step_tokens=n):
        break
run.finish(arrays, step, tokens, stream)              # 저장 + 스토어 발행
```

복원되는 상태: 파라미터 · 옵티마이저 · 스텝 · 누적 토큰 · **데이터 소비 위치** ·
소스 선택 RNG · 고갈된 소스 · 에폭 카운터 · 누적 디바이스 시간.

검증:

```bash
.venv/bin/python scripts/test_resume.py
```

4번 중단하고 재개한 런이 무중단 런과 파라미터·옵티마이저 비트 단위 일치,
문서 소비 순서 240개 완전 일치를 확인한다.

## 사용법

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

설정이 쿼터 안에 들어오는지 확인 (설정을 바꿀 때마다 실행):

```bash
python3 scripts/budget.py v5e1     # t4x2 / p100 / tpuv3 도 가능
```

토크나이저 파이프라인 검증 (인터넷 불필요):

```bash
.venv/bin/python scripts/train_tokenizer.py smoke
```

토크나이저 본 학습 (인터넷 필요):

```bash
.venv/bin/python scripts/train_tokenizer.py train --docs 400000
```

---

## 지켜야 할 제약

이 세 가지는 어기면 되돌리는 비용이 매우 크다.

1. **토크나이저는 W1에 확정하고 전 티어가 공유한다.**
   중간에 바꾸면 사다리 실험 결과가 전부 비교 불가능해진다.
2. **가속기가 bf16을 지원해야 지금 코드가 돈다.** Kaggle의 T4/P100은 지원하지 않는다.
   fp16 경로를 새로 쓰는 대신 Colab v5e-1을 쓴다.
3. **체크포인트/재개 인프라를 학습 코드보다 먼저 만든다.**
   무료 Colab은 예고 없이 끊기고 백그라운드 실행도 안 된다.

## 학습 돌리기

→ **[시작하기.md](시작하기.md)**

**Colab TPU v5e-1** 에서 학습한다. Kaggle에는 TPU 옵션이 없다 (T4/P100뿐).
T4로 200M은 111시간(16주)이고 bf16도 없어 fp16 loss scaling을 따로 써야 한다.
v5e-1은 bf16 네이티브라 지금 코드가 그대로 돈다.

| 가속기 | 실효 성능 | bf16 | 200M 소요 |
|---|---|---|---|
| GPU P100 (Kaggle) | 8 TFLOPS | ✗ | 630h |
| GPU T4 x2 (Kaggle) | 30 TFLOPS | ✗ | 168h |
| **TPU v5e-1 (Colab)** | **69 TFLOPS** | ✓ | **73h** |
| TPU v3-8 (TRC) | 147 TFLOPS | ✓ | 34h |

```bash
python3 scripts/budget.py v5e1     # 쿼터 안에 들어오는지 확인
```

| STEP | 모델 | v5e-1 시간 | 배수 |
|---|---|---|---|
| 1 | 50M | 2.3h | 42x |
| 2 | 100M | 9.5h | 41x |
| 3 | 200M | 73h | 75x |

합계 85h / 160h (버퍼 47%). 주 20시간이면 4.2주.

토크나이저만 Kaggle에서 만들어 Colab 시간을 아낄 수 있다 (선택): `./tokenizer`

## 현재 상태

- [x] 계획 확정 (PLAN.md)
- [x] 티어 설정 + 쿼터 검증 (85h / 160h, 버퍼 47%)
- [x] 토크나이저 설정 확정 + 파이프라인 검증
- [x] 체크포인트/재개 인프라 — `test_resume.py` 35/35
- [x] 모델 (JAX/Flax) — `test_model.py` 24/24, 파라미터 수 계산과 정확히 일치
- [x] 토큰 패킹 + 학습 루프 — CPU 스모크에서 재개까지 확인
- [x] Colab 노트북 (학습) + Kaggle 노트북 (토크나이저)
- [ ] **Colab에서 50M** ← 여기서부터 사람 손이 필요
- [ ] 100M → 200M
- [ ] `HFSource` 네이티브 재개 실증 (첫 세션이 자동 확인)
- [ ] MFU 실측 (35% 가정 검증 — 25% 미만이면 토큰 수 하향)
- [ ] 제품 트랙: Tauri 셸 + llama.cpp
