# Cati 🐈

오프라인으로 구동하는 에이전틱 AI 비서. 모델은 **처음부터 학습**한다.

- **모델**: cati-350m (343.5M dense) — Kaggle 무료 TPU로 25B 토큰 사전학습
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
  tier2_350m.json   최종 출시 모델
cati/             # 학습 인프라
  session.py        9시간 세션 가드 (정상종료/업로드 데드라인)
  stream.py         재개 가능한 데이터 스트림 ← 가장 틀리기 쉬운 부분
  checkpoint.py     원자적 저장 + 배열 백엔드 (Orbax / numpy)
  store.py          Kaggle Dataset 영속화
  telemetry.py      JSONL 로그 + MFU/쿼터 추정
  runner.py         위 전부를 묶은 세션 오케스트레이터
scripts/
  budget.py           파라미터/계산량/쿼터 계산기 — 설정 바꿀 때마다 돌린다
  train_tokenizer.py  토크나이저 학습 및 압축률 측정
  test_resume.py      재개 정확성 검증 (29 검사)
artifacts/        # 학습 산출물 (git 제외)
app/              # Tauri 앱 (제품 트랙, 미착수)
```

## 체크포인트/재개

Kaggle 세션은 9시간에 예고 없이 죽고, 새 세션은 `/kaggle/working` 이 비어 있다.
97시간짜리 350M 런은 최소 11번 끊긴다. 그래서 재개는 **정확해야** 한다.

```python
from cati import ResumableRun, ResumableStream, HFSource, default_store

run = ResumableRun("cati-350m", "/kaggle/working/ckpt",
                   params=343_500_000, target_tokens=25_000_000_000,
                   store=default_store(), save_every=500)

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
run.finish(arrays, step, tokens, stream)              # 저장 + Kaggle Dataset 발행
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

설정이 Kaggle 쿼터 안에 들어오는지 확인 (설정을 바꿀 때마다 실행):

```bash
python3 scripts/budget.py
```

토크나이저 파이프라인 검증 (인터넷 불필요):

```bash
.venv/bin/python scripts/train_tokenizer.py smoke
```

토크나이저 본 학습 (Kaggle, 인터넷 켜기):

```bash
.venv/bin/python scripts/train_tokenizer.py train --docs 2000000
```

---

## 지켜야 할 제약

이 세 가지는 어기면 되돌리는 비용이 매우 크다.

1. **토크나이저는 W1에 확정하고 전 티어가 공유한다.**
   중간에 바꾸면 사다리 실험 결과가 전부 비교 불가능해진다.
2. **Kaggle TPU 쿼터는 이월되지 않는다.** 주 20시간이 상한이라
   97시간짜리 350M 런은 최소 5주의 벽시계 시간이 필요하다.
3. **체크포인트/재개 인프라를 학습 코드보다 먼저 만든다.**
   Kaggle 세션은 9시간에 강제 종료된다.

## Kaggle에서 돌리기

노트북 2개를 순서대로 실행한다.

### 01_tokenizer (CPU 세션 — TPU 쿼터 안 씀)

1. Kaggle에 [notebooks/01_tokenizer.ipynb](notebooks/01_tokenizer.ipynb) 업로드
2. Settings → Accelerator **None** · Internet **On**
3. 첫 셀의 `GITHUB_URL` 또는 코드 Dataset 설정
4. **Save Version → Save & Run All**

FineWeb2 한국어 스트리밍을 검증하고 토크나이저를 학습한다. 10~30분.

### 02_train (TPU 세션)

1. [notebooks/02_train.ipynb](notebooks/02_train.ipynb) 업로드
2. Settings → Accelerator **TPU VM v3-8** · Internet **On**
3. Add Input → Notebook Output → 01_tokenizer
4. `TIER` 를 고른다: `tier0_50m` → `tier1_100m` → `tier2_350m`
5. **Save Version → Save & Run All**

`Save & Run All` 로 돌리면 브라우저를 닫아도 백그라운드에서 9시간을 다 쓴다.
세션이 끝나면 **다시 Save & Run All** 하면 체크포인트에서 이어진다.

| 티어 | TPU 시간 | 세션 수 |
|---|---|---|
| tier0_50m | 1.1h | 1 |
| tier1_100m | 4.4h | 1 |
| tier2_350m | 97.4h | 11 (5주) |

체크포인트를 세션 간에 넘기려면 Add-ons → Secrets 에 `KAGGLE_USERNAME` / `KAGGLE_KEY`
를 넣는다. 없으면 매 세션 이전 Output을 Input으로 직접 붙여야 한다.

## 현재 상태

- [x] 계획 확정 (PLAN.md v0.4)
- [x] 티어 설정 + 쿼터 검증 (102.9h / 160h, 버퍼 36%)
- [x] 토크나이저 설정 확정 + 파이프라인 검증
- [x] 체크포인트/재개 인프라 — `test_resume.py` 35/35
- [x] 모델 (JAX/Flax) — `test_model.py` 24/24, 파라미터 수 계산과 정확히 일치
- [x] 토큰 패킹 + 학습 루프 — CPU 스모크에서 재개까지 확인
- [x] Kaggle 노트북 2개
- [ ] **01_tokenizer 실행** ← 여기서부터 사람 손이 필요
- [ ] 02_train 으로 50M → 100M → 350M
- [ ] `HFSource` 네이티브 재개 실증 (01 노트북이 자동 확인)
- [ ] MFU 실측 (35% 가정 검증 — 25% 미만이면 토큰 수 하향)
- [ ] 제품 트랙: Tauri 셸 + llama.cpp
