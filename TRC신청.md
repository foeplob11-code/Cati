# TRC 신청

TPU Research Cloud. Google이 연구·학습 목적에 TPU를 30일 이상 무료로 준다.
주당 시간 제한도, 세션 제한도 없다.

## 왜 신청하나

| | Colab v5e-1 | TRC v3-8 |
|---|---|---|
| 실효 성능 | 69 TFLOPS | **147 TFLOPS** (2.1배) |
| 주당 한도 | 유동적 (~20h) | **없음** |
| 백그라운드 실행 | ✗ (탭 열어둬야) | ✓ |
| 최종 모델 | 200M / 15B 토큰 | **400M / 20B 토큰** |
| 소요 | 73h = 4.2주 | 91h = 30일 안에 여유 |

## 신청 절차

1. https://sites.research.google/trc/about/ → **Apply now**
2. 아래 양식을 채운다 (영어)
3. 승인 메일이 오면 Google Cloud 프로젝트를 연결한다
4. TPU VM을 만들고 코드를 받아 실행한다 (스크립트는 승인 후 제공)

**필요한 것**: Google 계정 + Google Cloud 프로젝트.
GCP 계정 생성 시 카드 등록을 요구하지만 **TPU는 무료 할당이라 청구되지 않는다.**

**소요**: 승인까지 며칠 ~ 2주.

---

## 신청서에 쓸 내용

제목 / 연구 요약에 붙여 쓰면 된다. 전부 사실이다.

```
Project: Cati — an offline Korean AI assistant trained from scratch

I am pretraining a small Llama-architecture language model from scratch to power
a fully offline AI assistant for Korean users. The model runs entirely on a
laptop with no network access, in a 2.5GB application bundle.

What I have done so far:
- Trained a custom byte-level BPE tokenizer (49,152 vocab) weighted toward
  Korean. It achieves 2.18 characters per token on Korean text, versus 0.47 for
  SmolLM2 at the same vocabulary size and 1.39 for Qwen3 at 3x the vocabulary.
  This roughly quadruples the Korean text that fits in a fixed token budget.
- Built the training stack in JAX/Flax with exact checkpoint/resume (verified
  bit-identical across interrupted and uninterrupted runs, including data
  iterator position and RNG state).
- Validated a model ladder of 50M and 100M parameters on Colab TPU v5e-1.

What I need TPU for:
- Pretraining a 400M-parameter model on 20B tokens (65% Korean from FineWeb2,
  25% English from FineWeb-Edu, 10% code). Estimated 91 TPU v3-8 hours.
- A book-heavy annealing phase (3B tokens) to improve Korean prose quality.

Why this needs TRC:
Colab's free tier caps me at roughly 20 hours per week with no background
execution, which limits the model to 200M parameters. TRC's uninterrupted
30-day access would let me train a 400M model that is meaningfully more capable
while remaining small enough to run offline on consumer hardware.

Requested: TPU v3-8 (or v4 if available). Estimated need: ~120 TPU-hours
over 2-3 weeks.

Code is open source: https://github.com/foeplob11-code/Cati
```

---

## 승인되면

1. `configs/alt/400m_trc.json` → `configs/tier2_400m.json` 으로 옮긴다
2. 기존 `configs/tier2_200m.json` 은 `configs/alt/` 로 보낸다
3. 토크나이저와 50M/100M 체크포인트는 그대로 재사용한다 (같은 토크나이저를 공유하므로)
4. 예산 확인: `python3 scripts/budget.py tpuv3`

## 거절되거나 v2-8을 받으면

**계획이 깨지지 않는다.** 200M으로 Colab에서 계속한다.

v2-8은 실효 63 TFLOPS로 Colab v5e-1(69)보다 오히려 느리다. 그 경우 TRC를 쓸
이유가 없다. 승인 메일에 어떤 세대를 주는지 나오므로 확인 후 판단할 것.
