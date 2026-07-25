#!/usr/bin/env python3
"""Cati 자체 BPE 토크나이저 학습 및 압축률 측정.

한국어 비중을 의도적으로 높여 학습한다. 한국어 압축률이 좋아지면
같은 컨텍스트 길이에 더 많은 내용이 들어가므로, 2.5GB 예산 안에서
실질 성능을 올리는 가장 싼 방법이다.

⚠️ 이 토크나이저는 W1에 확정하고 전 티어가 공유한다.
   중간에 바꾸면 사다리 실험 결과가 전부 비교 불가능해진다.

사용법:
    # 로컬 스모크 테스트 (인터넷 불필요, 작은 vocab)
    python3 scripts/train_tokenizer.py smoke

    # 본 학습 (Kaggle, 인터넷 켜기)
    python3 scripts/train_tokenizer.py train --docs 2000000

    # 압축률 측정
    python3 scripts/train_tokenizer.py measure --baseline Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "configs" / "tokenizer.json"
OUT_DIR = ROOT / "artifacts" / "tokenizer"

# configs/data.json 의 pretrain 소스를 그대로 쓴다. 한 곳에서만 관리한다.
def _sources_from_config() -> dict:
    data = json.loads((ROOT / "configs" / "data.json").read_text())
    out = {}
    for s in data["pretrain"]["sources"]:
        key = {"ko": "korean", "en": "english"}.get(s["name"], s["name"])
        out[key] = (s["repo"], s.get("config"), s.get("field", "text"))
    return out


SOURCES = _sources_from_config()


def build_tokenizer(cfg):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

    tok = Tokenizer(models.BPE(unk_token=None))
    # NFC: 한글 조합형/분해형 통일. macOS가 NFD를 쓰므로 이게 없으면
    # 같은 단어가 두 가지로 토큰화된다.
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        # 숫자를 한 자씩 분리 — 산술 일관성에 도움이 된다.
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def make_trainer(cfg, vocab_size=None):
    from tokenizers import pre_tokenizers, trainers

    return trainers.BpeTrainer(
        vocab_size=vocab_size or cfg["vocab_size"],
        special_tokens=cfg["special_tokens"],
        # 256개 바이트를 모두 알파벳에 넣어야 UNK가 원천적으로 발생하지 않는다.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )


def weighted_corpus(mix, total_docs, seed=0):
    """가중치에 따라 여러 스트리밍 데이터셋에서 문서를 섞어 내보낸다.

    접근할 수 없는 소스는 건너뛰고 가중치를 재정규화한다 — 공개 데이터셋이
    예고 없이 gated로 바뀌는 일이 있어서 하나 때문에 전체가 죽으면 안 된다.
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from cati.stream import HFSource, usable_sources

    rng = random.Random(seed)
    candidates, weights = [], []
    for kind, weight in mix.items():
        if kind.startswith("_") or weight <= 0:
            continue
        repo, config, field = SOURCES[kind]
        print(f"  {kind:8s} w={weight:.2f}  {repo}" + (f" [{config}]" if config else ""))
        candidates.append(HFSource(kind, repo, config, field, min_chars=200))
        weights.append(weight)

    print("  소스 접근 확인 중... (데이터셋별로 30초쯤 걸린다)", flush=True)
    sources, weights, _ = usable_sources(candidates, weights)
    streams = [(s.iterator(), s.name) for s in sources]
    print(f"  스트리밍 시작 — 목표 {total_docs:,} 문서\n", flush=True)

    counts = {name: 0 for _, name in streams}
    emitted = 0
    t0 = last = time.monotonic()
    # 진행 표시는 촘촘해야 한다. 10만마다 찍으면 첫 줄까지 10분 넘게 걸려
    # 멈춘 것과 구분이 안 된다.
    every = max(1000, total_docs // 40)
    while emitted < total_docs:
        (it, kind), = rng.choices(streams, weights=weights, k=1)
        try:
            text = next(it)
        except StopIteration:
            continue
        counts[kind] += 1
        emitted += 1
        if emitted % every == 0:
            now = time.monotonic()
            rate = every / max(1e-9, now - last)
            eta = (total_docs - emitted) / max(1e-9, rate) / 60
            print(f"  {emitted:>7,} / {total_docs:,} ({emitted/total_docs:4.0%})  "
                  f"{rate:>5.0f} docs/s  남은 시간 {eta:4.1f}분  {counts}", flush=True)
            last = now
        yield text
    print(f"\n  스트리밍 완료 — {emitted:,} 문서 / {(time.monotonic()-t0)/60:.1f}분",
          flush=True)
    print("  이제 BPE 병합을 계산한다 (몇 분 걸리고 진행 표시가 없다)", flush=True)


def local_corpus():
    """인터넷 없이 파이프라인을 검증하기 위한 최소 코퍼스."""
    ko = ("고양이는 창가에 앉아 오래 밖을 바라보았다. 눈이 내리기 시작했고, "
          "지붕 위로 흰 가루가 얇게 덮였다. 그는 글을 쓰다 말고 고개를 들었다. ")
    en = ("The white cat sat by the window for a long time. Snow began to fall, "
          "covering the roofs in a thin pale dust. He looked up from his writing. ")
    code = ("def summarize(text: str, max_tokens: int = 128) -> str:\n"
            "    tokens = tokenizer.encode(text)\n"
            "    return tokenizer.decode(tokens[:max_tokens])\n")
    for i in range(4000):
        yield ko * 3 if i % 3 == 0 else (en * 3 if i % 3 == 1 else code * 3)


PROBES = [
    "고양이는 창가에 앉아 오래 밖을 바라보았다. 눈이 내리기 시작했다.",
    "오프라인으로 구동하는 인공지능 비서를 처음부터 학습시키려고 합니다.",
    "그는 아무 말도 하지 않았지만, 그 침묵이 대답이었다.",
]


def _report(tok):
    print("\n한국어 압축률 (글자/토큰, 높을수록 좋다)")
    total_c = total_t = 0
    for s in PROBES:
        n = len(tok.encode(s).ids)
        total_c += len(s)
        total_t += n
        print(f"  {len(s):3d}자 → {n:3d}tok  {len(s)/n:.2f}")
    print(f"  전체: {total_c/total_t:.2f} 글자/토큰")
    return total_c / total_t


def cmd_train(args):
    cfg = json.loads(CFG_PATH.read_text())
    vocab = args.vocab or cfg["vocab_size"]
    print(f"vocab_size={vocab:,}  docs={args.docs:,}")
    print("데이터 소스:")

    tok = build_tokenizer(cfg)
    trainer = make_trainer(cfg, vocab)
    corpus = weighted_corpus(cfg["training_mix"], args.docs, seed=args.seed)
    tok.train_from_iterator(corpus, trainer=trainer, length=args.docs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "tokenizer.json"
    tok.save(str(path))
    print(f"\n저장: {path.relative_to(ROOT)}  (vocab {tok.get_vocab_size():,})")
    _report(tok)


def cmd_smoke(args):
    """작은 vocab으로 로컬에서 전체 경로를 검증한다."""
    cfg = json.loads(CFG_PATH.read_text())
    tok = build_tokenizer(cfg)
    tok.train_from_iterator(local_corpus(), trainer=make_trainer(cfg, 4096))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "tokenizer_smoke.json"
    tok.save(str(path))
    print(f"저장: {path.relative_to(ROOT)}  (vocab {tok.get_vocab_size():,})")
    _report(tok)

    # 왕복 검증 — 바이트 레벨 BPE는 어떤 입력에서도 무손실이어야 한다.
    print("\n왕복(encode→decode) 검증")
    ok_all = True
    for probe in ["고양이가 글을 쓴다", "Cati 2.5GB 오프라인", "def f(): pass", "🐈‍⬛ 이모지"]:
        ids = tok.encode(probe).ids
        back = tok.decode(ids)
        ok = unicodedata.normalize("NFC", back) == unicodedata.normalize("NFC", probe)
        ok_all &= ok
        print(f"  [{'OK' if ok else '실패'}] {len(ids):3d} tok  {probe!r}")
    if not ok_all:
        sys.exit("왕복 검증 실패 — 정규화/디코더 설정을 점검할 것")


def cmd_measure(args):
    from tokenizers import Tokenizer

    path = OUT_DIR / ("tokenizer_smoke.json" if args.smoke else "tokenizer.json")
    if not path.exists():
        sys.exit(f"{path} 가 없다. 먼저 train 또는 smoke 를 돌릴 것.")
    ours = _report(Tokenizer.from_file(str(path)))

    if args.baseline:
        from transformers import AutoTokenizer
        base = AutoTokenizer.from_pretrained(args.baseline)
        c = sum(len(s) for s in PROBES)
        t = sum(len(base.encode(s)) for s in PROBES)
        print(f"\n{args.baseline}: {c/t:.2f} 글자/토큰")
        print(f"→ Cati가 {ours/(c/t):.2f}배 " + ("우수" if ours > c / t else "열등"))

    cfg = json.loads(CFG_PATH.read_text())
    target = cfg["target_compression"]["korean_chars_per_token_min"]
    print(f"\n목표 {target} 대비: {'통과' if ours >= target else '미달 — 한국어 비중을 올릴 것'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="본 학습 (인터넷 필요)")
    p.add_argument("--docs", type=int, default=2_000_000)
    p.add_argument("--vocab", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("smoke", help="로컬 파이프라인 검증")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("measure", help="압축률 측정")
    p.add_argument("--baseline", default=None, help="비교용 기성 토크나이저 (HF id)")
    p.add_argument("--smoke", action="store_true")
    p.set_defaults(fn=cmd_measure)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
