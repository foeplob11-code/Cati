#!/usr/bin/env python3
"""티어 설정의 파라미터 수 / 계산량 / Kaggle TPU 쿼터 소요를 계산한다.

설정을 바꿀 때마다 이걸 돌려서 쿼터 안에 들어오는지 확인한다.

    python3 scripts/budget.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Kaggle 무료 티어 TPU v3-8 (W1에 실측해서 갱신할 것)
TPU_PEAK_TFLOPS = 420.0   # bf16 peak
TPU_MFU = 0.35            # 실효 비율 가정
TPU_HOURS_PER_WEEK = 20.0
WEEKS = 8

TPU_FLOPS = TPU_PEAK_TFLOPS * 1e12 * TPU_MFU
FLOPS_PER_TPU_HOUR = TPU_FLOPS * 3600
TOTAL_TPU_HOURS = TPU_HOURS_PER_WEEK * WEEKS


def count_params(c):
    V, d, L = c["vocab_size"], c["d_model"], c["n_layers"]
    nh, nkv, hd, f = c["n_heads"], c["n_kv_heads"], c["head_dim"], c["d_ff"]

    embed = V * d
    attn = (d * nh * hd) + 2 * (d * nkv * hd) + (nh * hd * d)
    mlp = 3 * d * f
    norms = 2 * d
    per_layer = attn + mlp + norms

    body = L * per_layer + d                      # + 최종 norm
    head = 0 if c["tie_embeddings"] else V * d
    return {
        "embed": embed,
        "per_layer": per_layer,
        "body": body,
        "total": embed + body + head,
        "non_embed": body + head,
    }


def kv_cache_bytes_per_token(c, dtype_bytes=2):
    """K와 V 두 개 × 층수 × KV헤드 × head_dim."""
    return 2 * c["n_layers"] * c["n_kv_heads"] * c["head_dim"] * dtype_bytes


def main():
    tiers = sorted(ROOT.glob("configs/tier*.json"))
    if not tiers:
        sys.exit("configs/tier*.json 를 찾을 수 없다")

    rows, total_hours = [], 0.0
    for path in tiers:
        c = json.loads(path.read_text())
        p = count_params(c)
        D = c["train_tokens"]
        flops = 6 * p["total"] * D
        hours = flops / FLOPS_PER_TPU_HOUR
        total_hours += hours

        kv = kv_cache_bytes_per_token(c)
        # AdamW: fp32 master + m + v = 12 B/param, bf16 사본 2 B/param
        ckpt_gb = p["total"] * 14 / 1024**3

        rows.append({
            "name": c["name"],
            "params_m": p["total"] / 1e6,
            "non_embed_m": p["non_embed"] / 1e6,
            "embed_pct": 100 * p["embed"] / p["total"],
            "tokens_b": D / 1e9,
            "ratio": D / p["total"],
            "flops": flops,
            "hours": hours,
            "weeks": hours / TPU_HOURS_PER_WEEK,
            "kv_kb": kv / 1024,
            "kv_8k_mb": kv * 8192 / 1024**2,
            "ckpt_gb": ckpt_gb,
            "gguf_f16_gb": p["total"] * 2 / 1024**3,
            "gguf_q8_gb": p["total"] * 8.5 / 8 / 1024**3,
            "gguf_q4_gb": p["total"] * 4.83 / 8 / 1024**3,
        })

    print(f"\nKaggle TPU v3-8 가정: {TPU_PEAK_TFLOPS:.0f} TFLOPS peak x MFU {TPU_MFU:.0%} "
          f"= {TPU_FLOPS/1e12:.0f} TFLOPS 실효")
    print(f"쿼터: {TPU_HOURS_PER_WEEK:.0f} h/주 x {WEEKS}주 = {TOTAL_TPU_HOURS:.0f} TPU-시간 "
          f"(이월 불가)\n")

    print(f"{'티어':<12} {'전체':>9} {'비임베딩':>9} {'임베딩%':>7} "
          f"{'토큰':>8} {'배수':>6} {'FLOPs':>9} {'TPU-h':>7} {'주':>5}")
    print("-" * 84)
    for r in rows:
        print(f"{r['name']:<12} {r['params_m']:>8.1f}M {r['non_embed_m']:>8.1f}M "
              f"{r['embed_pct']:>6.1f}% {r['tokens_b']:>7.1f}B {r['ratio']:>5.0f}x "
              f"{r['flops']:>9.2e} {r['hours']:>7.1f} {r['weeks']:>5.1f}")
    print("-" * 84)
    print(f"{'합계':<12} {'':>9} {'':>9} {'':>7} {'':>8} {'':>6} {'':>9} "
          f"{total_hours:>7.1f} {total_hours/TPU_HOURS_PER_WEEK:>5.1f}")

    buffer = TOTAL_TPU_HOURS - total_hours
    verdict = "OK" if buffer > 0 else "초과"
    print(f"\n쿼터 {TOTAL_TPU_HOURS:.0f}h 중 {total_hours:.0f}h 사용 "
          f"→ 버퍼 {buffer:.0f}h ({buffer/TOTAL_TPU_HOURS:.0%})  [{verdict}]")
    if buffer > 0 and buffer / TOTAL_TPU_HOURS < 0.15:
        print("  경고: 버퍼가 15% 미만이다. 실패/재시작을 흡수할 여유가 없다.")

    print("\n배포 (M1 16GB)")
    print(f"{'티어':<12} {'f16':>8} {'Q8_0':>8} {'Q4_K_M':>8} "
          f"{'KV/토큰':>9} {'KV@8K':>9} {'체크포인트':>10}")
    print("-" * 72)
    for r in rows:
        print(f"{r['name']:<12} {r['gguf_f16_gb']:>7.2f}G {r['gguf_q8_gb']:>7.2f}G "
              f"{r['gguf_q4_gb']:>7.2f}G {r['kv_kb']:>8.1f}K {r['kv_8k_mb']:>8.0f}M "
              f"{r['ckpt_gb']:>9.1f}G")
    print("\n체크포인트는 Kaggle /kaggle/working 20GB 한도와 비교할 값 "
          "(AdamW fp32 상태 + bf16 사본 기준)")


if __name__ == "__main__":
    main()
