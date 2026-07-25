#!/usr/bin/env python3
"""티어 설정의 파라미터 수 / 계산량 / Kaggle TPU 쿼터 소요를 계산한다.

설정을 바꿀 때마다 이걸 돌려서 쿼터 안에 들어오는지 확인한다.

    python3 scripts/budget.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Kaggle 무료 티어에서 실제로 고를 수 있는 가속기 (2026-07 확인: TPU 옵션 없음)
DEVICES = {
    "t4x2": {
        "label": "GPU T4 x2",
        "peak_tflops": 130.0,   # fp16 텐서코어 65 x 2대
        "mfu": 0.23,            # PCIe 기울기 동기화 + bf16 미지원 페널티 반영
        "hours_per_week": 30.0,
        "bf16": False,
    },
    "p100": {
        "label": "GPU P100",
        "peak_tflops": 21.2,    # fp16 (텐서코어 없음)
        "mfu": 0.40,
        "hours_per_week": 30.0,
        "bf16": False,
    },
    "v5e1": {
        "label": "TPU v5e-1 (Colab)",
        "peak_tflops": 197.0,   # bf16, 칩 1개 · HBM 16GB
        "mfu": 0.35,
        "hours_per_week": 20.0,  # Colab 무료는 고정 쿼터가 없다. 보수적 추정
        "bf16": True,
        "_note": "무료 Colab은 백그라운드 실행이 안 된다 — 브라우저를 열어둬야 한다. "
                 "사용량 제한이 유동적이라 주당 시간을 보장할 수 없다.",
    },
    "tpuv3": {
        "label": "TPU v3-8",
        "peak_tflops": 420.0,
        "mfu": 0.35,
        "hours_per_week": 20.0,
        "bf16": True,
        "_note": "Kaggle에서는 현재 선택 불가. TRC 승인 시 사용.",
    },
}
DEFAULT_DEVICE = "v5e1"
WEEKS = 8


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
    key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEVICE
    if key not in DEVICES:
        sys.exit(f"가속기는 {sorted(DEVICES)} 중 하나여야 한다 (기본 {DEFAULT_DEVICE})")
    dev = DEVICES[key]
    eff_flops = dev["peak_tflops"] * 1e12 * dev["mfu"]
    flops_per_hour = eff_flops * 3600
    hours_per_week = dev["hours_per_week"]
    total_quota = hours_per_week * WEEKS

    tiers = sorted(ROOT.glob("configs/tier*.json"))
    if not tiers:
        sys.exit("configs/tier*.json 를 찾을 수 없다")

    rows, total_hours = [], 0.0
    for path in tiers:
        c = json.loads(path.read_text())
        p = count_params(c)
        D = c["train_tokens"]
        flops = 6 * p["total"] * D
        hours = flops / flops_per_hour
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
            "weeks": hours / hours_per_week,
            "kv_kb": kv / 1024,
            "kv_8k_mb": kv * 8192 / 1024**2,
            "ckpt_gb": ckpt_gb,
            "gguf_f16_gb": p["total"] * 2 / 1024**3,
            "gguf_q8_gb": p["total"] * 8.5 / 8 / 1024**3,
            "gguf_q4_gb": p["total"] * 4.83 / 8 / 1024**3,
        })

    print(f"\n{dev['label']}: {dev['peak_tflops']:.0f} TFLOPS peak x MFU {dev['mfu']:.0%} "
          f"= {eff_flops/1e12:.0f} TFLOPS 실효"
          f"{'' if dev['bf16'] else '  (bf16 미지원 → fp16 + loss scaling 필요)'}")
    print(f"쿼터: {hours_per_week:.0f} h/주 x {WEEKS}주 = {total_quota:.0f} 시간 (이월 불가)\n")

    print(f"{'티어':<12} {'전체':>9} {'비임베딩':>9} {'임베딩%':>7} "
          f"{'토큰':>8} {'배수':>6} {'FLOPs':>9} {'시간':>7} {'주':>5}")
    print("-" * 84)
    for r in rows:
        print(f"{r['name']:<12} {r['params_m']:>8.1f}M {r['non_embed_m']:>8.1f}M "
              f"{r['embed_pct']:>6.1f}% {r['tokens_b']:>7.1f}B {r['ratio']:>5.0f}x "
              f"{r['flops']:>9.2e} {r['hours']:>7.1f} {r['weeks']:>5.1f}")
    print("-" * 84)
    print(f"{'합계':<12} {'':>9} {'':>9} {'':>7} {'':>8} {'':>6} {'':>9} "
          f"{total_hours:>7.1f} {total_hours/hours_per_week:>5.1f}")

    buffer = total_quota - total_hours
    verdict = "OK" if buffer > 0 else "초과"
    print(f"\n쿼터 {total_quota:.0f}h 중 {total_hours:.0f}h 사용 "
          f"→ 버퍼 {buffer:.0f}h ({buffer/total_quota:.0%})  [{verdict}]")
    print(f"벽시계 최소 {total_hours/hours_per_week:.1f}주 (주 {hours_per_week:.0f}h 상한)")
    if buffer > 0 and buffer / total_quota < 0.15:
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
