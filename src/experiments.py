# ============================================================
#  experiments.py — 전체 실험 실행 + 결과 시각화
#
#  ✅ 직접 실험 완료:
#     Whisper Baseline     17.94%
#     Untrained+Select     18.01%   CERR -0.41%
#     Untrained+Rewrite    26.90%   CERR -49.97%
#     Trained+Select       19.25%   CERR -7.28%
#
#  📄 논문 결과 대체 (런타임/환경 문제):
#     Trained+Rewrite      7.91%    CERR +24.2%
#     Trained+Select(opt)  8.90%    CERR +13.9%
#     Trained+Rewrite(opt) 7.35%    CERR +28.9%
# ============================================================

import os, json, torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from utils import (
    compute_cer, compute_cerr,
    rerank_by_ensemble,
    sllm_select, sllm_rewrite,
    BASELINE_CER,
)

# ── 한글 폰트 (코랩) ──────────────────────────────────────
plt.rcParams["font.family"]        = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 설정 ──────────────────────────────────────────────
EVAL_DATA_PATH  = "/content/eval_data.json"
BASE_MODEL_PATH = "davidkim205/komt-llama3.2-3b-blossom"
LORA_MODEL_PATH = "/content/drive/MyDrive/lora_final"
RESULTS_DIR     = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 실험 결과 하드코딩 (논문 기준 대체값) ──────────────────
# 직접 실험한 결과는 run_*_experiment() 에서 실제 계산
PAPER_SUBSTITUTE = {
    "Trained + Rewrite":              {"cer": 7.91,  "cerr": +24.2},
    "Trained + Select (N-best opt)":  {"cer": 8.90,  "cerr": +13.9},
    "Trained + Rewrite (N-best opt)": {"cer": 7.35,  "cerr": +28.9},
}


# ── 데이터 로드 ────────────────────────────────────────────

def load_eval_data() -> list:
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"평가 데이터: {len(data)}문장")
    return data


# ── 실험 러너 ──────────────────────────────────────────────

def run_experiment(eval_data: list, model, tokenizer,
                   mode: str, use_opt: bool, name: str) -> dict:
    """
    mode: "select" | "rewrite"
    use_opt: True → logprob + PPL 앙상블 재정렬
    """
    preds = []
    refs  = [d["reference"] for d in eval_data]

    for d in tqdm(eval_data, desc=name):
        nbest    = d["whisper_nbest"][:5]
        logprobs = d.get("logprobs", [0.0] * len(nbest))

        if use_opt and len(nbest) > 1:
            nbest = rerank_by_ensemble(nbest, logprobs, model, tokenizer)

        if mode == "select":
            pred = sllm_select(nbest, model, tokenizer)
        else:
            pred = sllm_rewrite(nbest, model, tokenizer)

        preds.append(pred)

    avg_cer = compute_cer(preds, refs)
    cerr    = compute_cerr(BASELINE_CER, avg_cer)
    print(f"  [{name}]  CER: {avg_cer:.2f}%  CERR: {cerr:+.2f}%")
    return {"name": name, "cer": round(avg_cer, 2), "cerr": round(cerr, 2),
            "source": "실험"}


# ── STEP 2: 미학습 실험 ────────────────────────────────────

def run_untrained_experiments(eval_data: list) -> list:
    print("\n" + "="*50)
    print("  STEP 2: 미학습 sLLM 실험")
    print("="*50)
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"미학습 모델 로드 완료 ({torch.cuda.get_device_name(0)})")

    results = [
        run_experiment(eval_data, model, tokenizer,
                       mode="select", use_opt=False,
                       name="Untrained + Select"),
        run_experiment(eval_data, model, tokenizer,
                       mode="rewrite", use_opt=False,
                       name="Untrained + Rewrite"),
    ]

    del model
    torch.cuda.empty_cache()
    return results


# ── STEP 3: 학습된 모델 실험 ──────────────────────────────

def run_trained_experiments(eval_data: list) -> list:
    print("\n" + "="*50)
    print("  STEP 3: 학습된 sLLM 실험")
    print("="*50)
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, LORA_MODEL_PATH)
    model.eval()
    print(f"학습된 모델 로드 완료 ({torch.cuda.get_device_name(0)})")

    results = [
        run_experiment(eval_data, model, tokenizer,
                       mode="select", use_opt=False,
                       name="Trained + Select"),
        # Trained+Rewrite: T4 메모리 한계로 런타임 중단
        # → 논문 결과로 대체
    ]

    del model, base
    torch.cuda.empty_cache()
    return results


# ── 전체 결과 통합 ─────────────────────────────────────────

def build_all_results(untrained: list, trained: list) -> pd.DataFrame:
    baseline = [{"name": "Whisper Medium (Baseline)",
                 "cer": BASELINE_CER, "cerr": 0.0, "source": "실험"}]

    paper = [
        {"name": k, "cer": v["cer"], "cerr": v["cerr"], "source": "논문"}
        for k, v in PAPER_SUBSTITUTE.items()
    ]

    # 순서: Baseline → Untrained → Trained(실험) → 나머지(논문)
    all_rows = baseline + untrained + trained + paper
    return pd.DataFrame(all_rows)


# ── 시각화 ────────────────────────────────────────────────

def plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    def color(row):
        if "Baseline" in row["name"]:  return "#888888"
        if "Untrained" in row["name"]: return "#FF8888"
        if "opt"       in row["name"]: return "#1144AA"
        if row["source"] == "논문":    return "#AABBDD"
        return "#4488CC"

    colors = [color(r) for _, r in df.iterrows()]
    x = range(len(df))

    # CER
    ax = axes[0]
    bars = ax.bar(x, df["cer"], color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel("CER (%)")
    ax.set_title("CER 비교")
    for b, v, s in zip(bars, df["cer"], df["source"]):
        label = f"{v:.2f}%" + (" *" if s == "논문" else "")
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                label, ha="center", va="bottom", fontsize=8)

    # CERR
    ax = axes[1]
    cerr_colors = ["#CC3333" if v < 0 else "#2277BB" for v in df["cerr"]]
    bars2 = ax.bar(x, df["cerr"], color=cerr_colors, edgecolor="white", width=0.6)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=38, ha="right", fontsize=8.5)
    ax.set_ylabel("CERR (%)")
    ax.set_title("CERR 비교 (↑ 개선)")
    for b, v, s in zip(bars2, df["cerr"], df["source"]):
        label = f"{v:+.1f}%" + (" *" if s == "논문" else "")
        offset = 0.5 if v >= 0 else -2.0
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + offset,
                label, ha="center", va="bottom", fontsize=8)

    fig.text(0.5, 0.01, "* 논문 결과 참조 (런타임/환경 문제로 직접 실험 미완료)",
             ha="center", fontsize=9, color="gray")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(RESULTS_DIR, "experiment_results.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


def print_table(df: pd.DataFrame):
    print("\n" + "="*62)
    print("  최종 실험 결과")
    print("="*62)
    print(f"{'Method':<38} {'CER':>7} {'CERR':>8}  {'출처':>4}")
    print("-"*62)
    for _, row in df.iterrows():
        star = " ⭐" if row["cerr"] == df["cerr"].max() else ""
        print(f"{row['name']:<38} {row['cer']:>6.2f}%  "
              f"{row['cerr']:>+7.1f}%  {row['source']:>4}{star}")
    print("="*62)
    print("⭐ 최고 성능  (* 논문 결과 참조)")


# ── 메인 ───────────────────────────────────────────────────

if __name__ == "__main__":
    eval_data = load_eval_data()

    untrained_results = run_untrained_experiments(eval_data)
    trained_results   = run_trained_experiments(eval_data)

    df = build_all_results(untrained_results, trained_results)

    print_table(df)
    plot_results(df)

    df.to_csv(os.path.join(RESULTS_DIR, "experiment_results.csv"),
              index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 완료: {RESULTS_DIR}/")
