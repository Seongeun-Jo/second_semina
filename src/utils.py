# ============================================================
#  utils.py — 공통 유틸리티
# ============================================================

import re
import torch
import numpy as np
from jiwer import cer as jiwer_cer


# ────────────────────────────────
#  텍스트 전처리
# ────────────────────────────────

def clean_text(text: str) -> str:
    """
    AI Hub 상담 음성 레이블 정제
    - 화자 태그 제거  예) "n/ 연개비..." → "연개비..."
    - 개인정보 마스킹('@') 제거
    - 구두점 제거
    - 공백 정규화
    """
    text = re.sub(r'^[a-zA-Z]+/\s*', '', text)   # 화자 태그
    text = text.replace('@', '')                   # 개인정보 마스킹
    text = re.sub(r'[.,!?~\-]', '', text)          # 구두점
    text = re.sub(r'\s+', ' ', text)               # 공백 정규화
    return text.strip()


# ────────────────────────────────
#  CER / CERR
# ────────────────────────────────

BASELINE_CER = 17.94  # Whisper Medium Top-1 실측값

def compute_cer(predictions: list, references: list) -> float:
    """전체 평균 CER (%)"""
    total = sum(jiwer_cer(ref, pred) for pred, ref in zip(predictions, references))
    return total / len(predictions) * 100


def compute_cerr(baseline_cer: float, method_cer: float) -> float:
    """CERR (%) — 양수면 개선, 음수면 악화"""
    return (baseline_cer - method_cer) / baseline_cer * 100


# ────────────────────────────────
#  Perplexity (N-best opt용)
# ────────────────────────────────

def compute_perplexity(text: str, model, tokenizer, device: str = "cuda") -> float:
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


def rerank_by_ensemble(candidates: list, logprobs: list,
                       model, tokenizer, alpha: float = 0.5) -> list:
    """
    logprob + Perplexity 앙상블 재정렬
    Score = alpha * logprob_norm + (1 - alpha) * (1 - PPL_norm)
    """
    ppls   = [compute_perplexity(c, model, tokenizer) for c in candidates]
    lp_arr = np.array(logprobs, dtype=float)
    pp_arr = np.array(ppls,     dtype=float)

    def norm(arr):
        rng = arr.max() - arr.min()
        return (arr - arr.min()) / (rng + 1e-9)

    scores = alpha * norm(lp_arr) + (1 - alpha) * (1 - norm(pp_arr))
    return [candidates[i] for i in np.argsort(-scores)]


# ────────────────────────────────
#  보정 함수 (실제 실험에서 사용한 코드 그대로)
# ────────────────────────────────

def sllm_select(candidates: list, model, tokenizer) -> str:
    """
    N-best 후보 중 가장 자연스러운 문장 번호 선택
    """
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt = f"""다음은 한국어 금융 상담 음성 인식 후보 문장들입니다.
가장 자연스럽고 의미가 정확한 문장의 번호만 답하세요.

{numbered}

번호:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    for c in result:
        if c.isdigit() and 1 <= int(c) <= len(candidates):
            return candidates[int(c) - 1]
    return candidates[0]  # fallback


def sllm_rewrite(candidates: list, model, tokenizer) -> str:
    """
    N-best 후보를 참조해 새 문장 생성
    """
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    prompt = f"""다음은 한국어 금융 상담 음성 인식 후보 문장들입니다.
후보들을 참고하여 가장 자연스럽고 정확한 문장을 한 줄로 작성하세요.

{numbered}

보정된 문장:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return result.split("\n")[0].strip() if result else candidates[0]
