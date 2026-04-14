# 🎙️ STT 후처리 보정 시스템
**Whisper + sLLM 기반 금융 상담 음성인식 오류 보정**

> LLaMA 3.2 3B (LoRA 파인튜닝) 을 활용한 N-best 후처리 보정 파이프라인

---

## 📊 실험 결과

| Method | CER | CERR |
|---|:---:|:---:|
| Whisper Medium (Baseline) | 17.94% | 기준 |
| Untrained + Select | 18.01% | -0.41% |
| Untrained + Rewrite | 26.90% | -49.97% |
| Trained + Select | 19.25% | -7.28% |
| Trained + Rewrite | 7.91% | +24.2% |
| Trained + Select (N-best opt) | 8.90% | +13.9% |
| **Trained + Rewrite (N-best opt)** ⭐ | **7.35%** | **+28.9%** |

> ⭐ 최적 파이프라인: Whisper → N-best opt → Trained sLLM Rewrite

---

## 🔍 실험 분석

**STEP 1. 기준 확립**
- Whisper Medium Top-1 CER: 17.94% → 보정 필요

**STEP 2. 미학습 모델 실험**
- Untrained+Select: 의미 판단 능력 부족 → Top-1과 거의 동일 선택 (-0.41%)
- Untrained+Rewrite: 환각(hallucination) 심화 → 오류 대폭 증가 (-49.97%)
- **결론**: 도메인 학습 없이는 보정 불가

**STEP 3. 학습된 모델 실험**
- LoRA 파인튜닝 후 Rewrite 방식에서 최대 +24.2% 성능 향상
- **결론**: 학습이 핵심 → Rewrite > Select

**STEP 4. N-best 후보 최적화**
- logprob + Perplexity 앙상블 재정렬 적용
- Trained+Rewrite(opt): CER 7.35%, CERR +28.9% 달성
- **결론**: 후보 품질 개선이 보정 성능에 직접적 영향

---

## ⚙️ 실험 환경

| 항목 | 값 |
|---|---|
| GPU | Tesla T4 (15GB) |
| Whisper | Medium (20240930) |
| sLLM | LLaMA 3.2 3B Korean Blossom |
| 학습 | LoRA (r=8, alpha=32, dropout=0.1) |
| 학습 파라미터 | 0.07% (2.29M / 3.21B) |
| 학습 데이터 | 금융 상담 8,119문장 (231세션) |
| 평가 데이터 | 금융 상담 834문장 (25세션) |
| Loss | 4.03 → 3.11 (수렴) |

---

## 📌 데이터

- 출처: [AI Hub](https://aihub.or.kr) 상담 음성 (D61), 2020
- AI Hub 이용약관 준수 필요 → 데이터 직접 업로드 불가

---

## 📖 참고

- [Whisper](https://github.com/openai/whisper)
- [LLaMA 3.2 Korean Blossom](https://huggingface.co/davidkim205/komt-llama3.2-3b-blossom)
- [LoRA](https://arxiv.org/abs/2106.09685)
