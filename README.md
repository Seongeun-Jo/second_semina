# 🎙️ STT 후처리 보정 시스템
**Whisper + sLLM 기반 금융 상담 음성인식 오류 보정**

> LLaMA 3.2 3B (LoRA 파인튜닝) 을 활용한 N-best 후처리 보정 파이프라인

---

## 📊 실험 결과

### ✅ 직접 실험 완료

| Method | CER | CERR |
|---|:---:|:---:|
| Whisper Medium (Baseline) | 17.94% | 기준 |
| Untrained + Select | 18.01% | -0.41% |
| Untrained + Rewrite | 26.90% | -49.97% |
| Trained + Select | 19.25% | -7.28% |

### 📄 논문 결과 참조 (런타임/환경 문제로 미완료)

| Method | CER | CERR | 비고 |
|---|:---:|:---:|:---:|
| Trained + Rewrite | 7.91% | +24.2% | 런타임 중단 |
| Trained + Select (N-best opt) | 8.90% | +13.9% | 미실시 |
| **Trained + Rewrite (N-best opt)** ⭐ | **7.35%** | **+28.9%** | 미실시 |

> ⭐ 논문 기준 최적 파이프라인: `Whisper → N-best opt → Trained sLLM Rewrite`

---

## 🔍 실험 분석

**Untrained 실험 (STEP 2)**
- Untrained+Select: 의미 판단 능력 부족 → Top-1과 거의 동일 선택
- Untrained+Rewrite: 환각(hallucination) 심화 → 오류 대폭 증가
- **결론**: 도메인 학습 없이는 보정 불가

**Trained 실험 (STEP 3)**
- Trained+Select: 예상보다 성능 저하 → 후보군 품질 문제로 추정
- Trained+Rewrite: T4 15GB 환경 제약으로 런타임 중단

---

## 🏗️ 시스템 구조

```
음성 입력
    ↓
Whisper Medium (beam=5, temperature=0.0)
    ↓
N-best 후보 생성 (logprob 기준)
    ↓
[opt] logprob + Perplexity 앙상블 재정렬
    ↓
학습된 sLLM 3B → Select 또는 Rewrite 보정
    ↓
최종 텍스트 출력
```

---

## 📁 프로젝트 구조

```
stt-correction/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── utils.py                # CER/CERR/PPL/프롬프트 유틸
│   ├── data_preprocessing.py   # AI Hub D61 전처리 + Whisper 전사
│   ├── train_lora.py           # LoRA 학습
│   └── experiments.py          # 실험 실행 + 시각화
├── results/
│   └── experiment_results.csv
└── notebooks/
    └── full_experiment.ipynb
```

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

## 🚀 실행 방법

```bash
pip install -r requirements.txt
python src/data_preprocessing.py   # 전처리
python src/train_lora.py           # 학습
python src/experiments.py          # 실험
```

---

## 📌 데이터

- 출처: [AI Hub](https://aihub.or.kr) — 상담 음성 (D61), 2020
- ⚠️ AI Hub 이용약관 준수 필요 → 데이터 직접 업로드 불가

---

## 📖 참고

- [Whisper](https://github.com/openai/whisper)
- [LLaMA 3.2 Korean Blossom](https://huggingface.co/davidkim205/komt-llama3.2-3b-blossom)
- [LoRA](https://arxiv.org/abs/2106.09685)
