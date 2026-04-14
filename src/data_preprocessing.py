# ============================================================
#  data_preprocessing.py
#  AI Hub 상담 음성 (D61) 전처리 + Whisper 전사
# ============================================================

import os, json, zipfile, glob, torch, whisper
from tqdm import tqdm
from utils import clean_text

# ── 경로 설정 (코랩 기준) ──────────────────────────────────
BASE_PATH      = "/content/drive/MyDrive/상담음성데이터금융"  # ← 수정
TRAINING_DIR   = os.path.join(BASE_PATH, "Training")
VALIDATION_DIR = os.path.join(BASE_PATH, "Validation")

EXTRACT_DIR  = "/content/data_extracted"
LABEL_DIR    = os.path.join(EXTRACT_DIR, "labels")
WAV_DIR      = os.path.join(EXTRACT_DIR, "wavs")
EVAL_OUTPUT  = "/content/eval_data.json"
TRAIN_OUTPUT = "/content/train_corpus.json"

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(WAV_DIR,   exist_ok=True)


# ── ZIP 압축 해제 ───────────────────────────────────────────

def extract_zip(zip_path: str, dest: str):
    print(f"압축 해제: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"  → {dest}")


def extract_all():
    extract_zip(
        os.path.join(VALIDATION_DIR, "label_valid_D61.zip"),
        os.path.join(LABEL_DIR, "valid"),
    )
    extract_zip(
        os.path.join(VALIDATION_DIR, "wav_valid_D61.zip"),
        WAV_DIR,
    )
    for fname in ["label_train_D61_0.zip", "label_train_D61_1.zip"]:
        fpath = os.path.join(TRAINING_DIR, fname)
        if os.path.exists(fpath):
            extract_zip(fpath, os.path.join(LABEL_DIR, "train"))


# ── 레이블 파싱 ────────────────────────────────────────────

def parse_label_file(json_path: str) -> dict:
    """AI Hub JSON 레이블 파싱 (다양한 포맷 대응)"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wav_name = reference = None

    for key in ["fileName", "file_name", "FileName", "파일명"]:
        if key in data:
            wav_name = data[key]
            break

    if "발화정보" in data:
        for key in ["LabelText", "ReadText", "labelText", "text"]:
            val = data["발화정보"].get(key, "")
            if val:
                reference = clean_text(val)
                break
    elif "transcript" in data:
        reference = clean_text(data["transcript"])
    elif "text" in data:
        reference = clean_text(data["text"])

    return {"wav_name": wav_name, "reference": reference}


def load_all_labels(label_dir: str) -> list:
    json_files = glob.glob(os.path.join(label_dir, "**", "*.json"), recursive=True)
    print(f"레이블 파일: {len(json_files)}개")
    labels = [parse_label_file(jf) for jf in json_files]
    labels = [l for l in labels if l["wav_name"] and l["reference"]]
    print(f"유효 레이블: {len(labels)}개")
    return labels


# ── Whisper 전사 ────────────────────────────────────────────

def transcribe_top1(wav_path: str, model) -> dict:
    """
    Whisper Top-1 전사 + logprob 반환
    beam_size=5 로 탐색하되 Top-1만 사용 (N-best 미지원 시)
    """
    result = model.transcribe(
        wav_path,
        language="ko",
        beam_size=5,
        temperature=0.0,
    )
    top1   = clean_text(result["text"])
    avg_lp = result["segments"][0]["avg_logprob"] if result["segments"] else 0.0
    return {"nbest": [top1], "logprobs": [avg_lp]}


def build_eval_data(labels: list, wav_dir: str,
                    whisper_model, max_samples: int = 834) -> list:
    wav_index = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)
    }
    print(f"WAV 파일: {len(wav_index)}개")

    eval_data, not_found = [], 0
    for label in tqdm(labels[:max_samples], desc="Whisper 전사"):
        wav_name = label["wav_name"]
        if not wav_name.endswith(".wav"):
            wav_name += ".wav"
        if wav_name not in wav_index:
            not_found += 1
            continue

        res = transcribe_top1(wav_index[wav_name], whisper_model)
        eval_data.append({
            "wav_name":      wav_name,
            "whisper_nbest": res["nbest"],
            "logprobs":      res["logprobs"],
            "reference":     label["reference"],
        })

    print(f"완료: {len(eval_data)}개 | WAV 미발견: {not_found}개")
    return eval_data


# ── LoRA 학습용 코퍼스 ─────────────────────────────────────

def build_train_corpus(label_dir: str, output_path: str) -> list:
    """Training 레이블에서 텍스트 추출 → 8119문장"""
    json_files = glob.glob(
        os.path.join(label_dir, "train", "**", "*.json"), recursive=True
    )
    texts = []
    for jf in json_files:
        parsed = parse_label_file(jf)
        if parsed["reference"]:
            texts.append(parsed["reference"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"학습 코퍼스 저장: {output_path} ({len(texts)}문장)")
    return texts


# ── 메인 ───────────────────────────────────────────────────

if __name__ == "__main__":
    extract_all()

    whisper_model = whisper.load_model("medium")
    print(f"Whisper 로드 완료 ({torch.cuda.get_device_name(0)})")

    # 평가셋 구축 (834문장)
    valid_labels = load_all_labels(os.path.join(LABEL_DIR, "valid"))
    eval_data    = build_eval_data(valid_labels, WAV_DIR, whisper_model)

    with open(EVAL_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"eval_data 저장: {EVAL_OUTPUT} ({len(eval_data)}문장)")

    # 학습 코퍼스 구축 (8119문장)
    build_train_corpus(LABEL_DIR, TRAIN_OUTPUT)

    # Baseline CER 확인
    from utils import compute_cer
    preds = [d["whisper_nbest"][0] for d in eval_data]
    refs  = [d["reference"]        for d in eval_data]
    print(f"\n[STEP 1] Whisper Baseline CER: {compute_cer(preds, refs):.2f}%")
