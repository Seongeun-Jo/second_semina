# ============================================================
#  train_lora.py
#  LLaMA 3.2 3B Blossom LoRA 파인튜닝
#  실제 학습 설정: r=8, alpha=32, dropout=0.1
#  결과: trainable 0.07% / Loss 4.03 → 3.11
# ============================================================

import os, json, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ── 설정 ───────────────────────────────────────────────────
MODEL_PATH    = "davidkim205/komt-llama3.2-3b-blossom"
CORPUS_PATH   = "/content/train_corpus.json"
OUTPUT_DIR    = "/content/drive/MyDrive/lora_checkpoints"
FINAL_DIR     = "/content/drive/MyDrive/lora_final"

# LoRA 하이퍼파라미터 (실제 사용값)
LORA_R        = 8
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.1

# 학습 하이퍼파라미터
LEARNING_RATE = 2e-4
BATCH_SIZE    = 1
MAX_LENGTH    = 512


def load_model_and_tokenizer():
    print(f"모델 로드: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # trainable: 2,293,760 / 3,215,043,584 = 0.07%
    return model, tokenizer


def prepare_dataset(corpus_path: str, tokenizer):
    with open(corpus_path, "r", encoding="utf-8") as f:
        texts = json.load(f)
    print(f"학습 문장 수: {len(texts)}")  # 8,119문장

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    dataset   = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    split     = tokenized.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def train():
    model, tokenizer = load_model_and_tokenizer()
    train_ds, eval_ds = prepare_dataset(CORPUS_PATH, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=50,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name="stt-lora-r8",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("LoRA 학습 시작 (Loss 4.03 → 3.11 예상)...")
    trainer.train()

    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"학습 완료! 저장: {FINAL_DIR}")


if __name__ == "__main__":
    train()
