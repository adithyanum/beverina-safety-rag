import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# =========================
# CONFIG
# =========================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "./lora-output"

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD BASE MODEL (4-bit)
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# =========================
# LoRA CONFIG
# =========================
lora_config = LoraConfig(
    r=16,                     # rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =========================
# DATASET (REPLACE WITH YOUR DATA)
# =========================
dataset = load_dataset("json", data_files="data/train.json")

def format_prompt(example):
    return {
        "text": f"""<s>[INST] {example['input']} [/INST] {example['output']} </s>"""
    }

dataset = dataset.map(format_prompt)

# =========================
# TRAINING CONFIG
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

# =========================
# TRAINER
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args
)

# =========================
# TRAIN
# =========================
def train():
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ LoRA training complete and saved.")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    train()