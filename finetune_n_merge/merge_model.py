from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "./lora-output"
OUTPUT_DIR = "./merged-model"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_PATH)

# Merge LoRA into base model
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(OUTPUT_DIR)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Model merged and saved.")