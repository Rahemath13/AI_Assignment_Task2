# train_finetune.py
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# disable wandb if installed
os.environ.setdefault("WANDB_DISABLED", "true")

DATAFILE = "recipes_dataset.jsonl"
MODEL_NAME = "gpt2"
OUTPUT_DIR = "./model_output"

def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / DATAFILE
    if not data_path.exists():
        raise FileNotFoundError(f"{DATAFILE} not found in project root. Create {DATAFILE} first.")

    print("📦 Loading dataset from", data_path)
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    print("📘 Loading tokenizer and model (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,               # change to 1 for very quick demo
        per_device_train_batch_size=1,    # keep small on CPU
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        fp16=False
    )

    print("⚙️ Loading model and starting Trainer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("🚀 Training...")
    trainer.train()

    print("💾 Saving final model to", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Fine-tuning complete!")

if __name__ == "__main__":
    main()
