import os
import sys
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Paths

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, "..")
TRAIN_DATA  = os.path.join(ROOT_DIR, "data", "processed", "trump_train.txt")
OUTPUT_DIR  = os.path.join(ROOT_DIR, "models", "trump")

# Training configuration
EPOCHS      = 8
BATCH_SIZE  = 8
GRAD_ACCUM  = 2
BLOCK_SIZE  = 512
LEARN_RATE  = 5e-5
RESUME      = False
USE_FP16    = True
BASE_MODEL  = "gpt2-medium"

# Build the dataset.

def build_dataset(tokenizer, file_path: str, block_size: int):
    print(f"  Loading training file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    examples = [e.strip() for e in raw_text.split("\n\n") if e.strip()]
    print(f"  Number of raw examples: {len(examples):,}")

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_dict({"text": examples})
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )
    dataset.set_format(type="torch")
    return dataset

# Main training function

def train():
    print("=" * 60)
    print("  Trump GPT-2 Medium Fine-Tuning")
    print("=" * 60)

    if not os.path.isfile(TRAIN_DATA):
        print(f"[ERROR] Training data not found at: {TRAIN_DATA}")
        print("  Run  python scripts/preprocess.py  first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n  Loading tokeniser from  {BASE_MODEL}  …")
    tokenizer = GPT2TokenizerFast.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {
        "additional_special_tokens": [
            "<|startoftext|>", "<|endoftext|>",
            "[BIDEN]:", "[TRUMP]:",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"  Loading model  {BASE_MODEL}  …")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    dataset = build_dataset(tokenizer, TRAIN_DATA, BLOCK_SIZE)
    split   = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train examples: {len(train_ds):,}  |  Eval examples: {len(eval_ds):,}")

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LEARN_RATE,
        weight_decay                = 0.01,
        warmup_steps                = 16,
        lr_scheduler_type           = "cosine",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        fp16                        = USE_FP16 and torch.cuda.is_available(),
        dataloader_pin_memory       = torch.cuda.is_available(),
        logging_steps               = 5,
        report_to                   = "none",
        seed                        = 42,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        data_collator = data_collator,
    )

    resume_ckpt = OUTPUT_DIR if RESUME and os.path.isdir(OUTPUT_DIR) else None
    print("\n  Starting training …\n")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"\n  Saving fine-tuned Trump model to  {OUTPUT_DIR}  …")
    trainer.save_model(OUTPUT_DIR)
    print("  Done.")
    print("=" * 60)

if __name__ == "__main__":
    train()
