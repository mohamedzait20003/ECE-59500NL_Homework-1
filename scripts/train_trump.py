import os
import sys
import argparse

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, "..")
TRAIN_DATA  = os.path.join(ROOT_DIR, "data", "processed", "trump_train.txt")
DEFAULT_OUT = os.path.join(ROOT_DIR, "models", "trump")
BASE_MODEL  = "gpt2-medium"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 Medium on Trump speech data."
    )
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--grad_accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--block_size",  type=int,   default=512)
    parser.add_argument("--output_dir",  type=str,   default=DEFAULT_OUT)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--fp16",        action="store_true")
    return parser.parse_args()

def build_dataset(tokenizer, file_path: str, block_size: int):
    from datasets import Dataset

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

def train(args):
    import torch
    from transformers import (
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    print("=" * 60)
    print("  Trump GPT-2 Medium Fine-Tuning")
    print("=" * 60)

    if not os.path.isfile(TRAIN_DATA):
        print(f"[ERROR] Training data not found at: {TRAIN_DATA}")
        print("  Run  python scripts/preprocess.py  first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

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
    tokenizer.save_pretrained(args.output_dir)

    print(f"  Loading model  {BASE_MODEL}  …")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    dataset = build_dataset(tokenizer, TRAIN_DATA, args.block_size)
    split   = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train examples: {len(train_ds):,}  |  Eval examples: {len(eval_ds):,}")

    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        overwrite_output_dir        = True,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        weight_decay                = 0.01,
        warmup_ratio                = 0.06,
        lr_scheduler_type           = "cosine",
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        fp16                        = args.fp16 and torch.cuda.is_available(),
        logging_steps               = 50,
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

    resume_ckpt = args.resume if args.resume else None
    print("\n  Starting training …\n")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print(f"\n  Saving fine-tuned Trump model to  {args.output_dir}  …")
    trainer.save_model(args.output_dir)
    print("  Done.")
    print("=" * 60)

if __name__ == "__main__":
    args = parse_args()
    train(args)
