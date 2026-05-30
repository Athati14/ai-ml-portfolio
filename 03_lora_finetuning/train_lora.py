"""QLoRA fine-tuning script: 4-bit base model + LoRA adapters via TRL's SFTTrainer."""
from __future__ import annotations
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./qlora-adapter"


def load_quantized_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token
    return model, tok


def format_example(row):
    return (f"### Instruction:\n{row['instruction']}\n\n"
            f"### Response:\n{row['output']}")


def main():
    model, tok = load_quantized_model()
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    ds = load_dataset("json", data_files="train.jsonl", split="train")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4,
        num_train_epochs=3, bf16=True, logging_steps=10,
        save_strategy="epoch", report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds,
        peft_config=lora, tokenizer=tok,
        formatting_func=lambda r: [format_example(x) for x in r] if isinstance(r, list)
                                   else format_example(r),
        max_seq_length=1024,
    )
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
