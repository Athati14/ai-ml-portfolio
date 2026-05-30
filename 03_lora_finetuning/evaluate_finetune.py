"""Compare base vs fine-tuned model on a held-out set with a simple win-rate."""
from __future__ import annotations
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "mistralai/Mistral-7B-v0.1"
ADAPTER = "./qlora-adapter"


def generate(model, tok, prompt, max_new=200):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def load_pair():
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto",
                                                torch_dtype=torch.bfloat16)
    tuned = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE, device_map="auto",
                                             torch_dtype=torch.bfloat16),
        ADAPTER)
    return tok, base, tuned


def main():
    tok, base, tuned = load_pair()
    with open("eval.jsonl", encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    results = []
    for r in rows:
        prompt = f"### Instruction:\n{r['instruction']}\n\n### Response:\n"
        results.append({"instruction": r["instruction"],
                        "base": generate(base, tok, prompt),
                        "tuned": generate(tuned, tok, prompt),
                        "reference": r.get("output", "")})
    with open("comparison.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {len(results)} comparisons to comparison.json")


if __name__ == "__main__":
    main()
