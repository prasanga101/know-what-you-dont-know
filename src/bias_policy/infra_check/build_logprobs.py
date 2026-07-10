import os
import json
from pprint import pprint

INPUT_PATH = "data/infra_check/ollama_logprobs.json"
OUTPUT_PATH = "data/infra_check/resample_ollama_logprobs.json"

LOGPROB_CONTROL_TEMPLATE = """{prompt}

IMPORTANT:
Return exactly one uppercase letter from A, B, C, or D.
Do not explain.

Answer:"""

with open(INPUT_PATH, encoding="utf-8") as f:
    logprob_rows = json.load(f)

pprint(logprob_rows[:1])

resample_ollama_logprobs = []

seen = set()

for item in logprob_rows:
    # keep only one prompt per item, not 15 duplicate resamples
    key = item["item_id"]

    if key in seen:
        continue

    seen.add(key)

    resample_ollama_logprobs.append({
        "item_id": item["item_id"],
        "link": item["link"],
        "question_number": item["question_number"],
        "language": item["language"],

        "classification": item["classification"],
        "audit_label": item["audit_label"],

        "correct_answer_num": item["correct_answer_num"],
        "correct_answer_letter": item["correct_answer_letter"],
        "hint_en": item["hint_en"],

        "original_prompt": item["prompt"],
        "logprob_prompt": LOGPROB_CONTROL_TEMPLATE.format(
            prompt=item["prompt"].strip()
        ),
    })

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(resample_ollama_logprobs, f, indent=2, ensure_ascii=False)
print(f"saved {len(resample_ollama_logprobs)} unique logprob prompts to {OUTPUT_PATH}")
assert len(resample_ollama_logprobs) == 12, "expected 12 unique confounded items"
print(f"saved {len(resample_ollama_logprobs)} unique logprob prompts to {OUTPUT_PATH}")