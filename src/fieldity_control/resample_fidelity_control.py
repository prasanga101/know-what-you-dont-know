import json
import os
import time

from src.fieldity_control.answer_fidelty_mcq import extract_passage_mcq
from fire_api_call.src import call_ollama

MODEL = "hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M"
# MODEL = "qwen2.5:1.5b-instruct"
# MODEL = "qwen3.5:9b"
TEMPERATURE = 0.7
N_RESAMPLES = 15

INPUT_FILE = "data/fidelity_control1/fidelity_control_prompts.json"
OUTPUT_FILE = "data/fidelity_control1/resampled_fidelity_control_output_qwen_model.json"

with open(INPUT_FILE, encoding="utf-8") as f:
    sample = json.load(f)

# TEST MODE
# N_RESAMPLES = 1
# sample = sample[:1]

results = []

for item in sample:
    prompt = item["control_prompt"]
    correct_answer = item["correct_answer_num"]

    for resample_id in range(N_RESAMPLES):
        start_time = time.perf_counter()

        try:
            raw_response = call_ollama(MODEL, TEMPERATURE, prompt)
            extracted = extract_passage_mcq(raw_response)
            is_correct = extracted == correct_answer
            error = None

        except Exception as e:
            raw_response = None
            extracted = None
            is_correct = None
            error = str(e)

        end_time = time.perf_counter()

        results.append({
            "item_id": item["item_id"],
            "link": item["link"],
            "question_number": item["question_number"],
            "language": item["language"],
            "resample_id": resample_id,

            "prompt": prompt,
            "raw_response": raw_response,
            "extracted_answer": extracted,
            "is_correct": is_correct,

            "correct_answer_letter": item["correct_answer_letter"],
            "correct_answer_num": correct_answer,

            "audit_label": item["audit_label"],
            "hint_en": item["hint_en"],

            "original_phase2_extracted_answer": item.get("original_phase2_extracted_answer"),
            "original_phase2_is_correct": item.get("original_phase2_is_correct"),
            "original_phase2_classification": item.get("original_phase2_classification"),

            "error": error,
            "latency": end_time - start_time,
        })

        print(
            f"Done: item {item['item_id']} "
            f"#{item['question_number']} "
            f"resample={resample_id} "
            f"extracted={extracted} "
            f"correct={correct_answer} "
            f"is_correct={is_correct}"
        )

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} rows to {OUTPUT_FILE}")