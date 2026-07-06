import os
import json
import time
from fire_api_call.src import call_ollama
from src.pilot_phase.build_mcq import build_mcq_prompt
from src.pilot_phase.extract_answer_mcq import extract_mcq_response

MODEL = "hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M"
TEMPERATURE = 0.7
N_RESAMPLES = 15
INPUT_FILE = "data/pilot1/sample_items.json"
OUTPUT_FILE = "data/pilot1/resampling_results.json"

with open(INPUT_FILE, encoding="utf-8") as f:
    sample_items = json.load(f)


results = []

for item in sample_items:
    for lang in ["ne", "en"]:
        question = item["question_" + lang]
        options = [
            item["mc1_answer_" + lang],
            item["mc2_answer_" + lang],
            item["mc3_answer_" + lang],
            item["mc4_answer_" + lang],
        ]
        correct_answer = item["correct_answer_num"]
        prompt = build_mcq_prompt(question, options)

        for resample_id in range(N_RESAMPLES):
            start_time = time.perf_counter()
            try:
                raw_response = call_ollama(MODEL, TEMPERATURE, prompt)
                extracted = extract_mcq_response(raw_response)
                is_correct = (extracted == correct_answer)
                error = None
            except Exception as e:
                raw_response = None
                extracted = None
                is_correct = None
                error = str(e)
            end_time = time.perf_counter()

            results.append({
                "link": item["link"],
                "question_number": item["question_number"],
                "language": lang,
                "resample_id": resample_id,
                "prompt": prompt,
                "raw_response": raw_response,
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "correct_answer_num": correct_answer,
                "error": error,
                "latency": end_time - start_time,
            })

    print(f"Done: {item['link']} #{item['question_number']}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Total trials: {len(results)}")