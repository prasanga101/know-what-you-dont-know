import json
import os
import time
from src.passage_phase.build_passage_mcq import build_passage_mcq
from src.passage_phase.answer_passage_mcq import extract_passage_mcq
from fire_api_call.src import call_ollama
MODEL = "hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M"
TEMPERATURE = 0.7
N_RESAMPLES = 15
INPUT_FILE = "data/passage1/unresolved_sample.json"
with open(INPUT_FILE, encoding="utf-8") as f:
    sample = json.load(f)

# N_RESAMPLES = 1
# sample = sample[:1]

results = []
for items in sample:
    for lang in ["ne","en"]:
        passage = items[f"passage_{lang}"]
        question = items[f"question_{lang}"]
        options = [
            items[f"mc1_answer_{lang}"],
            items[f"mc2_answer_{lang}"],
            items[f"mc3_answer_{lang}"],
            items[f"mc4_answer_{lang}"],
        ]
        correct_answer = items["correct_answer_num"]
        prompt = build_passage_mcq(question, options, passage)
        for resample_id in range(N_RESAMPLES):
            start_time = time.perf_counter()
            try:
                raw_response = call_ollama(MODEL, TEMPERATURE, prompt)
                extracted = extract_passage_mcq(raw_response)
                is_correct = (extracted == correct_answer)
                error = None
            except Exception as e:
                raw_response = None
                extracted = None
                is_correct = None
                error = str(e)
            end_time = time.perf_counter()
            
            results.append({
                "link": items["link"],
                "question_number": items["question_number"],
                "passage_en":items["passage_en"],
                "passage_ne":items["passage_ne"],
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
            print(f"Done: {items['link']} #{items['question_number']}")
            
            
os.makedirs("data/passage1",exist_ok=True)
with open("data/passage1/resampled_passage_output.json","w",encoding="utf-8") as f:
    json.dump(results,f,indent=2,ensure_ascii=False)

print(f"Total trials: {len(results)}")