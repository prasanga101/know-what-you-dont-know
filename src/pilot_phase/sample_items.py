import json
import random
import os

with open("data/processed/belebele_en.json", encoding="utf-8") as f:
    belebele_en = json.load(f)

with open("data/processed/belebele_ne.json", encoding="utf-8") as f:
    belebele_ne = json.load(f)

en_lookup = {(row["link"], row["question_number"]): row for row in belebele_en}
random.seed(42)

sample_ne_rows = random.sample(belebele_ne, 250)

paired_sample = []
for ne_row in sample_ne_rows:
    key = (ne_row["link"], ne_row["question_number"])
    en_row = en_lookup.get(key)  
    if en_row is None:
        continue

    paired_sample.append({
        "link": ne_row["link"],
        "question_number": ne_row["question_number"],
        "question_ne": ne_row["question"],
        "question_en": en_row["question"],
        "mc1_answer_ne": ne_row["mc_answer1"], "mc1_answer_en": en_row["mc_answer1"],
        "mc2_answer_ne": ne_row["mc_answer2"], "mc2_answer_en": en_row["mc_answer2"],
        "mc3_answer_ne": ne_row["mc_answer3"], "mc3_answer_en": en_row["mc_answer3"],
        "mc4_answer_ne": ne_row["mc_answer4"], "mc4_answer_en": en_row["mc_answer4"],
        "correct_answer_num": ne_row["correct_answer_num"],
    })

os.makedirs("data/pilot1", exist_ok=True)
with open("data/pilot1/sample_items.json", "w", encoding="utf-8") as f:
    json.dump(paired_sample, f, indent=2, ensure_ascii=False)

print(f"Sampled {len(paired_sample)} parallel items")