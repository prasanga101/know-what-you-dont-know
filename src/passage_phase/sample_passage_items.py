import json
import os

with open("data/processed/belebele_en.json", encoding="utf-8") as f:
    belebele_en = json.load(f)

with open("data/processed/belebele_ne.json", encoding="utf-8") as f:
    belebele_ne = json.load(f)

with open("data/pilot1/sample_items.json", encoding="utf-8") as f:
    sample_items = json.load(f)

en_lookup = {
    (row["link"], row["question_number"]): row
    for row in belebele_en
}

ne_lookup = {
    (row["link"], row["question_number"]): row
    for row in belebele_ne
}

    
paired_sample=[]
for items in sample_items:
    key =(items["link"], items["question_number"])
    en_row = en_lookup.get(key)
    ne_row = ne_lookup.get(key)
    if en_row and ne_row:
        paired_sample.append({
            "link": items["link"],
            "question_number": items["question_number"],
            "passage_en": en_row["flores_passage"],
            "passage_ne": ne_row["flores_passage"],
            "question_en": en_row["question"],
            "question_ne": ne_row["question"],
            "mc1_answer_ne": ne_row["mc_answer1"], "mc1_answer_en": en_row["mc_answer1"],
            "mc2_answer_ne": ne_row["mc_answer2"], "mc2_answer_en": en_row["mc_answer2"],
            "mc3_answer_ne": ne_row["mc_answer3"], "mc3_answer_en": en_row["mc_answer3"],
            "mc4_answer_ne": ne_row["mc_answer4"], "mc4_answer_en": en_row["mc_answer4"],
            "correct_answer_num": ne_row["correct_answer_num"]
        })

os.makedirs("data/passage1", exist_ok=True)
with open("data/passage1/sample_passage_items.json", "w", encoding="utf-8") as f:
    json.dump(paired_sample, f, indent=2, ensure_ascii=False)

print(f"Sampled {len(paired_sample)} parallel items")