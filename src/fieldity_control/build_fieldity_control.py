import os
import json

INPUT_PATH = "data/fidelity_control1/still_fails_items.json"
OUTPUT_PATH = "data/fidelity_control1/fidelity_control_prompts.json"

NUM_TO_LETTER = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D"
}

CONTROL_TEMPLATE = """तपाईं एउटा प्रश्नको सही जवाफ पहिल्यै थाहा पाउनुभएको छ। तल दिइएको जानकारीको आधारमा,
सही विकल्पको अक्षर (A, B, C, वा D) मात्र लेख्नुहोस्।

अनुच्छेद:
{passage}

{question_block}

सुराक (सहायताको लागि): सही जवाफ भन्नु हो — {hint}

तपाईंको जवाफ (एक अक्षर मात्र, A/B/C/D):"""

def extract_question_block(item):
    prompt = item["prompt"]
    passage = item["passage_ne"]

    if prompt.startswith(passage):
        remainder = prompt[len(passage):]
    else:
        remainder = prompt

    return remainder.split("Answer with only the letter")[0].strip()

with open(INPUT_PATH, encoding="utf-8") as f:
    load_fail_data = json.load(f)

resample_fidelity_control = []

for idx, item in enumerate(load_fail_data, start=1):
    correct_num = str(item["correct_answer_num"])
    correct_letter = NUM_TO_LETTER[correct_num]

    resample_fidelity_control.append({
        "item_id": f"{idx:03d}",
        "link": item["link"],
        "question_number": item["question_number"],
        "language": item["language"],

        "correct_answer_num": correct_num,
        "correct_answer_letter": correct_letter,

        "audit_label": item["audit_label"],
        "hint_en": item["english_hint"],

        "original_phase2_extracted_answer": item.get("extracted_answer"),
        "original_phase2_is_correct": item.get("is_correct"),
        "original_phase2_classification": item.get("classification"),

        "control_prompt": CONTROL_TEMPLATE.format(
            passage=item["passage_ne"],
            question_block=extract_question_block(item),
            hint=item["english_hint"],
        ),
    })

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(resample_fidelity_control, f, ensure_ascii=False, indent=2)

print(f"Saved {len(resample_fidelity_control)} control prompts to {OUTPUT_PATH}")
print(resample_fidelity_control[0]["control_prompt"])