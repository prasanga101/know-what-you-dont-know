import os
import json
from pprint import pprint

with open('data/passage1/resampled_passage_output.json',encoding="utf-8") as f:
    sample_items = json.load(f)

with open('data/passage1/retrieval_comparison.json',encoding="utf-8") as f:
    sample_collection = json.load(f)

print(set(row["classification"] for row in sample_collection))

fail_lookup = {
    (row["link"], str(row["question_number"]), row["language"]): row["classification"]
    for row in sample_collection
}

seen = set()
still_failing_res = []

for items in sample_items:
    key = (items["link"], str(items["question_number"]), items["language"])
    classification = fail_lookup.get(key)

    if classification == "STILL_FAILS" and key not in seen and items["language"] == "ne":
        still_failing_res.append(
            {
                **items,
                "classification": classification,
                "english_hint": "",
                "audit_label": ""
            }
        )
        seen.add(key)

print(len(still_failing_res))

os.makedirs("data/fidelity_control1", exist_ok=True)

with open("data/fidelity_control1/still_fails_items.json","w",encoding="utf-8") as f:
    json.dump(still_failing_res, f, indent=2, ensure_ascii=False)