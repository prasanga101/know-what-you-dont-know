import json
import os
from pprint import pprint
from collections import Counter

with open("data/passage1/retrieval_comparison.json", encoding="utf-8") as f:
    ret_tran_items = json.load(f)

with open("data/fidelity_control1/phase3_classification.json", encoding="utf-8") as f:
    field_items = json.load(f)

with open("data/infra_check/aggregated_debasing.json", encoding="utf-8") as f:
    debased_items = json.load(f)

retrival_lookup = {
    (r["link"], r["question_number"], r["language"]): r
    for r in ret_tran_items
}

field_lookup = {
    (r["item_id"], r["link"], r["question_number"]): r
    for r in field_items
}

debased_lookup = {
    (r["item_id"], r["link"]): r
    for r in debased_items
}

THRESHOLD = 8 / 15

def determine_action(item):
    link = item["link"]
    question_number = item["question_number"]
    item_id = item["item_id"]

    en_row = retrival_lookup.get((link, question_number, "en"))
    if en_row and en_row["no_passage_accuracy"] >= THRESHOLD:
        return "TRANSLATE"

    ne_row = retrival_lookup.get((link, question_number, "ne"))
    if ne_row and ne_row["with_passage_accuracy"] >= THRESHOLD:
        return "RETRIEVE"

    debias_row = debased_lookup.get((item_id, link))
    if debias_row:
        repair_passes = debias_row["verdict"] == "permutation_stable"
    else:
        repair_passes = item.get("classification") == "COMPREHENSION_GAP"

    if repair_passes:
        return "REPAIR"

    return "ABSTAIN"

results = []
for item in field_items:
    action = determine_action(item)
    results.append({
        "item_id": item["item_id"],
        "link": item["link"],
        "question_number": item["question_number"],
        "action": action,
    })

assert len(results) == 73, f"expected 73, got {len(results)}"

for r in ret_tran_items:
    if r["language"] == "ne" and r["classification"] == "RETRIEVE":
        results.append({
            "item_id": None,
            "link": r["link"],
            "question_number": r["question_number"],
            "action": "RETRIEVE",
        })

for r in ret_tran_items:
    if r["language"] == "ne" and r["classification"] == "PARTIAL_IMPROVEMENT":
        if r["with_passage_accuracy"] >= THRESHOLD:
            results.append({
                "item_id": None,
                "link": r["link"],
                "question_number": r["question_number"],
                "action": "RETRIEVE",
            })
            continue
        en_row = retrival_lookup.get((r["link"], r["question_number"], "en"))
        if en_row and en_row["no_passage_accuracy"] >= THRESHOLD:
            results.append({
                "item_id": None,
                "link": r["link"],
                "question_number": r["question_number"],
                "action": "TRANSLATE",
            })

assert len(results) == 92, f"expected 92, got {len(results)}"
print(Counter(r["action"] for r in results))

os.makedirs("data/policy_formulation", exist_ok=True)

with open("data/policy_formulation/policy_Action_determination.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)