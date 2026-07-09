import os
import json
from pprint import pprint
from collections import defaultdict, Counter

with open("data/fidelity_control1/resampled_fidelity_control_output.json", encoding="utf-8") as f:
    resampled_items = json.load(f)
pprint(resampled_items[0])

results = []
for items in resampled_items:
    if items["audit_label"] in {"CORRUPTED_EVIDENCE", "AMBIGUOUS_ITEM"}:
        continue
    results.append({**items})

os.makedirs("data/fidelity_control1", exist_ok=True)
with open("data/fidelity_control1/filtered_resampled_items.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Kept {len(results)} rows across {len({r['item_id'] for r in results})} items")

# --- Sub-phase E: classify each item ---
by_item = defaultdict(list)
for row in results:
    by_item[row["item_id"]].append(row)

classified = []
for item_id, rows in by_item.items():
    correct_count = sum(1 for r in rows if r["is_correct"])
    correct_answer_letter = rows[0]["correct_answer_letter"]

    if correct_count >= 8:
        if correct_answer_letter == "A":
            classification = "PASS_CONFOUNDED_BY_BIAS"
        else:
            classification = "COMPREHENSION_GAP"
    else:
        classification = "GENERATION_FIDELITY_ISSUE"

    classified.append({
        "item_id": item_id,
        "link": rows[0]["link"],
        "question_number": rows[0]["question_number"],
        "audit_label": rows[0]["audit_label"],
        "correct_answer_letter": correct_answer_letter,
        "correct_count": correct_count,
        "n_resamples": len(rows),
        "classification": classification,
    })

with open("data/fidelity_control1/phase3_classification.json", "w") as f:
    json.dump(classified, f, indent=2, ensure_ascii=False)

print(f"Classified {len(classified)} items")
counts = Counter(c["classification"] for c in classified)
print(counts)

total = len(classified)
summary = {
    "total_items": total,
    "threshold": "correct_count >= 8 / 15",
    "counts": dict(counts),
    "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
}

with open("data/fidelity_control1/phase3_classification_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Summary written to phase3_classification_summary.json")
print(summary)