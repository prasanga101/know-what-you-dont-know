import os
import json
from collections import defaultdict, Counter

with open("data/infra_check/debasic_ollama_logprob_resampling_result.json", encoding="utf-8") as f:
    debase_items = json.load(f)

aggregate_results = defaultdict(list)
for item in debase_items:
    aggregate_results[item["item_id"]].append(item)


def compute_verdict(match_count, picks_by_shift):
    if match_count == 4:
        return "permutation_stable"

    most_common_pick, most_common_count = Counter(picks_by_shift).most_common(1)[0]
    if most_common_count >= 3:
        return "position_anchored"

    return "complex_interaction"


final_results = []
for item_id, rows in aggregate_results.items():
    sorted_rows = sorted(rows, key=lambda r: r["shift"])
    picks_by_shift = [r["top_option"] for r in sorted_rows]
    match_count = sum(r["top_option_is_correct"] for r in sorted_rows)

    final_results.append({
        "item_id": item_id,
        "link": sorted_rows[0]["link"],
        "original_correct_letter": sorted_rows[0]["original_correct_letter"],
        "picks_by_shift": picks_by_shift,
        "match_count": match_count,
        "verdict": compute_verdict(match_count, picks_by_shift),
    })

assert len(final_results) == 12, f"expected 12, got {len(final_results)}"

os.makedirs("ddata/infra_check/", exist_ok=True)
with open("data/infra_check/aggregated_debasing.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"saved {len(final_results)} verdicts")