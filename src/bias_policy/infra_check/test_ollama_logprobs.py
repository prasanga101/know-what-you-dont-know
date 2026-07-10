import json
import os
from pprint import pprint
with open("data/fidelity_control1/phase3_classification.json", encoding="utf-8") as f:
    bias_item = json.load(f)

with open("data/fidelity_control1/resampled_fidelity_control_output.json", encoding="utf-8") as f:
    resampled_items = json.load(f)
    
print("bias items /n")
pprint(bias_item[:1])
print("resampled items /n")
pprint(resampled_items[:1])


bias_lookup = {
    (row["link"], str(row["question_number"]), row["item_id"]): row["classification"]
    for row in bias_item
}

ollama_logprobs = []
for item in resampled_items:
    key = (item["link"], str(item["question_number"]), item["item_id"])
    classification = bias_lookup.get(key)
    if classification != "PASS_CONFOUNDED_BY_BIAS":
        continue
    ollama_logprobs.append({**item, "classification": classification})

os.makedirs("data/infra_check", exist_ok=True)
with open("data/infra_check/ollama_logprobs.json", "w", encoding="utf-8") as f:
    json.dump(ollama_logprobs, f, indent=2, ensure_ascii=False)

print(f"matched {len(ollama_logprobs)} items")