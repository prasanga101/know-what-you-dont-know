from data.loader.load_data import belebele_ne
import os
import json
test_belebele_ne = belebele_ne["test"]
print(belebele_ne)
print(test_belebele_ne[0])
os.makedirs("data/processed", exist_ok=True)
records = [
    {k: v for k, v in row.items() if k != "ds"}
    for row in test_belebele_ne
]
with open("processed/belebele_ne.json", "w",encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)