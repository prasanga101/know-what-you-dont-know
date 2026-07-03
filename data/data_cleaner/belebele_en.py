from data.loader.load_data import belebele_en
import os
import json
test_belebele_en = belebele_en["test"]
print(belebele_en)
print(test_belebele_en[0])
os.makedirs("data/processed", exist_ok=True)
records = [
    {k: v for k, v in row.items() if k != "ds"}
    for row in test_belebele_en
]
with open("processed/belebele_en.json", "w",encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)