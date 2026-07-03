from data.loader.load_data import belebele_bn
import json
import os
test_belebele_bn = belebele_bn["test"]
print(belebele_bn)
print(test_belebele_bn[0])
os.makedirs("data/processed", exist_ok=True)
records = [
    {k: v for k, v in row.items() if k != "ds"}
    for row in test_belebele_bn
]
with open("processed/belebele_bn.json", "w",encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)