from data.loader.load_data import squad_v2
import os
import json
train_squad2_en=squad_v2["train"]
val_squad2_en =squad_v2["validation"]
print(squad_v2)
print(train_squad2_en[0])
print(val_squad2_en[0])
os.makedirs("data/processed", exist_ok=True)
with open("processed/train_squad2_en.json", "w", encoding="utf-8") as f:
    json.dump(list(train_squad2_en), f, indent=2, ensure_ascii=False)

with open("processed/val_squad2_en.json", "w", encoding="utf-8") as f:
    json.dump(list(val_squad2_en), f, indent=2, ensure_ascii=False)