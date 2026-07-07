import json
import os

with open("data/pilot1/questions_bucket.json", encoding="utf-8") as f:
    bucket_rows = json.load(f)
    
with open("data/passage1/sample_passage_items.json", encoding="utf-8") as f:
    sample_items = json.load(f)

bucket_lookup = {
    (row["link"], row["question_number"]): row["bucket"]
    for row in bucket_rows
}
phase2_items=[]
for items in sample_items:
    key = (items["link"], items["question_number"])
    bucket = bucket_lookup.get(key)
    if bucket in {"UNRESOLVED","BORDERLINE"}:
        
        phase2_items.append({
            **items,
            "bucket":bucket
        })
os.makedirs("data/passage1", exist_ok=True)
with open("data/passage1/unresolved_sample.json" , "w", encoding="utf-8") as f:
    json.dump(phase2_items,f,indent=2,ensure_ascii=False)
print(f"Filtered unresolved items: {len(phase2_items)}")
print(f"Bucket rows loaded: {len(bucket_rows)}")
print(f"Passage items loaded: {len(sample_items)}")
