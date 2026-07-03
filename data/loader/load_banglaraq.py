import json
from huggingface_hub import hf_hub_download
from datasets import DatasetDict, Dataset

train_path = hf_hub_download(repo_id="sartajekram/BanglaRQA", filename="Train.json", repo_type="dataset")
with open(train_path, encoding="utf-8") as f:
    banglarqa_train = json.load(f)
    
    
print(type(banglarqa_train))


if isinstance(banglarqa_train, DatasetDict):
    print(banglarqa_train.keys())          # e.g. dict_keys(['train', 'validation', 'test'])
    print(banglarqa_train["train"])         # shows features + num_rows
    print(banglarqa_train["train"][0])      # first actual example
elif isinstance(banglarqa_train, dict):
    print(banglarqa_train.keys())
elif isinstance(banglarqa_train, list):
    print(len(banglarqa_train))
    print(banglarqa_train[0])
else:
    print("Unhandled type:", type(banglarqa_train))

data = banglarqa_train["data"]
print("Number of top-level entries:", len(data))
print(json.dumps(data[0], indent=2, ensure_ascii=False))