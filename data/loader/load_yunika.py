from huggingface_hub import hf_hub_download
from datasets import Dataset
import json

# Download the raw file directly
yunika_path = hf_hub_download(
    repo_id="Yunika/Nepali-QA",
    filename="nepali-qa.json",
    repo_type="dataset",
)

with open(yunika_path, encoding="utf-8") as f:
    yunika_raw = json.load(f)

# Unwrap the top-level "data" key
yunika_items = yunika_raw["data"]
print(len(yunika_items))
print(yunika_items[0])

# Wrap into a proper Dataset object, now that yunika_items actually exists
yunika_ds = Dataset.from_list(yunika_items)
print(yunika_ds)