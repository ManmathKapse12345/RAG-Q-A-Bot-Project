# fix_sources.py
import json

with open("eval_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for item in dataset:
    if item.get("source") == "smart_dialer_assignment.md":
        item["source"] = "document.pdf"

with open("eval_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("Fixed all source fields to 'document.pdf'")