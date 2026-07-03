from data.loader.load_yunika import yunika_ds
import json
import os
def flatten_yunika(yunika_ds):
    records_yunika = []
    for record in yunika_ds:
        item_id = record["id"]          
        context = record["context"]
        question = record["question"]

        answer_texts = record["answers"]["text"]
        answer_starts = record["answers"]["answer_start"]

        primary_answer = answer_texts[0] if answer_texts else None
        primary_answer_start = answer_starts[0] if answer_starts else None

        records_yunika.append({
            "id": item_id,
            "context": context,
            "question": question,
            "answer": primary_answer,
            "answer_start": primary_answer_start,
            "all_answers": answer_texts,
            "all_answer_starts": answer_starts,
        })
    return records_yunika


if __name__ == "__main__":
    yunika_records = flatten_yunika(yunika_ds)
    print(f"Total questions: {len(yunika_records)}")
    print(yunika_records[0])
    n_unanswerable = sum(1 for r in yunika_ds if len(r["answers"]["text"]) == 0)
    print(f"Unanswerable in Yunika: {n_unanswerable} / {len(yunika_ds)}")
    os.makedirs("data/processed", exist_ok="True")
    with open("processed/yunika.json", "w") as f:
        json.dump(yunika_records, f, indent=2, ensure_ascii=False)