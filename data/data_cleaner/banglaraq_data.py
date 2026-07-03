from data.loader.load_banglaraq import data
from collections import Counter
import json
import os
def bangla_flatten(data):
    records = []
    for passage in data:
        passage_id = passage["passage_id"]
        context = passage["context"]
        title = passage["title"]

        for qn in passage["qas"]:
            question_id = qn["question_id"]
            question_text = qn["question_text"].strip()
            question_type = qn["question_type"]
            is_answerable = qn["is_answerable"] == "1"

            answer_texts = qn["answers"]["answer_text"]
            answer_types = qn["answers"]["answer_type"]

            primary_answer = answer_texts[0] if answer_texts else None
            primary_answer_type = answer_types[0] if answer_types else None

            records.append({
                "passage_id": passage_id,
                "question_id": question_id,
                "context": context,
                "title": title,
                "question_text": question_text,
                "is_answerable": is_answerable,
                "question_type": question_type,
                "primary_answer": primary_answer,
                "primary_answer_type": primary_answer_type,
                "all_answer_texts": answer_texts,
                "all_answer_types": answer_types,
            })

    return records


if __name__ == "__main__":
    banglarqa_records = bangla_flatten(data)
    os.makedirs("data/processed", exist_ok=True)
    with open("processed/banglaraq.json", "w") as f:
        json.dump(banglarqa_records, f, indent=2, ensure_ascii=False)
    print(f"Total passages: {len(data)}")
    print(f"Total questions: {len(banglarqa_records)}")

    n_unanswerable = sum(1 for r in banglarqa_records if not r["is_answerable"])
    print(f"Unanswerable: {n_unanswerable} / {len(banglarqa_records)} ({n_unanswerable/len(banglarqa_records):.1%})")

    print(Counter(r["question_type"] for r in banglarqa_records))