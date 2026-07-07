import json
import random

DATA_PATH = "data/pilot1/resampling_results.json"
OUTPUT_PATH = "data/pilot1/review_sample.json"

SEED = 42
SAMPLE_SIZES = {
    "UNRESOLVED": 20,
    "BORDERLINE": 10,
    "TRANSLATE": 5,
    "DIRECT": 5,
}


def group_trials(all_trials):
    groups = {}
    for trial in all_trials:
        key = (trial["link"], trial["question_number"], trial["language"])
        if key not in groups:
            groups[key] = []
        groups[key].append(trial)
    return groups


def compute_accuracy(group_trials):
    correct = sum(1 for t in group_trials if t["is_correct"] is True)
    return correct / len(group_trials)


def get_bucket(ne_acc, en_acc):
    if ne_acc >= 0.70:
        return "DIRECT"
    elif ne_acc <= 0.30 and en_acc >= 0.60:
        return "TRANSLATE"
    elif ne_acc <= 0.30 and en_acc <= 0.30:
        return "UNRESOLVED"
    else:
        return "BORDERLINE"


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        output = json.load(f)

    groups = group_trials(output)

    question_accuracies = {}
    for key, trials in groups.items():
        link, question_number, language = key
        acc = compute_accuracy(trials)
        qkey = (link, question_number)
        if qkey not in question_accuracies:
            question_accuracies[qkey] = {}
        question_accuracies[qkey][language] = acc

    buckets = {"DIRECT": [], "TRANSLATE": [], "UNRESOLVED": [], "BORDERLINE": []}
    for qkey, accs in question_accuracies.items():
        b = get_bucket(accs["ne"], accs["en"])
        buckets[b].append(qkey)

    random.seed(SEED)
    sample_items = []
    for bucket_name, n in SAMPLE_SIZES.items():
        pool = buckets[bucket_name]
        sample = random.sample(pool, min(n, len(pool)))
        for link, question_number in sample:
            ne_trials = groups[(link, question_number, "ne")]
            en_trials = groups[(link, question_number, "en")]
            ne_acc = question_accuracies[(link, question_number)]["ne"]
            en_acc = question_accuracies[(link, question_number)]["en"]

            sample_items.append({
                "link": link,
                "question_number": question_number,
                "bucket": bucket_name,
                "ne_accuracy": ne_acc,
                "en_accuracy": en_acc,
                "correct_answer_num": ne_trials[0]["correct_answer_num"],
                "ne_prompt": ne_trials[0]["prompt"],
                "en_prompt": en_trials[0]["prompt"],
                "ne_raw_responses": [t["raw_response"] for t in ne_trials],
                "en_raw_responses": [t["raw_response"] for t in en_trials],
                # fields for you to fill in:
                "translation_valid": None,
                "gold_valid": None,
                "options_aligned": None,
                "question_clear_without_passage": None,
                "likely_dataset_artifact": None,
                "review_notes": ""
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sample_items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(sample_items)} items to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()