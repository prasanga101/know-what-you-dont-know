import json

NO_PASSAGE_FILE = "data/pilot1/resampling_results.json"
WITH_PASSAGE_FILE = "data/passage1/resampled_passage_output.json"

FAIL_THRESHOLD = 0.30
SUCCESS_THRESHOLD = 0.60


def group_by_question_language(trials):
    groups = {}
    for trial in trials:
        key = (trial["link"], trial["question_number"], trial["language"])
        if key not in groups:
            groups[key] = []
        groups[key].append(trial)
    return groups


def compute_accuracy(group_trials):
    correct = sum(1 for t in group_trials if t["is_correct"] is True)
    return correct / len(group_trials)


def build_question_accuracy(trials):
    groups = group_by_question_language(trials)
    question_acc = {}
    for (link, qnum, lang), group_trials in groups.items():
        acc = compute_accuracy(group_trials)
        qkey = (link, qnum)
        if qkey not in question_acc:
            question_acc[qkey] = {}
        question_acc[qkey][lang] = acc
    return question_acc


def classify(no_passage_acc, with_passage_acc):
    if no_passage_acc is None or with_passage_acc is None:
        return "MISSING_DATA"
    if no_passage_acc > FAIL_THRESHOLD:
        return "WAS_NOT_FAILING"
    if with_passage_acc >= SUCCESS_THRESHOLD:
        return "RETRIEVE"
    elif with_passage_acc <= FAIL_THRESHOLD:
        return "STILL_FAILS"
    else:
        return "PARTIAL_IMPROVEMENT"


def main():
    with open(NO_PASSAGE_FILE, encoding="utf-8") as f:
        no_passage_trials = json.load(f)
    with open(WITH_PASSAGE_FILE, encoding="utf-8") as f:
        with_passage_trials = json.load(f)

    no_passage_acc = build_question_accuracy(no_passage_trials)
    with_passage_acc = build_question_accuracy(with_passage_trials)

    results = []
    for qkey in with_passage_acc:
        if qkey not in no_passage_acc:
            continue
        link, qnum = qkey
        for lang in ["ne", "en"]:
            np_acc = no_passage_acc[qkey].get(lang)
            wp_acc = with_passage_acc[qkey].get(lang)
            classification = classify(np_acc, wp_acc)
            results.append({
                "link": link,
                "question_number": qnum,
                "language": lang,
                "no_passage_accuracy": np_acc,
                "with_passage_accuracy": wp_acc,
                "classification": classification,
            })

    with open("data/passage1/retrieval_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for lang in ["ne", "en"]:
        lang_results = [r for r in results if r["language"] == lang]
        counts = {}
        for r in lang_results:
            counts[r["classification"]] = counts.get(r["classification"], 0) + 1
        print(f"\n--- {lang} ---")
        for k, v in counts.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()