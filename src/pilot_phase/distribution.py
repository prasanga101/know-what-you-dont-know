import json
from pprint import pprint
import numpy as np
import os
with open("data/pilot1/resampling_results.json", encoding="utf-8") as f:
    output = json.load(f)

pprint(output[:2])

counts = {"ne": {"correct": 0, "total": 0}, "en": {"correct": 0, "total": 0}}

for item in output:
    lang = item["language"]
    counts[lang]["total"] += 1
    if item["is_correct"] is True:
        counts[lang]["correct"] += 1

for lang in ["ne", "en"]:
    correct = counts[lang]["correct"]
    total = counts[lang]["total"]
    accuracy = correct / total if total else 0
    print(f"{lang}: {correct}/{total} = {accuracy:.4%}")


def group_trials_by_questions_and_language(all_trials):
    groups = {}
    for trial in all_trials:
        key = (trial["link"], trial["question_number"], trial["language"])
        if key not in groups:
            groups[key] = []
        groups[key].append(trial)
    return groups


def build_answer_counts(group_trials):
    answer_counts = {}
    for trial in group_trials:
        answer = trial["extracted_answer"]
        if answer is None:
            continue
        if answer not in answer_counts:
            answer_counts[answer] = 0
        answer_counts[answer] += 1
    return answer_counts


def compute_question_accuracy(group_trials):
    correct_count = 0
    for trial in group_trials:
        if trial["is_correct"] is True:
            correct_count += 1
    total = len(group_trials)
    accuracy = correct_count / total
    return accuracy


def compute_modal_answer_and_agreement(group_trials):
    answer_counts = build_answer_counts(group_trials)
    modal_answer = max(answer_counts, key=answer_counts.get)
    modal_count = answer_counts[modal_answer]
    total_valid = sum(answer_counts.values())
    agreement = modal_count / total_valid
    return modal_answer, agreement


def compute_normalized_entropy(group_trials):
    answer_counts = build_answer_counts(group_trials)
    total = sum(answer_counts.values())
    entropy = 0
    for count in answer_counts.values():
        p = count / total
        entropy -= p * np.log2(p)
    max_possible_entropy = np.log2(4)
    normalized_entropy = entropy / max_possible_entropy
    return normalized_entropy


groups = group_trials_by_questions_and_language(output)
print(f"\nTotal groups: {len(groups)}")

bad_groups = []
for key, trials in groups.items():
    if len(trials) != 15:
        bad_groups.append((key, len(trials)))
print(f"Bad groups: {len(bad_groups)}")
if bad_groups:
    print(bad_groups[:10])


results = []
for key, trials in groups.items():
    link, question_number, language = key
    modal_answer, agreement = compute_modal_answer_and_agreement(trials)
    entropy = compute_normalized_entropy(trials)
    correct_answer_num = trials[0]["correct_answer_num"]
    modal_is_correct = (modal_answer == correct_answer_num)

    results.append({
        "link": link,
        "question_number": question_number,
        "language": language,
        "modal_answer": modal_answer,
        "agreement": agreement,
        "entropy": entropy,
        "modal_is_correct": modal_is_correct,
    })


for lang in ["ne", "en"]:
    lang_results = [r for r in results if r["language"] == lang]
    agreements = [r["agreement"] for r in lang_results]
    entropies = [r["entropy"] for r in lang_results]
    modal_correct_count = sum(1 for r in lang_results if r["modal_is_correct"])

    print(f"\n--- {lang} ---")
    print(f"Questions: {len(lang_results)}")
    print(f"Mean agreement: {np.mean(agreements):.4%}")
    print(f"Median agreement: {np.median(agreements):.4%}")
    print(f"Mean entropy: {np.mean(entropies):.4f}")
    print(f"Median entropy: {np.median(entropies):.4f}")
    print(f"Modal-answer accuracy: {modal_correct_count}/{len(lang_results)} = {modal_correct_count/len(lang_results):.4%}")


for lang in ["ne", "en"]:
    lang_results = [r for r in results if r["language"] == lang]
    confidently_wrong = [
        r for r in lang_results
        if r["agreement"] >= 0.80 and not r["modal_is_correct"]
    ]
    print(f"\nConfidently wrong ({lang}): {len(confidently_wrong)}")


question_accuracies = {}

for key, trials in groups.items():
    link, question_number, language = key
    accuracy = compute_question_accuracy(trials)
    question_key = (link, question_number)
    if question_key not in question_accuracies:
        question_accuracies[question_key] = {}
    question_accuracies[question_key][language] = accuracy

print(f"\nTotal unique questions: {len(question_accuracies)}")

bucket_counts = {"DIRECT": 0, "TRANSLATE": 0, "UNRESOLVED": 0, "BORDERLINE": 0}

for question_key, accuracies in question_accuracies.items():
    ne_acc = accuracies["ne"]
    en_acc = accuracies["en"]

    if ne_acc >= 0.70:
        bucket = "DIRECT"
    elif ne_acc <= 0.30 and en_acc >= 0.60:
        bucket = "TRANSLATE"
    elif ne_acc <= 0.30 and en_acc <= 0.30:
        bucket = "UNRESOLVED"
    else:
        bucket = "BORDERLINE"

    bucket_counts[bucket] += 1
    

bucket_counts = {
    "DIRECT": 0,
    "TRANSLATE": 0,
    "UNRESOLVED": 0,
    "BORDERLINE": 0,
}

bucket_rows = []

for question_key, accuracies in question_accuracies.items():
    link, question_number = question_key

    ne_acc = accuracies["ne"]
    en_acc = accuracies["en"]

    if ne_acc >= 0.70:
        bucket = "DIRECT"
    elif ne_acc <= 0.30 and en_acc >= 0.60:
        bucket = "TRANSLATE"
    elif ne_acc <= 0.30 and en_acc <= 0.30:
        bucket = "UNRESOLVED"
    else:
        bucket = "BORDERLINE"

    bucket_counts[bucket] += 1

    bucket_rows.append({
        "link": link,
        "question_number": question_number,
        "ne_accuracy": ne_acc,
        "en_accuracy": en_acc,
        "bucket": bucket,
    })
    
os.makedirs("data/pilot1",exist_ok=True)
with open("data/pilot1/questions_bucket.json","w",encoding="utf-8") as f:
    json.dump(bucket_rows,f,indent=2,ensure_ascii=False)

print("\n--- Threshold buckets ---")
for bucket, count in bucket_counts.items():
    print(f"{bucket}: {count}")