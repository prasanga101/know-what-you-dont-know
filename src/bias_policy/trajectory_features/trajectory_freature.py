import json
import math
import os
from collections import Counter, defaultdict

import requests


MODEL = "hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M"
OLLAMA_URL = "http://localhost:11434/api/generate"

POLICY_FILE = "data/policy_formulation/policy_Action_determination.json"
RESAMPLING_FILE = "data/pilot1/resampling_results.json"
OUTPUT_FILE = "data/policy_formulation/trajectory_feature.json"
CHECKPOINT_FILE = "data/policy_formulation/trajectory_feature.checkpoint.json"

TARGET_LANGUAGE = "ne"
TARGET_OPTIONS = ["A", "B", "C", "D"]
ANSWER_NUM_TO_OPTION = {"1": "A", "2": "B", "3": "C", "4": "D"}
EXPECTED_ITEM_COUNT = 92

FEATURE_NAMES = [
    "entropy",
    "agreement_rate",
    "resample_prob_A",
    "resample_prob_B",
    "resample_prob_C",
    "resample_prob_D",
    "resample_top2_margin",
    "invalid_response_rate",
    "token_probability_variance",
    "logprob_prob_A",
    "logprob_prob_B",
    "logprob_prob_C",
    "logprob_prob_D",
    "logprob_entropy",
    "logprob_top2_margin",
    "resample_logprob_top_match",
    "resample_logprob_js_divergence",
]


def compute_entropy_and_agreement(rows):
    answers = [row["extracted_answer"] for row in rows]
    counts = Counter(answers)
    total = len(answers)

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    most_common_count = counts.most_common(1)[0][1]
    agreement_rate = most_common_count / total

    return entropy, agreement_rate


def compute_resample_distribution(rows):
    option_counts = Counter()
    invalid_count = 0

    for row in rows:
        option = ANSWER_NUM_TO_OPTION.get(str(row.get("extracted_answer")))
        if option is None:
            invalid_count += 1
        else:
            option_counts[option] += 1

    valid_count = sum(option_counts.values())
    if valid_count == 0:
        raise ValueError("Cannot build a resample distribution without valid A/B/C/D answers")

    probabilities = {
        option: option_counts[option] / valid_count
        for option in TARGET_OPTIONS
    }
    invalid_response_rate = invalid_count / len(rows)

    return probabilities, invalid_response_rate


def get_first_token_top_logprobs(data):
    logprobs = data.get("logprobs", [])

    if not logprobs:
        return []

    return logprobs[0].get("top_logprobs", [])


def extract_abcd_logprobs(data):
    top_logprobs = get_first_token_top_logprobs(data)

    abcd = {option: None for option in TARGET_OPTIONS}

    for row in top_logprobs:
        token = row.get("token", "").strip().upper()
        logprob = row.get("logprob")

        # top_logprobs is rank-ordered (most to least likely). Some
        # tokenizers emit more than one vocab token that normalizes to the
        # same letter (e.g. "A" vs " A"). Keep only the FIRST (highest
        # probability) match; a later duplicate must never overwrite it.
        if token in abcd and abcd[token] is None:
            abcd[token] = logprob

    return abcd


def logprobs_to_probs(abcd_logprobs):
    available = {
        option: lp
        for option, lp in abcd_logprobs.items()
        if lp is not None
    }

    probs = {option: None for option in TARGET_OPTIONS}

    if not available:
        return probs

    max_lp = max(available.values())
    exp_values = {
        option: math.exp(lp - max_lp)
        for option, lp in available.items()
    }

    total = sum(exp_values.values())

    for option, value in exp_values.items():
        probs[option] = value / total

    return probs


def get_top_option(abcd_logprobs):
    available = {
        option: logprob
        for option, logprob in abcd_logprobs.items()
        if logprob is not None
    }

    if not available:
        return None

    return max(available, key=available.get)


def compute_token_probability_variance(abcd_probs):
    """Population variance across the normalized A/B/C/D probabilities."""
    missing_options = [
        option for option in TARGET_OPTIONS if abcd_probs[option] is None
    ]
    if missing_options:
        raise ValueError(
            "Cannot compute a four-option probability variance; missing "
            f"probabilities for {missing_options}"
        )

    probabilities = [abcd_probs[option] for option in TARGET_OPTIONS]
    mean_probability = sum(probabilities) / len(probabilities)
    return sum(
        (probability - mean_probability) ** 2
        for probability in probabilities
    ) / len(probabilities)


def compute_probability_entropy(probabilities):
    return -sum(
        probability * math.log2(probability)
        for probability in probabilities.values()
        if probability > 0
    )


def compute_top2_margin(probabilities):
    ranked_probabilities = sorted(probabilities.values(), reverse=True)
    return ranked_probabilities[0] - ranked_probabilities[1]


def get_top_probability_option(probabilities):
    return max(TARGET_OPTIONS, key=probabilities.get)


def compute_jensen_shannon_divergence(first, second):
    midpoint = {
        option: (first[option] + second[option]) / 2
        for option in TARGET_OPTIONS
    }

    def kl_divergence(distribution, reference):
        return sum(
            distribution[option]
            * math.log2(distribution[option] / reference[option])
            for option in TARGET_OPTIONS
            if distribution[option] > 0
        )

    return (
        kl_divergence(first, midpoint) + kl_divergence(second, midpoint)
    ) / 2


def call_ollama_for_logprobs(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 20,
        "options": {
            "temperature": 0,
            "repeat_penalty": 1.0,
            "top_k": 100,
            "top_p": 1.0,
            "mirostat": 0,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def get_baseline_prompt(rows):
    prompts = {row["prompt"] for row in rows}
    if len(prompts) != 1:
        raise ValueError(
            f"Expected one baseline prompt across the resamples, got {len(prompts)}"
        )
    return next(iter(prompts))


def write_json(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE) or os.path.getsize(CHECKPOINT_FILE) == 0:
        return []

    with open(CHECKPOINT_FILE, encoding="utf-8") as file:
        rows = json.load(file)

    for row in rows:
        missing_features = [feature for feature in FEATURE_NAMES if feature not in row]
        if missing_features:
            raise ValueError(
                f"Checkpoint uses an older schema; missing {missing_features}. "
                f"Remove {CHECKPOINT_FILE} and rerun."
            )

    return rows


def feature_range(rows, feature):
    values = [row[feature] for row in rows]
    return min(values), sum(values) / len(values), max(values)


def eta_squared(results, feature):
    """Share of one feature's variance explained by the action groups."""
    values = [row[feature] for row in results]
    overall_mean = sum(values) / len(values)
    total_variation = sum((value - overall_mean) ** 2 for value in values)

    if total_variation == 0:
        return 0.0

    rows_by_action = defaultdict(list)
    for row in results:
        rows_by_action[row["action"]].append(row)

    between_group_variation = 0.0
    for rows in rows_by_action.values():
        group_values = [row[feature] for row in rows]
        group_mean = sum(group_values) / len(group_values)
        between_group_variation += len(rows) * (group_mean - overall_mean) ** 2

    return between_group_variation / total_variation


def print_group_by_action_summary(results):
    rows_by_action = defaultdict(list)
    for row in results:
        rows_by_action[row["action"]].append(row)

    print("\nGroup-by-action feature summary (min / mean / max)")
    for action in sorted(rows_by_action):
        rows = rows_by_action[action]
        print(f"\n{action} (n={len(rows)})")
        for feature in FEATURE_NAMES:
            minimum, mean, maximum = feature_range(rows, feature)
            print(
                f"  {feature}: "
                f"{minimum:.6f} / {mean:.6f} / {maximum:.6f}"
            )

    # This is a descriptive, univariate separation check, not classifier
    # validation. Phase E should still use held-out or cross-validated metrics.
    print("\nUnivariate separation by action (eta-squared; higher is better)")
    for feature in FEATURE_NAMES:
        print(f"  {feature}: {eta_squared(results, feature):.6f}")


def main():
    with open(POLICY_FILE, encoding="utf-8") as file:
        policy_items = json.load(file)

    with open(RESAMPLING_FILE, encoding="utf-8") as file:
        resampling_results = json.load(file)

    if len(policy_items) != EXPECTED_ITEM_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ITEM_COUNT} policy items, got {len(policy_items)}"
        )

    item_lookup = defaultdict(list)
    for row in resampling_results:
        key = (row["link"], row["question_number"], row["language"])
        item_lookup[key].append(row)

    results = load_checkpoint()
    completed_keys = {
        (row["link"], row["question_number"])
        for row in results
    }
    warnings = []

    for index, item in enumerate(policy_items, start=1):
        key = (item["link"], item["question_number"], TARGET_LANGUAGE)
        output_key = (item["link"], item["question_number"])

        if output_key in completed_keys:
            print(
                f"[{index}/{EXPECTED_ITEM_COUNT}] item={item.get('item_id')} "
                "restored from checkpoint"
            )
            continue

        resample_rows = item_lookup.get(key)
        if not resample_rows:
            raise KeyError(f"No Nepali baseline resamples found for {key}")

        entropy, agreement_rate = compute_entropy_and_agreement(resample_rows)
        resample_probs, invalid_response_rate = compute_resample_distribution(
            resample_rows
        )
        baseline_prompt = get_baseline_prompt(resample_rows)

        try:
            ollama_data = call_ollama_for_logprobs(baseline_prompt)
        except requests.RequestException as error:
            raise RuntimeError(
                f"Ollama logprob call failed for item {item.get('item_id')} ({key})"
            ) from error

        abcd_logprobs = extract_abcd_logprobs(ollama_data)
        abcd_probs = logprobs_to_probs(abcd_logprobs)
        token_probability_variance = compute_token_probability_variance(abcd_probs)

        resample_top_option = get_top_probability_option(resample_probs)
        logprob_top_option = get_top_probability_option(abcd_probs)
        resample_logprob_top_match = int(
            resample_top_option == logprob_top_option
        )

        top_option = get_top_option(abcd_logprobs)
        response_letter = (ollama_data.get("response") or "").strip().upper()[:1]
        if response_letter in TARGET_OPTIONS and top_option != response_letter:
            warnings.append(
                f"item={item.get('item_id')}: top_option={top_option}, "
                f"emitted={response_letter}"
            )

        results.append(
            {
                "item_id": item.get("item_id"),
                "link": item["link"],
                "question_number": item["question_number"],
                "language": TARGET_LANGUAGE,
                "resample_count": len(resample_rows),
                "action": item["action"],
                "entropy": entropy,
                "agreement_rate": agreement_rate,
                "resample_prob_A": resample_probs["A"],
                "resample_prob_B": resample_probs["B"],
                "resample_prob_C": resample_probs["C"],
                "resample_prob_D": resample_probs["D"],
                "resample_top2_margin": compute_top2_margin(resample_probs),
                "invalid_response_rate": invalid_response_rate,
                "token_probability_variance": token_probability_variance,
                "logprob_prob_A": abcd_probs["A"],
                "logprob_prob_B": abcd_probs["B"],
                "logprob_prob_C": abcd_probs["C"],
                "logprob_prob_D": abcd_probs["D"],
                "logprob_entropy": compute_probability_entropy(abcd_probs),
                "logprob_top2_margin": compute_top2_margin(abcd_probs),
                "resample_logprob_top_match": resample_logprob_top_match,
                "resample_logprob_js_divergence": (
                    compute_jensen_shannon_divergence(
                        resample_probs,
                        abcd_probs,
                    )
                ),
            }
        )
        completed_keys.add(output_key)
        write_json(CHECKPOINT_FILE, results)

        print(
            f"[{index}/{EXPECTED_ITEM_COUNT}] item={item.get('item_id')} "
            f"action={item['action']} probabilities={abcd_probs} "
            f"variance={token_probability_variance:.6f}"
        )

    if len(results) != EXPECTED_ITEM_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ITEM_COUNT} trajectory rows, got {len(results)}"
        )

    result_lookup = {
        (row["link"], row["question_number"]): row
        for row in results
    }
    results = [
        result_lookup[(item["link"], item["question_number"])]
        for item in policy_items
    ]

    write_json(OUTPUT_FILE, results)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\nSaved {len(results)} trajectory feature rows to {OUTPUT_FILE}")
    print_group_by_action_summary(results)

    if warnings:
        print("\nExtraction warnings — inspect before trusting the feature:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("\nNo extraction warnings.")


if __name__ == "__main__":
    main()
