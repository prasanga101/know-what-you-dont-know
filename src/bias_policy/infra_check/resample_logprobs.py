import json
import os
import requests
import time
import math

MODEL = "hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M"

INPUT_FILE = "data/infra_check/resample_ollama_logprobs.json"
OUTPUT_FILE = "data/infra_check/ollama_logprob_probe_results.json"

TARGET_OPTIONS = ["A", "B", "C", "D"]


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
        option: lp
        for option, lp in abcd_logprobs.items()
        if lp is not None
    }

    if not available:
        return None, None

    top_option = max(available, key=available.get)
    return top_option, available[top_option]


with open(INPUT_FILE, encoding="utf-8") as f:
    sample = json.load(f)

results = []
warnings = []

for item in sample:
    payload = {
        "model": MODEL,
        "prompt": item["logprob_prompt"],
        "stream": False,
        "logprobs": True,
        "top_logprobs": 20,
        "options": {
            "temperature": 0,
            "repeat_penalty": 1.0,
            "top_k": 100,
            "top_p": 1.0,
            "mirostat": 0
        }
    }

    start_time = time.perf_counter()

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        error = None

    except Exception as e:
        data = {}
        error = str(e)

    end_time = time.perf_counter()

    available_keys = list(data.keys())
    has_logprobs = "logprobs" in data

    abcd_logprobs = extract_abcd_logprobs(data)
    abcd_probs = logprobs_to_probs(abcd_logprobs)
    top_option, top_option_logprob = get_top_option(abcd_logprobs)

    correct_letter = item["correct_answer_letter"]
    correct_option_logprob = abcd_logprobs.get(correct_letter)
    correct_option_prob = abcd_probs.get(correct_letter)

    # Sanity invariant: at temperature=0, the model's own emitted token is
    # by definition the argmax over the FULL vocabulary, so if it emitted
    # one of A/B/C/D, that letter MUST also be the argmax within our
    # extracted A/B/C/D subset. If it isn't, extraction is still broken.
    response_letter = (data.get("response") or "").strip().upper()[:1]
    if has_logprobs and response_letter in TARGET_OPTIONS and top_option != response_letter:
        warning = (
            f"item={item['item_id']}: top_option={top_option} "
            f"but model emitted {response_letter} — extraction likely broken"
        )
        warnings.append(warning)
        print(f"WARNING {warning}")

    result = {
        "item_id": item["item_id"],
        "link": item["link"],
        "question_number": item["question_number"],
        "classification": item["classification"],
        "audit_label": item.get("audit_label"),

        "correct_answer_letter": correct_letter,
        "correct_answer_num": item["correct_answer_num"],

        "model": MODEL,
        "prompt": item["logprob_prompt"],
        "response_text": data.get("response"),

        "has_logprobs": has_logprobs,
        "available_keys": available_keys,

        "logprob_A": abcd_logprobs["A"],
        "logprob_B": abcd_logprobs["B"],
        "logprob_C": abcd_logprobs["C"],
        "logprob_D": abcd_logprobs["D"],

        "prob_A_known_options": abcd_probs["A"],
        "prob_B_known_options": abcd_probs["B"],
        "prob_C_known_options": abcd_probs["C"],
        "prob_D_known_options": abcd_probs["D"],

        "top_option": top_option,
        "top_option_logprob": top_option_logprob,
        "correct_option_logprob": correct_option_logprob,
        "correct_option_prob_known_options": correct_option_prob,
        "top_option_is_correct": top_option == correct_letter,
        "top_option_matches_response": top_option == response_letter,

        "first_token_top_logprobs": get_first_token_top_logprobs(data),
        "ollama_raw_response": data,
        "error": error,
        "latency": end_time - start_time,
    }

    results.append(result)

    print(f"item={item['item_id']} response={data.get('response')}")
    print(
        f"A={abcd_logprobs['A']} "
        f"B={abcd_logprobs['B']} "
        f"C={abcd_logprobs['C']} "
        f"D={abcd_logprobs['D']}"
    )
    print(f"top_option={top_option} correct={correct_letter} ok={top_option == correct_letter}")
    print(f"has_logprobs={has_logprobs} error={error}")
    print("-" * 80)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} logprob probe results to {OUTPUT_FILE}")

if warnings:
    print(f"\n{len(warnings)} extraction warnings — investigate before trusting this data:")
    for w in warnings:
        print(f"  {w}")
else:
    print("\nNo extraction warnings — top_option matches the model's own emitted letter on every item.")