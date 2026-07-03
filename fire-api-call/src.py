import json
import math
import re
import time
import unicodedata
from collections import Counter
from typing import Any

import ollama


CONFIG = {
    "MODELS": [
        "hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M",
        "hf.co/CohereLabs/tiny-aya-global-GGUF:Q4_K_M",
    ],
    "N_RESAMPLES": 15,
    "TEMPERATURE": 0.7,
    "TOP_P": 0.9,
    "MAX_TOKENS": 20,
    "OUTPUT_FILE": "resampling_test_results_bn.json",
    # Filter which prompts run this pass — set to None to run everything.
    "LANGUAGES": ["bn"],
}


PROMPTS = [
    {
        "id": "nepal_capital_ne",
        "fact_id": "nepal_capital",
        "lang": "ne",
        "text": "नेपालको राजधानी के हो?",
        "expected_canonical": "kathmandu",
    },
    {
        "id": "nepal_capital_en",
        "fact_id": "nepal_capital",
        "lang": "en",
        "text": "What is the capital of Nepal?",
        "expected_canonical": "kathmandu",
    },
    {
        "id": "nepal_capital_bn",
        "fact_id": "nepal_capital",
        "lang": "bn",
        "text": "নেপালের রাজধানী কী?",
        "expected_canonical": "kathmandu",
    },
    {
        "id": "bangladesh_capital_bn",
        "fact_id": "bangladesh_capital",
        "lang": "bn",
        "text": "বাংলাদেশের রাজধানী কী?",
        "expected_canonical": "dhaka",
    },
    {
        "id": "bangladesh_capital_en",
        "fact_id": "bangladesh_capital",
        "lang": "en",
        "text": "What is the capital of Bangladesh?",
        "expected_canonical": "dhaka",
    },
    {
        "id": "bangladesh_capital_ne",
        "fact_id": "bangladesh_capital",
        "lang": "ne",
        "text": "बङ्गलादेशको राजधानी के हो?",
        "expected_canonical": "dhaka",
    },
]


ALIASES = {
    "kathmandu": {
        "kathmandu",
        "katmandu",
        "काठमाडौँ",
        "काठमान्डौ",
        "काठमाडौं",
        "काठमाण्डौ",
        "काठमांडू",
        "কাঠমান্ডু",
        "কাটমন্ডু",
    },
    "dhaka": {
        "dhaka",
        "dacca",
        "ঢাকা",
        "ঢাক",
        "ढाका",
        "डाका",
    },
}

# Known inflectional suffixes for Bengali and Nepali — used to recognize
# grammatically inflected forms of a correct answer (e.g. ঢাকার = "Dhaka's").
# Deliberately a fixed linguistic list, not a generic length heuristic,
# so matching stays precise rather than accidentally over-matching.
KNOWN_SUFFIXES = {
    # Bengali case/possessive/demonymic/locative suffixes
    "র", "ের", "এর", "ই", "রা", "তে", "কে", "য়", "টা", "টি", "ায়",
    # Nepali case/possessive/locative/ergative suffixes
    "को", "मा", "ले", "लाई", "बाट", "हरू", "बाट",
}


def normalize_text(text: str) -> str:
    """Normalize Unicode, case, whitespace, and lightweight punctuation."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("।", " ")
    normalized = re.sub(r"[^\w\s\u0900-\u097F\u0980-\u09FF]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


NORMALIZED_ALIASES = {
    canonical: {normalize_text(alias) for alias in aliases}
    for canonical, aliases in ALIASES.items()
}


def strip_known_suffix(token: str) -> str | None:
    """
    If `token` is `alias + known_suffix` for some known suffix, return
    the token with that suffix removed. Otherwise return None.
    """
    for suffix in KNOWN_SUFFIXES:
        if len(token) > len(suffix) and token.endswith(suffix):
            return token[: -len(suffix)]
    return None


def build_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Answer with only the city name. "
        "Do not explain, add dates, or provide any other information."
    )


def api_call(model: str, temperature: float, prompt: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": build_prompt(prompt)}],
        options={
            "temperature": temperature,
            "top_p": CONFIG["TOP_P"],
            "num_predict": CONFIG["MAX_TOKENS"],
        },
    )

    raw_text = response["message"]["content"]
    return raw_text.replace("<|END_RESPONSE|>", "").strip()


def extract_canonical_answer(response: str | None) -> str | None:
    """
    Map a short generated answer to a canonical answer.

    Tries, in order:
      1. Exact match against known aliases.
      2. Exact match after stripping a known grammatical suffix
         (handles inflected forms like ঢাকার, ঢাকাই, काठमाडौंमा).
      3. Token-level match (for slightly longer outputs).

    Returns:
        "kathmandu", "dhaka", or None.
    """
    if not response:
        return None

    normalized_response = normalize_text(response)

    # Tier 1: exact match
    for canonical, aliases in NORMALIZED_ALIASES.items():
        if normalized_response in aliases:
            return canonical

    # Tier 2: suffix-stripped match — catches inflected forms
    stripped = strip_known_suffix(normalized_response)
    if stripped:
        for canonical, aliases in NORMALIZED_ALIASES.items():
            if stripped in aliases:
                return canonical

    # Tier 3: token-level fallback for outputs like "the answer is Kathmandu"
    tokens = normalized_response.split()
    for token in tokens:
        for canonical, aliases in NORMALIZED_ALIASES.items():
            if token in aliases:
                return canonical
            token_stripped = strip_known_suffix(token)
            if token_stripped and token_stripped in aliases:
                return canonical

    return None


def correctness_check(
    response: str | None,
    expected_canonical: str,
) -> tuple[bool, str | None]:
    extracted = extract_canonical_answer(response)
    return extracted == expected_canonical, extracted


def entropy(labels: list[str]) -> float:
    """Shannon entropy in bits."""
    if not labels:
        return float("nan")

    counts = Counter(labels)
    total = len(labels)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def get_active_prompts() -> list[dict[str, Any]]:
    if CONFIG["LANGUAGES"] is None:
        return PROMPTS
    return [p for p in PROMPTS if p["lang"] in CONFIG["LANGUAGES"]]


def run_experiment() -> list[dict[str, Any]]:
    results = []
    active_prompts = get_active_prompts()

    for prompt_data in active_prompts:
        print(f"Running experiment for prompt: {prompt_data['id']}")

        result_entry = {
            "prompt_id": prompt_data["id"],
            "fact_id": prompt_data["fact_id"],
            "language": prompt_data["lang"],
            "original_prompt": prompt_data["text"],
            "expected_canonical": prompt_data["expected_canonical"],
            "responses": [],
        }

        for model in CONFIG["MODELS"]:
            print(f"  Testing model: {model}")

            model_results = {
                "model": model,
                "samples": [],
            }

            for i in range(CONFIG["N_RESAMPLES"]):
                print(f"    Resample {i + 1}/{CONFIG['N_RESAMPLES']}")
                start_time = time.perf_counter()

                try:
                    response = api_call(
                        model=model,
                        temperature=CONFIG["TEMPERATURE"],
                        prompt=prompt_data["text"],
                    )

                    is_correct, extracted_answer = correctness_check(
                        response=response,
                        expected_canonical=prompt_data["expected_canonical"],
                    )
                    error = None

                except Exception as exc:
                    response = None
                    extracted_answer = None
                    is_correct = None
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"      ERROR on resample {i}: {error}")

                latency = time.perf_counter() - start_time

                model_results["samples"].append(
                    {
                        "resample_id": i,
                        "response": response,
                        "extracted_answer": extracted_answer,
                        "is_correct": is_correct,
                        "latency": latency,
                        "error": error,
                    }
                )

            result_entry["responses"].append(model_results)

        results.append(result_entry)

    with open(CONFIG["OUTPUT_FILE"], "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(
        "Experiment completed. Results saved to",
        CONFIG["OUTPUT_FILE"],
    )

    return results


def summarize(results: list[dict[str, Any]]) -> None:
    print("\n=== SUMMARY ===")

    for entry in results:
        print(
            f"\nPrompt: {entry['prompt_id']} "
            f"({entry['language']}) — expected: "
            f"{entry['expected_canonical']}"
        )

        for model_result in entry["responses"]:
            samples = model_result["samples"]
            valid_samples = [
                sample for sample in samples
                if sample["error"] is None
            ]

            n_valid = len(valid_samples)
            n_correct = sum(
                sample["is_correct"] is True
                for sample in valid_samples
            )
            n_errors = len(samples) - n_valid

            accuracy = (
                n_correct / n_valid
                if n_valid
                else float("nan")
            )

            answer_labels = [
                sample["extracted_answer"] or "unparseable"
                for sample in valid_samples
            ]

            answer_counts = Counter(answer_labels)
            answer_entropy = entropy(answer_labels)

            modal_answer, modal_count = (
                answer_counts.most_common(1)[0]
                if answer_counts
                else ("none", 0)
            )

            agreement = (
                modal_count / n_valid
                if n_valid
                else float("nan")
            )

            average_latency = (
                sum(sample["latency"] for sample in valid_samples) / n_valid
                if n_valid
                else float("nan")
            )

            print(f"  Model: {model_result['model']}")
            print(
                f"    accuracy={accuracy:.2f} "
                f"({n_correct}/{n_valid})"
            )
            print(f"    errors={n_errors}")
            print(f"    modal_answer={modal_answer}")
            print(f"    agreement={agreement:.2f}")
            print(f"    entropy_bits={answer_entropy:.3f}")
            print(f"    avg_latency={average_latency:.2f}s")
            print(f"    distribution={dict(answer_counts)}")

            # Show which "unparseable" responses were actually near-misses
            # worth a human look (e.g. inflected forms the suffix list missed)
            unparseable_examples = [
                s["response"] for s in valid_samples
                if s["extracted_answer"] is None
            ]
            if unparseable_examples:
                preview = unparseable_examples[:5]
                print(f"    unparseable examples: {preview}")


if __name__ == "__main__":
    experiment_results = run_experiment()
    summarize(experiment_results)