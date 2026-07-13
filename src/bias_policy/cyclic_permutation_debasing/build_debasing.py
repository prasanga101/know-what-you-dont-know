import json
import os
import re
from collections import Counter

INPUT_FILE = "data/infra_check/ollama_logprob_probe_results.json"
OUTPUT_FILE = "data/infra_check/debasic_ollama_log.json"

with open(INPUT_FILE, encoding="utf-8") as f:
    log_items = json.load(f)

OPTION_LINE_RE = re.compile(
    r"^([ABCD])\)\s*(.+)$",
    re.MULTILINE
)

LETTERS = ["A", "B", "C", "D"]


def extract_options(prompt):
    matches = OPTION_LINE_RE.findall(prompt)

    options = {
        letter: text
        for letter, text in matches
    }

    if set(options) != {"A", "B", "C", "D"}:
        raise ValueError(
            f"Expected exactly 4 options, got {options}"
        )

    return options


def rotate_options(shift, options):
    original_values = []

    # Get the actual option text in A, B, C, D order.
    for letter in LETTERS:
        original_values.append(options[letter])

    rotated_values = []

    # Find which old option should enter each new position.
    for new_pos in range(4):
        old_pos = (new_pos - shift) % 4
        value = original_values[old_pos]
        rotated_values.append(value)

    # Connect the rotated texts back to A, B, C, and D.
    rotated_options = {}

    for index in range(4):
        letter = LETTERS[index]
        rotated_options[letter] = rotated_values[index]

    return rotated_options


def rebuild_prompt(original_prompt, rotated_options):
    new_prompt = original_prompt

    for letter in LETTERS:
        pattern = rf"^{letter}\)\s*.+$"
        new_line = f"{letter}) {rotated_options[letter]}"

        # Use a lambda replacement instead of a plain string — re.subn
        # scans string replacements for backreference syntax like \1,
        # which could misfire if option text ever contains a backslash.
        # A lambda is used literally, no escaping risk.
        new_prompt, replacement_count = re.subn(
            pattern,
            lambda m: new_line,
            new_prompt,
            count=1,
            flags=re.MULTILINE
        )

        if replacement_count != 1:
            raise ValueError(f"expected to replace exactly 1 line for {letter}, replaced {replacement_count}")

    return new_prompt


def track_new_correct_letter(correct_letter, shift):
    orig_index = LETTERS.index(correct_letter)
    new_index = (orig_index + shift) % 4
    return LETTERS[new_index]


# --- validate extraction + rotation on every item before building anything ---
for i, item in enumerate(log_items):
    options = extract_options(item["prompt"])

    for shift in range(4):
        rotated = rotate_options(shift, options)

        assert set(rotated.keys()) == {"A", "B", "C", "D"}, \
            f"[{i}] shift={shift} missing a letter"
        assert Counter(rotated.values()) == Counter(options.values()), \
            f"[{i}] shift={shift} lost or duplicated content"
        if shift == 0:
            assert rotated == options, \
                f"[{i}] shift=0 should equal original, got {rotated}"

print(f"validated rotation on all {len(log_items)} items — OK")

# --- build the full 48-record export: rebuilt prompts + tracked correct letters ---
result = []

for item in log_items:
    options = extract_options(item["prompt"])
    correct_letter = item["correct_answer_letter"]

    for shift in range(4):
        rotated = rotate_options(shift, options)
        rotated_prompt = rebuild_prompt(item["prompt"], rotated)
        rotated_correct_letter = track_new_correct_letter(correct_letter, shift)

        result.append({
            "item_id": item["item_id"],
            "link": item["link"],
            "question_number": item["question_number"],
            "classification": item["classification"],
            "shift": shift,
            "original_correct_letter": correct_letter,
            "rotated_correct_letter": rotated_correct_letter,
            "rotated_options": rotated,
            "rotated_prompt": rotated_prompt,
        })

assert len(result) == 48, f"expected 48 records, got {len(result)}"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"saved {len(result)} rotated prompts to {OUTPUT_FILE}")