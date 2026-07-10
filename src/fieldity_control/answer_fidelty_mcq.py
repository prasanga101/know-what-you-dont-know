import re

LETTER_TO_NUM = {
    "A": "1",
    "B": "2",
    "C": "3",
    "D": "4",
}

OPTION_TEXT_TO_NUM = {
    "A)": "1",
    "B)": "2",
    "C)": "3",
    "D)": "4",
    "A.": "1",
    "B.": "2",
    "C.": "3",
    "D.": "4",
}

def extract_passage_mcq(response):
    if not response:
        return None

    text = response.strip().upper()

    # Exact single-letter response
    if text in LETTER_TO_NUM:
        return LETTER_TO_NUM[text]

    # Strong option patterns: "A)", "B.", "**C)**", "Answer: D"
    match = re.search(r"(?:^|[^A-Z])([ABCD])\s*[\)\.:\-]", text)
    if match:
        return LETTER_TO_NUM[match.group(1)]

    # Patterns like "ANSWER IS B", "CORRECT OPTION D", "सही उत्तर B"
    match = re.search(
        r"(?:ANSWER|OPTION|CORRECT|JAWAF|JAVAF|उत्तर|विकल्प|जवाफ)[^ABCD]{0,40}\b([ABCD])\b",
        text
    )
    if match:
        return LETTER_TO_NUM[match.group(1)]

    # Parenthesized answer: "(B)"
    match = re.search(r"\(([ABCD])\)", text)
    if match:
        return LETTER_TO_NUM[match.group(1)]

    return None