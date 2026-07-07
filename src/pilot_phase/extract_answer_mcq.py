import re
LETTER_TO_NUM={
    "A":"1",
    "B":"2",
    "C":"3",
    "D":"4"
}
def extract_mcq_response(response):
    if not response:
        return None
    else:
        normalize_res = response.strip().upper()
        
        if normalize_res in LETTER_TO_NUM:
            return LETTER_TO_NUM[normalize_res]
        if normalize_res and normalize_res[0] in LETTER_TO_NUM:
            return LETTER_TO_NUM[normalize_res[0]]
        match = re.search(r'\b([ABCD])\b', normalize_res)
        if match:
            return LETTER_TO_NUM[match.group(1)]

    return None

if __name__ == "__main__":
    print(extract_mcq_response("B)"))