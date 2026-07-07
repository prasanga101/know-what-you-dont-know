def build_mcq_prompt(question, options):
    labels = ["A", "B", "C", "D"]
    formatted_lines = []
    for i in range(4):
        line = labels[i] + ") " + options[i]
        formatted_lines.append(line)

    options_block = "\n".join(formatted_lines)

    prompt = (
        question + "\n\n"
        + options_block + "\n\n"
        + "Answer with only the letter (A, B, C, or D). Do not explain."
    )
    return prompt

if __name__ == "__main__":
    question = "Which of the following will not be held in Beijing in 2022?"
    options = ["Opening ceremonies", "Taizicheng ski area events", "Closing ceremonies", "Indoor ice events"]
    print(build_mcq_prompt(question, options))