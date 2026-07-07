import json
import os

def build_passage_mcq(question, options, passage):
    LABELS =["A","B","C","D"]
    formatted_lines=[]
    for i in range (4):
        lines = LABELS[i] + ") " + options[i]
        formatted_lines.append(lines)
        
        option_block = "\n".join(formatted_lines)
        
        prompt = (
        passage + "\n\n"+
        question + "\n\n"
        + option_block + "\n\n"
        + "Answer with only the letter (A, B, C, or D). Do not explain."
    )
    return prompt
    


if __name__ == "__main__":
    passage ="hello hello hello"
    question = "Which of the following will not be held in Beijing in 2022?"
    options = ["Opening ceremonies", "Taizicheng ski area events", "Closing ceremonies", "Indoor ice events"]
    print(build_passage_mcq(question, options,passage))