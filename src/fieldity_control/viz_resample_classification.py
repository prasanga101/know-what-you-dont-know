import json
import os
import matplotlib.pyplot as plt
from pprint import pprint
gen_fidelity_counts = []
with open("data/fidelity_control1/phase3_classification.json" , encoding="utf-8") as f:
    item_classification = json.load(f)

pprint(item_classification[:1])
for items in item_classification:
    classification = items["classification"]
    correct_ans_letter = items["correct_answer_letter"]
    correct_count = items["correct_count"]
    item_id = items["item_id"]
    link = items["link"]
    question_number = items["question_number"]
    if classification == "GENERATION_FIDELITY_ISSUE":
        gen_fidelity_counts.append(correct_count)
    
bins = range(0, 9)  # correct_count ranges 0-7 for a failed item (0-15 possible, but <=7 = fail)

plt.hist(gen_fidelity_counts, bins=bins, align='left', rwidth=0.8, edgecolor='black')
plt.xlabel("correct_count (out of 15 resamples)")
plt.ylabel("number of items")
plt.title("Distribution of correct_count within GENERATION_FIDELITY_ISSUE items (n=48)")
plt.xticks(range(0, 8))
plt.savefig("gen_fidelity_correct_count_hist.png")
plt.show()

print(len(gen_fidelity_counts)) 