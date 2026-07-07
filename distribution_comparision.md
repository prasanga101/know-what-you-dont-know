# Phase 1 & Passage-Retrieval Experiment: Results Summary

## 1. Phase 1 Pilot Recap

A pilot resampling study was conducted on 250 parallel Nepali/English question-answer
pairs (Belebele-derived), with 15 resamples per question per language (7,500 trials
total) using Tiny Aya Fire via Ollama.

**Aggregate accuracy:**
- Nepali: 940/3,750 = 25.07%
- English: 1,678/3,750 = 44.75%
- English advantage: +19.68 percentage points

**Threshold-based action buckets** (Nepali accuracy ≥0.70 → DIRECT; Nepali ≤0.30 and
English ≥0.60 → TRANSLATE; both ≤0.30 → UNRESOLVED; otherwise → BORDERLINE):

| Bucket | Questions | % |
|---|---|---|
| DIRECT | 21 | 8.4% |
| TRANSLATE | 63 | 25.2% |
| UNRESOLVED | 91 | 36.4% |
| BORDERLINE | 75 | 30.0% |

**Manual review** of a stratified 40-item sample (20 UNRESOLVED, 10 BORDERLINE,
5 TRANSLATE, 5 DIRECT) found:

- 27 of 40 items initially marked "answerable without the source passage" were
  reclassified as passage-dependent on native-speaker review — the large majority of
  UNRESOLVED/BORDERLINE failures are evidence gaps, not knowledge gaps.
- 6 confirmed dataset defects: 3 translation errors (mistranslated distractor options)
  and 2 gold-label errors (source text contradicted the labeled correct answer),
  identified by direct comparison against the original English source articles.
- These 6 defects (of 40 sampled, 15%) were excluded from further analysis; this is
  treated as a lower-bound estimate of the defect rate across the full 250-item set,
  since only 16% of items received manual review.

This motivated the central hypothesis: most UNRESOLVED failures reflect **missing
evidence** (the passage was never shown to the model) rather than missing knowledge,
which argues for a RETRIEVE action distinct from ABSTAIN in the five-way policy.

## 2. Passage-Retrieval Experiment

**Design:** the 166 questions in the UNRESOLVED and BORDERLINE buckets were re-run
with the correct source passage prepended to the prompt, using the same 15-resamples
per question per language design (4,980 trials total; 0 errors; 14 unparsed
responses, ~0.3%).

**Classification rule:** for each question/language pair with no-passage accuracy
≤0.30 (i.e., previously failing):
- With-passage accuracy ≥0.60 → **RETRIEVE** (evidence gap resolved by retrieval)
- With-passage accuracy ≤0.30 → **STILL_FAILS** (not resolved by retrieval alone)
- Otherwise → **PARTIAL_IMPROVEMENT**

**Results:**

| Language | RETRIEVE | STILL_FAILS | PARTIAL_IMPROVEMENT | (was not failing) |
|---|---|---|---|---|
| Nepali | 15 (13.9%) | 78 (72.2%) | 15 (13.9%) | 58 |
| English | 80 (70.2%) | 27 (23.7%) | 7 (6.1%) | 52 |

*(Percentages are of the 108 Nepali / 114 English questions that were failing
pre-passage.)*

## 3. The Nepali Comprehension Gap

The headline asymmetry — retrieval resolves ~70% of English failures but only ~14%
of Nepali failures — raised the question of whether this reflects a genuine model
limitation or a data-quality artifact (corrupted/misaligned Nepali passages).

A manual spot-check of 12 of the 37 cases where Nepali stayed at/near 0% accuracy
while English reached ~100% with the same passage found:

- **2 confirmed data-corruption cases**: the Nepali passage contained a sentence
  entirely unrelated to the article (apparently spliced from a different source),
  removing the specific evidence needed to answer the question.
- **1 case of minor translation distortion**, still broadly usable.
- **~9 cases of complete, accurate Nepali translations** where the model still failed
  despite valid, correct evidence being present in the prompt.

Extrapolating this ~17% corruption/distortion rate (2-3 of 12) across the full
37-item anomalous set suggests the large majority (an estimated 30+ of 37, ~80%+)
are genuine cases where the model was given correct Nepali evidence and still could
not extract the correct answer — a real, replicable comprehension/extraction
limitation specific to Nepali, not an artifact of translation quality.

**This finding is a caveat, not a full audit**: only 12 of 37 anomalous items and 40
of 250 total pilot items received direct manual verification. The corruption-rate
estimate should be reported as indicative, not exhaustive.

## 4. Implication for the Five-Way Action Policy

The original RETRIEVE/ABSTAIN framing assumed failure decomposes cleanly into:
- **Evidence gap** → passage retrieval resolves it → RETRIEVE
- **Knowledge gap** → passage retrieval does not resolve it → ABSTAIN

The passage-retrieval results show a third, populous case that this framing does not
cleanly capture: **evidence is present and correct, but the model still fails** —
observed in roughly 70-80+ of the 108 previously-failing Nepali questions. This is
neither a pure evidence gap (retrieval was performed) nor obviously a knowledge gap in
the traditional sense (the fact was available in the prompt).

This suggests Layer 2's policy needs to account for **language-conditioned retrieval
efficacy**, not just retrieval availability — e.g., a question may warrant RETRIEVE
in English but ABSTAIN (or ESCALATE) in Nepali even when the same underlying passage
is available in both languages, because the model's ability to *use* retrieved
Nepali evidence is measurably weaker than its ability to use English evidence.

**Open methodological question carried forward:** whether this should be modeled as
a variant of ABSTAIN ("retrieval attempted, evidence present, still insufficient")
or as a distinct fifth outcome the Layer 2 classifier must learn to distinguish from
both clean RETRIEVE and clean ABSTAIN cases.