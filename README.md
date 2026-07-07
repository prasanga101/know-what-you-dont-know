# LRCal-Agent

LRCal-Agent studies whether a low-resource-language LLM system can recognize when it should answer, translate, retrieve evidence, abstain, or escalate instead of answering confidently and incorrectly.

The current project has advanced from setup into a completed Phase 1 no-passage pilot and a Phase 2 passage-retrieval experiment on parallel English/Nepali Belebele multiple-choice questions. The key finding so far is that retrieval behaves very differently by language: adding the correct passage resolves most English failures, but only a small minority of Nepali failures.

## Research Question

Can an LLM agent learn a calibrated action policy for low-resource-language QA?

The target action space is:

- `ANSWER`: answer directly when the model is reliable
- `TRANSLATE`: translate or route through a higher-resource language when the model knows the answer in English but not in Nepali
- `RETRIEVE`: fetch missing evidence when the question is answerable only with context
- `ABSTAIN`: decline when the answer remains unsupported or unreliable
- `CLARIFY` / `ESCALATE`: defer when the system cannot safely classify the failure mode

The project is not just measuring accuracy. It is building evidence for a trained policy layer that distinguishes different kinds of failure.

## Core Idea

LRCal-Agent separates the system into two layers:

| Layer | Role | Status |
|---|---|---|
| Layer 1: Calibration signal | Resample the base model many times and measure accuracy, answer agreement, entropy, and failure patterns | Implemented for MCQ pilots |
| Layer 2: Action policy | Learn which action should be taken from calibration and trajectory features | Research target; current phases are generating labels/evidence |

The important methodological shift from the recent work is that failure is not binary. A wrong answer can come from:

- missing evidence
- language-specific access/comprehension failure
- corrupted or mistranslated data
- a true unanswerable/unsupported case
- model instability or confident wrongness

## Current Advancement Summary

### Phase 0: Setup and Model Validation

Completed.

This phase established the local model and scoring pipeline:

- selected Tiny Aya Fire via Ollama as the working local multilingual model
- confirmed English vs Nepali/Bengali performance gaps
- observed confidently wrong answers and script-drift failures
- built MCQ answer extraction for A/B/C/D responses
- fixed early false-negative scoring issues caused by language/script variation
- prepared cleaned datasets under `data/processed/`

### Phase 1: No-Passage English/Nepali Pilot

Completed.

A 250-item paired English/Nepali Belebele sample was created. The passage was stripped, and each item was asked in both languages with 15 stochastic resamples per language.

Total trials:

- 250 questions
- 2 languages: Nepali and English
- 15 resamples per question/language
- 7,500 no-passage trials

Aggregate no-passage accuracy:

| Language | Correct / Total | Accuracy |
|---|---:|---:|
| Nepali | 940 / 3,750 | 25.07% |
| English | 1,678 / 3,750 | 44.75% |

Threshold buckets from `data/pilot1/questions_bucket.json`:

| Bucket | Rule | Count | Share |
|---|---|---:|---:|
| `DIRECT` | Nepali accuracy >= 0.70 | 21 | 8.4% |
| `TRANSLATE` | Nepali <= 0.30 and English >= 0.60 | 63 | 25.2% |
| `UNRESOLVED` | Nepali <= 0.30 and English <= 0.30 | 91 | 36.4% |
| `BORDERLINE` | Everything else | 75 | 30.0% |

The first interpretation was that `TRANSLATE` represented language-specific gaps, while `UNRESOLVED` represented likely unanswerable or missing-knowledge cases. Manual review complicated that story.

### Manual Review Finding

A stratified 40-item sample was reviewed from the Phase 1 buckets:

- 20 `UNRESOLVED`
- 10 `BORDERLINE`
- 5 `TRANSLATE`
- 5 `DIRECT`

The review found that many failures were not pure knowledge gaps. In the comparison note, 27 of 40 items initially treated as answerable without the source passage were reclassified as passage-dependent after review. That shifted the hypothesis: many failures were likely evidence gaps because the passage had been removed.

The review also identified dataset issues:

- mistranslated distractor options
- gold-label problems where the source text contradicted the labeled answer
- passage/question alignment concerns

These artifacts should be excluded or tracked separately before treating bucket labels as clean policy labels.

### Phase 2: Passage-Retrieval Experiment

Completed.

The 166 questions from the `UNRESOLVED` and `BORDERLINE` buckets were re-run with the correct source passage prepended to the prompt.

Total with-passage trials:

- 166 questions
- 2 languages
- 15 resamples per question/language
- 4,980 trials

Classification rule in `src/passage_phase/compare_passage_effect.py`:

| Condition | Label |
|---|---|
| no-passage accuracy > 0.30 | `WAS_NOT_FAILING` |
| no-passage <= 0.30 and with-passage >= 0.60 | `RETRIEVE` |
| no-passage <= 0.30 and with-passage <= 0.30 | `STILL_FAILS` |
| otherwise | `PARTIAL_IMPROVEMENT` |

Results from `data/passage1/retrieval_comparison.json`:

| Language | RETRIEVE | STILL_FAILS | PARTIAL_IMPROVEMENT | WAS_NOT_FAILING |
|---|---:|---:|---:|---:|
| Nepali | 15 | 78 | 15 | 58 |
| English | 80 | 27 | 7 | 52 |

Among previously failing items only:

| Language | Previously failing | RETRIEVE | STILL_FAILS | PARTIAL_IMPROVEMENT |
|---|---:|---:|---:|---:|
| Nepali | 108 | 15 (13.9%) | 78 (72.2%) | 15 (13.9%) |
| English | 114 | 80 (70.2%) | 27 (23.7%) | 7 (6.1%) |

## Main Research Finding So Far

The passage experiment shows a major asymmetry:

- English failures are often evidence gaps. When the correct passage is supplied, about 70% of previously failing English cases recover.
- Nepali failures are usually not resolved by supplying the same evidence. Only about 14% of previously failing Nepali cases become clean `RETRIEVE` cases.

This suggests the policy cannot treat retrieval as language-neutral. For Nepali, the model often receives the correct passage but still cannot reliably extract the answer.

That means the action policy may need to distinguish:

- `RETRIEVE`: evidence missing, retrieval likely fixes the problem
- `ABSTAIN` or `ESCALATE`: evidence present, but the model still cannot use it reliably
- `TRANSLATE`: English recovers where Nepali does not

## Project Layout

```text
LRCal-Agent/
  README.md
  fire_api_call/
    src.py                         # Ollama model-call helper
  src/
    pilot_phase/
      sample_items.py              # Build 250 paired EN/NE sample
      build_mcq.py                 # Build no-passage MCQ prompt
      extract_answer_mcq.py        # Extract A/B/C/D model answer
      resample_mcq.py              # Run 15 no-passage resamples
      distribution.py              # Accuracy, entropy, agreement, buckets
      review_sample.py             # Stratified manual-review sample
    passage_phase/
      sample_passage_items.py      # Reattach source passages
      filter_unresolved_items.py   # Keep UNRESOLVED/BORDERLINE items
      build_passage_mcq.py         # Build with-passage MCQ prompt
      answer_passage_mcq.py        # Extract with-passage answer
      resample_unresolved_pmcq.py  # Run with-passage resamples
      compare_passage_effect.py    # Compare no-passage vs with-passage
    tests/
      distribution_comparision.md  # Current experiment summary and interpretation
  data/
    processed/                     # Cleaned dataset caches
    pilot1/                        # Phase 1 samples, trials, buckets, review sample
    passage1/                      # Phase 2 passage samples and retrieval comparison
    loader/                        # Dataset loading helpers
    data_cleaner/                  # Dataset-specific cleaners
  results/                         # Early resampling sanity checks
  phases/                          # Phase diagrams
```

## Important Data Artifacts

| File | Meaning |
|---|---|
| `data/pilot1/sample_items.json` | 250 paired English/Nepali Belebele items |
| `data/pilot1/resampling_results.json` | 7,500 no-passage trials |
| `data/pilot1/questions_bucket.json` | Phase 1 threshold bucket assignment |
| `data/pilot1/review_sample.json` | Stratified 40-item manual-review file |
| `data/passage1/sample_passage_items.json` | Same items with source passages attached |
| `data/passage1/unresolved_sample.json` | 166 `UNRESOLVED`/`BORDERLINE` items selected for Phase 2 |
| `data/passage1/resampled_passage_output.json` | 4,980 with-passage trials |
| `data/passage1/retrieval_comparison.json` | No-passage vs with-passage classification |
| `src/tests/distribution_comparision.md` | Human-readable results summary and interpretation |

## Reproducing The Pipeline

Run commands from the repository root.

### Phase 1: No-Passage Pilot

```bash
python -m src.pilot_phase.sample_items
python -m src.pilot_phase.resample_mcq
python -m src.pilot_phase.distribution
python -m src.pilot_phase.review_sample
```

### Phase 2: Passage-Retrieval Experiment

```bash
python -m src.passage_phase.sample_passage_items
python -m src.passage_phase.filter_unresolved_items
python -m src.passage_phase.resample_unresolved_pmcq
python -m src.passage_phase.compare_passage_effect
```

The scripts assume Ollama is available locally and that the configured model can be called through `fire_api_call/src.py`.

Current model constant used by the resampling scripts:

```text
hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M
```

## Current Interpretation

The project has moved from "can resampling expose low-resource unreliability?" to a sharper finding:

> Evidence retrieval helps English much more than Nepali, even when both languages receive the corresponding source passage.

This matters for LRCal-Agent because a policy trained only on no-passage confidence could mislabel many Nepali cases as retrieval candidates. The Phase 2 results suggest the policy needs features that capture language-conditioned retrieval efficacy, not just whether retrieval is available.

## Next Steps

Near-term:

1. Cleanly separate dataset artifacts from genuine model failures.
2. Expand manual verification of Nepali `STILL_FAILS` cases where English becomes `RETRIEVE`.
3. Decide whether "retrieval attempted but still unusable" should map to `ABSTAIN`, `ESCALATE`, or a distinct policy label.
4. Build a policy-label table from Phase 1 and Phase 2 outputs.
5. Add Bengali and BanglaRQA abstention data back into the action-policy framing.

Later:

1. Integrate translation as an actual tool path.
2. Compare English vs Hindi as recovery targets for Nepali `TRANSLATE` candidates.
3. Train the Layer 2 policy module using calibration features, language, retrieval status, and action labels.
4. Evaluate whether the trained policy outperforms fixed threshold rules.

## Notes On Reliability

The current findings are strong enough to guide the next research phase, but they are not a full dataset audit.

Known caveats:

- only a subset of items received manual review
- some Nepali passages may be corrupted or misaligned
- some gold labels and distractors have documented defects
- MCQ letter extraction is intentionally simple and should be stress-tested before larger runs
- current thresholds are research heuristics, not final policy boundaries

These caveats are part of the research contribution: LRCal-Agent is explicitly trying to learn when model behavior is unsupported, unstable, or language-conditionally unreliable.
