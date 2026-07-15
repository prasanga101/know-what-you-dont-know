# LRCal-Agent

LRCal-Agent studies whether a low-resource-language LLM system can recognize when it should answer, translate, retrieve evidence, abstain, or escalate instead of answering confidently and incorrectly.

The current project has advanced from setup through a completed Phase 1 no-passage pilot, a Phase 2 passage-retrieval experiment, a Phase 3 fidelity-control experiment, and a completed Phase 4 bias-aware policy formalization and Layer 2 groundwork pass, on parallel English/Nepali Belebele multiple-choice questions. Key findings so far: retrieval behaves very differently by language (adding the correct passage resolves most English failures but only a small minority of Nepali failures); among the Nepali items retrieval still can't fix, most turn out to be generation-fidelity failures rather than comprehension gaps — the model often can't reliably state the correct answer even when it is handed to it directly; and once position bias is controlled for as strictly as possible (cyclic-permutation debasing), only a third of the remaining ambiguous "passes" hold up as genuine comprehension. Phase 4 formalizes all of this into a concrete `TRANSLATE`/`RETRIEVE`/`REPAIR`/`ABSTAIN` action-policy label per item, builds trajectory-metric features from that label set, and trains a first Layer 2 classifier against it — finding that Layer-1 confidence/self-consistency signals carry a small but real signal above a dummy baseline, but are not yet sufficient on their own to reliably drive the 4-way action decision.

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
| Layer 2: Action policy | Learn which action should be taken from calibration and trajectory features | Fixed-rule hierarchy implemented (`policy_formulation.py`); first trained classifier built and validated (`train_agent/main.py`) — macro-F1 ~0.28 vs. a 0.18 dummy floor, a real but modest signal, not yet a reliable 4-way decision-maker |

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

### Phase 3: Fidelity-Control Experiment

Completed.

Phase 2 left an open question: for the 78 Nepali `STILL_FAILS` items — cases where the model had the correct passage and still failed — is that a genuine comprehension gap, or an inability to reliably generate/select the right answer even when comprehension isn't the issue? Phase 3 isolates the two.

**Design.** Each of the 78 `STILL_FAILS` Nepali items (from `data/passage1/retrieval_comparison.json`, pulled via `src/fieldity_control/failing_items.py`) was manually audited and given an `english_hint` (an English-language paraphrase of the correct answer) and an `audit_label`:

| `audit_label` | Count | Meaning |
|---|---:|---|
| `VALID_EVIDENCE_MODEL_FAILURE` | 63 | Clean item, evidence is valid |
| `AWKWARD_BUT_USABLE_TRANSLATION` | 10 | Translation is rough but usable |
| `AMBIGUOUS_ITEM` | 3 | Excluded — item itself is ambiguous |
| `CORRUPTED_EVIDENCE` | 2 | Excluded — input data is compromised |

`src/fieldity_control/build_fieldity_control.py` then builds a "control prompt" per item: the passage, question, options, and the English hint, asking the model to answer with a single letter. If the model still can't answer correctly when handed the answer, the failure is generation fidelity, not comprehension.

Each control prompt was resampled 15 times against Tiny Aya Fire at temperature 0.7 (`src/fieldity_control/resample_fidelity_control.py`, graded with `src/fieldity_control/answer_fidelty_mcq.py`) — 78 x 15 = 1,170 calls, logged to `data/fidelity_control1/resampled_fidelity_control_output.json`. The answer-extraction regex was tightened partway through the run to require punctuation or an answer-keyword near the letter (not just any bare A-D character); re-validated retroactively against the same raw responses, the corrected/total count held at 472/1,170, confirming the fix only reclassified previously-wrong guesses as unparseable rather than changing genuine correct/incorrect calls.

**Finding 1 — overall accuracy is only 40.3% (472/1,170)** on a task where the model is handed the answer directly. That's far below what a "control" condition should produce, and is the core evidence that generation fidelity, not comprehension, is the dominant failure mode.

**Finding 2 — Tiny Aya Fire has a real position bias toward option A**, confirmed both by raw-response inspection and by letter-conditioned accuracy:

| Correct letter | Correct-answer share | Selected share | Accuracy when this is the correct letter |
|---|---:|---:|---:|
| A | 29.5% | 45.7% | 60.3% |
| B | 19.2% | 13.5% | 29.8% |
| C | 17.9% | 16.8% | 35.2% |
| D | 33.3% | 15.7% | 31.5% |

A is picked far more than its ground-truth share (blind guessing toward A partly masquerades as "correct"), while B, C, and D are under-picked relative to their share and answered correctly far less often when they are the right choice. This means any pass/fail result on an A-correct item is confounded with blind position bias.

**Side investigation — Qwen2.5-1.5B-instruct as a candidate Layer 2 model.** The same 1,170-call battery was run against Qwen. It scored lower overall (345/1,170 = 29.5%) and showed an even more extreme bias, selecting D 77.1% of the time. Its apparent per-bucket "wins" evaporated under a bias-controlled check: splitting the `AWKWARD_BUT_USABLE_TRANSLATION` bucket into D-correct vs. non-D-correct subsets, Qwen scored 71.1% vs. 0.0% respectively — its bucket advantage was fully explained by that bucket's ground-truth letters happening to skew toward D, not genuine capability. No complementary Layer 2 candidate found here; the general takeaway is to never trust a per-bucket model comparison without conditioning on answer position first.

**Classification.** The 5 flagged items (`AMBIGUOUS_ITEM` + `CORRUPTED_EVIDENCE`) were excluded, leaving 73 clean items (`src/fieldity_control/filter_resampled_items.py`, `data/fidelity_control1/filtered_resampled_items.json`). Each item was classified by majority vote (>= 8/15 correct, the only meaningful "more right than wrong" threshold with an odd resample count):

| Classification | Count | Share | Meaning |
|---|---:|---:|---|
| `GENERATION_FIDELITY_ISSUE` | 48 | 65.8% | Failed even with the answer handed to it |
| `COMPREHENSION_GAP` | 13 | 17.8% | Passed, correct letter wasn't A — a real win |
| `PASS_CONFOUNDED_BY_BIAS` | 12 | 16.4% | Passed, but correct letter was A — can't rule out blind bias |

(`data/fidelity_control1/phase3_classification.json`, `data/fidelity_control1/phase3_classification_summary.json`)

**Follow-up refinement.** Breaking the 48 `GENERATION_FIDELITY_ISSUE` items down by raw `correct_count` (0-15, via `src/fieldity_control/viz_resample_classification.py`, plotted in `gen_fidelity_correct_count_hist.png`) shows the failures aren't uniformly "near-miss":

- 18 items (37.5%): near-total failure, 0-2/15 correct even with the answer handed to them
- 16 items (33.3%): weak partial signal, 3-4/15 correct
- 14 items (29.2%): borderline/unstable, 5-7/15 correct

This suggests three different remedies rather than one — near-total failures likely need a translate/retrieve/abstain fallback rather than voting; weak-partial cases need stronger verification against evidence; only the borderline tier is a good candidate for consistency/voting-based repair, and even that needs confirmation that the correct answer is the actual plurality winner across resamples rather than assumed from the count.

![Distribution of correct_count within GENERATION_FIDELITY_ISSUE items (n=48)](gen_fidelity_correct_count_hist.png)

### Follow-up: Logprob-Based Bias Verification (`infra_check`)

Completed.

Phase 3 flagged 12 items as `PASS_CONFOUNDED_BY_BIAS` — cases where the model passed the control task (>= 8/15 correct at temperature 0.7) but the correct letter was A, so the pass couldn't be distinguished from Tiny Aya Fire's position bias toward A. The resample-based vote can't resolve this because sampling noise and bias both push toward the same letter.

`src/bias_policy/infra_check` builds a deterministic check instead of a stochastic one:

- `test_ollama_logprobs.py` pulls the 12 `PASS_CONFOUNDED_BY_BIAS` items out of `data/fidelity_control1/phase3_classification.json` / `resampled_fidelity_control_output.json` into `data/infra_check/ollama_logprobs.json`.
- `build_logprobs.py` dedupes to one prompt per item (12 unique) and wraps each in a single-letter-answer template, saved to `data/infra_check/resample_ollama_logprobs.json`.
- `resample_logprobs.py` calls Ollama's `/api/generate` with `logprobs: true`, `top_logprobs: 20`, and `temperature: 0`, extracts the logprobs for A/B/C/D from the first generated token, and converts them to a softmax distribution over just those four options. A sanity invariant is checked on every item: at temperature 0 the model's emitted token is by construction the argmax over the *full* vocabulary, so if it emitted a letter A-D, that letter must also be the argmax within the extracted A/B/C/D subset — this held on all 12/12 items, confirming the logprob extraction is trustworthy. Results are saved to `data/infra_check/ollama_logprob_probe_results.json`.

**Finding.** At temperature 0 (no sampling noise), the model's true top choice (by logprob) matches the correct letter A on only **7 of 12 items (58.3%)**. The other 5 have a different deterministic argmax (D on 2, and B and C on one each), meaning their earlier temperature-0.7 "pass" was resample noise landing on the biased letter, not a genuine model preference for the correct answer. This further shrinks the trustworthy comprehension-gap count from Phase 3: of the 13 `COMPREHENSION_GAP` + 12 `PASS_CONFOUNDED_BY_BIAS` items previously counted as passes, at most 13 + 7 = 20 look genuinely earned once bias is deterministically controlled for, not 25.

### Cyclic-Permutation Debasing (`bias_policy/cyclic_permutation_debasing`)

Completed.

The temperature-0 logprob check confirms *whether* the model's top pick is the correct letter, but it can't tell whether that pick tracks the answer's **content** or just its **position** — a model anchored on "always guess A" and a model that genuinely knows the answer happens to be A would look identical in a single-ordering check. Cyclic-permutation debasing resolves this by moving the content around and watching whether the model's pick follows it.

- `build_debasing.py` takes the 12 `PASS_CONFOUNDED_BY_BIAS` items (`data/infra_check/ollama_logprob_probe_results.json`), extracts the four option texts out of each prompt, and generates all 4 cyclic rotations (shift 0-3, i.e. `ABCD`/`BCDA`/`CDAB`/`DABC` reassigned back onto letters A-D) with the prompt rebuilt around each rotation and the correct letter re-tracked to its new position. 12 items x 4 shifts = 48 rotated prompts, saved to `data/infra_check/debasic_ollama_log.json`. Every rotation is validated (no lost/duplicated option text, shift-0 equals the original) before anything is written.
- `resample_debasing_logprobs.py` queries Ollama at temperature 0 for all 48 rotated prompts, extracts A/B/C/D logprobs the same way as the earlier infra_check, and records which option the model's top logprob lands on for each rotation. The same temperature-0 argmax sanity invariant from the earlier infra_check is re-checked on every record. Saved to `data/infra_check/debasic_ollama_logprob_resampling_result.json`.
- `aggregate_debasing.py` groups the 48 results back by item (4 rows each) and compares the sequence of top-picked options across the 4 rotations against the correct letter's new position each time, producing one verdict per item:

| Verdict | Rule | Meaning |
|---|---|---|
| `permutation_stable` | Top pick matches the (moving) correct letter on all 4/4 rotations | Model tracks the answer's content, not its position — genuine comprehension |
| `position_anchored` | Same letter position picked on >= 3/4 rotations, regardless of content | Model is anchored to a letter slot — pure position bias |
| `complex_interaction` | Neither of the above | Mixed/inconsistent pattern, not cleanly attributable to either |

**Finding.** Of the 12 `PASS_CONFOUNDED_BY_BIAS` items (`data/infra_check/aggregated_debasing.json`):

| Verdict | Count |
|---|---:|
| `permutation_stable` | 4 |
| `position_anchored` | 2 |
| `complex_interaction` | 6 |

Only **4 of 12 (33.3%)** items survive the strongest available bias check — the model's pick follows the correct content through every option reordering. The other 8 (66.7%) either show clear position anchoring or an inconsistent pattern that can't be trusted as genuine comprehension. This narrows the trustworthy pass count from Phase 3/infra_check further: of the 12 `PASS_CONFOUNDED_BY_BIAS` items, only these 4 hold up as real, content-tracking correctness once bias is controlled for as strictly as possible with this method.

### Action-Policy Formalization (`bias_policy/action_policy`)

Completed.

`policy_formulation.py` formalizes the actual decision rule the earlier phases were building evidence for: given an item, which of `TRANSLATE` / `RETRIEVE` / `REPAIR` / `ABSTAIN` should the policy take? It joins three prior outputs — `data/passage1/retrieval_comparison.json` (Phase 2), `data/fidelity_control1/phase3_classification.json` (Phase 3), and `data/infra_check/aggregated_debasing.json` (cyclic-permutation debasing) — and applies a strict priority order per item, using an odd-resample majority threshold of `8/15` throughout:

1. `TRANSLATE` — if the English no-passage accuracy already clears `8/15`, the model knows the answer in English; route through it rather than fixing Nepali directly.
2. `RETRIEVE` — otherwise, if the Nepali with-passage accuracy clears `8/15`, supplying the correct Nepali passage is enough.
3. `REPAIR` — otherwise (evidence alone doesn't resolve it): trust a consistency-based repair only if the item survived the strongest bias check available — `permutation_stable` in the debasing verdict for the 12 items that went through it, or a confirmed `COMPREHENSION_GAP` classification (Phase 3) for items that didn't need debasing at all.
4. `ABSTAIN` — everything else: generation-fidelity failures that don't survive repair, and `PASS_CONFOUNDED_BY_BIAS` items whose debasing verdict came back `position_anchored` or `complex_interaction`.

Run first over the 73 clean Phase 3 items alone, no item clears the `RETRIEVE` bar in Nepali at all (0/73) — direct, mechanical confirmation of the Phase 2/3 finding that Nepali retrieval essentially never resolves a failing item. The script then folds in the Nepali rows from Phase 2 that were never part of the "still failing" set, so the final label set covers every Nepali item that had a clear outcome:

- all 15 `ne` rows Phase 2 already classified `RETRIEVE` are carried through as `RETRIEVE` directly;
- the 15 `ne` `PARTIAL_IMPROVEMENT` rows are re-checked against the same `8/15` bar: 3 clear the with-passage threshold and become `RETRIEVE`, 1 more clears the English no-passage threshold instead and becomes `TRANSLATE`, and the remaining 11 clear neither and are left without an assigned action (a known gap — see Notes On Reliability).

Final label counts (`data/policy_formulation/policy_Action_determination.json`, 92 labeled items):

| Action | Count | Share |
|---|---:|---:|
| `ABSTAIN` | 53 | 57.6% |
| `RETRIEVE` | 18 | 19.6% |
| `REPAIR` | 17 | 18.5% |
| `TRANSLATE` | 4 | 4.3% |

This is the first end-to-end, non-`ANSWER` action label set the project has produced — the 92 labeled items exclude the 58 Nepali items that were never failing in the first place (implicitly `ANSWER`), the 5 items excluded in Phase 3 as `AMBIGUOUS_ITEM`/`CORRUPTED_EVIDENCE`, and the 11 unresolved `PARTIAL_IMPROVEMENT` items noted above.

### Trajectory-Metric Features (`bias_policy/trajectory_features`)

Completed.

Before training a Layer 2 classifier, each of the 92 labeled items needed a feature vector. `trajectory_freature.py` builds one per item, reusing data already on hand plus one new deterministic Ollama call per item on the plain Nepali baseline prompt (the same neutralized-decoding setup validated in `infra_check`):

- **Resample-based features** (zero new model calls) — Shannon entropy and agreement-rate (mode frequency) over each item's 15 baseline Nepali resamples from `data/pilot1/resampling_results.json`, plus the per-option resample distribution, its top-2 margin, and the invalid-response rate.
- **Logprob-based features** (one deterministic call per item, 92 total) — a normalized A/B/C/D probability distribution from the first-token logprobs, its entropy, top-2 margin, and the population variance across the four probabilities (`token_probability_variance`).
- **Cross-signal features** — whether the resample-mode answer and the logprob-argmax answer agree (`resample_logprob_top_match`), and the Jensen–Shannon divergence between the two distributions.

All 92 items ran clean (checkpointed, resumable via `trajectory_feature.checkpoint.json`) — no crashes, no missing-letter extraction failures, 92 unique item keys, results independently re-verified against the printed group-by-action summary.

**Finding.** None of the three headline features (entropy, agreement-rate, token-probability variance) separate the four actions well on their own. Eta-squared (share of variance explained by the action label):

| Feature | Eta-squared |
|---|---:|
| `entropy` | 4.45% |
| `agreement_rate` | 3.68% |
| `token_probability_variance` | 1.07% |

`entropy` and `agreement_rate` are near-redundant (correlation −0.967) — they capture almost the same thing. `token_probability_variance` is only weakly related to either (−0.529 with entropy, +0.449 with agreement) and adds the *least* separating power of the three, not the most — its `TRANSLATE`-bucket mean is inflated by a single outlier. `REPAIR` trends toward higher entropy / lower agreement than the other actions, but all four classes overlap heavily on every feature.

**Conclusion, correctly scoped.** This is a real negative result, not a failed step: Layer-1 confidence/self-consistency signals alone tell you *whether* the model is confident, not *what kind* of corrective action a failure needs. That is the argument for Layer 2 as a distinct, multi-signal system rather than a confidence threshold — whether a trained classifier can actually close that gap is what the next step (E) tests.

**Caution carried into the classifier step.** Candidate signals like retrieval-improvement, EN/NE recovery, fidelity classification, and the permutation verdict already fed into building the action labels themselves (Phase 4C). Using raw versions of them as classifier *inputs* is not strictly circular — C used hard thresholds while the raw values carry more nuance — but it is a real leakage risk that has to be reported explicitly if used, not folded in silently.

Full output: `data/policy_formulation/trajectory_feature.json` (92 records, 17 features each).

### Layer 2 Classifier (`bias_policy/train_agent`)

Completed.

`train_agent/main.py` trains a decision-tree classifier against C's action labels using D's features, and validates it four different ways to stress-test the result before trusting it, addressing the leakage risk flagged above head-on:

- a **dummy baseline** (always predict the majority action, `ABSTAIN`), for a real floor to compare against instead of raw accuracy (`ABSTAIN` alone is 58% of the data);
- a **3-feature tree** (`entropy`, `agreement_rate`, `token_probability_variance` — the D headline features only) under `StratifiedGroupKFold(k=4)` outer validation, grouped by source link (8 of 92 links appear twice and could otherwise leak across train/test), with a nested `GridSearchCV` inner loop pruning over `max_depth`, `min_samples_leaf`, `class_weight`, and `ccp_alpha`, selected only on outer-train data;
- a **16-feature tree** using the full D feature set (raw resample/logprob probability vectors, top-2 margins, entropy, cross-source top-match flag, Jensen–Shannon divergence) under the same grouped/nested-pruned methodology — all sourced from A/B/D's own data, not from C's label-building thresholds, so this version carries no leakage risk.

Out-of-fold results (pooled predictions across all outer folds):

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Dummy (always `ABSTAIN`) | 0.576 | 0.250 | 0.183 |
| Decision tree, 3 features, grouped + nested-pruned | 0.554 | 0.289 | 0.279 |
| Decision tree, 16 features, grouped + nested-pruned | 0.467 | 0.258 | 0.244 |

An earlier, unrestricted 3-feature tree (plain `StratifiedKFold`, no grouping, no pruning) was also run as a sanity check and scored a similar macro-F1 (0.290) despite training-set memorization (training macro-F1 = 1.00, depth 11, 47 leaves fit to ~92 rows) — the close agreement with the properly grouped/pruned run (0.279) is itself informative: it shows the headline result isn't an artifact of overfitting or fold leakage, not that the naive version should be trusted on its own. Only the grouped/nested-pruned versions are kept in the shipped script.

Along the way, a real `scikit-learn` API issue surfaced and was fixed: `make_scorer(f1_score, average="macro", ...)` inherits `f1_score`'s own default `pos_label=1` and validates it against the fitted estimator's string class labels before `average` is ever applied, raising a `ValueError`; fixed by passing a plain scorer callable instead of `make_scorer`.

**Finding.** Macro-F1 stays in a narrow 0.24–0.29 band across every methodology tried, only modestly above the 0.183 dummy floor. Going from 3 to 16 features made results *worse*, not better (0.279 → 0.244) — most likely because splitting 92 items across outer and inner folds leaves only ~40–50 rows to learn 16 dimensions from, not enough data for the extra features to pay off. `REPAIR` and `TRANSLATE` remain essentially unresolved in every version (`TRANSLATE`: 0/4 correct in nearly every run; `REPAIR` F1 never exceeds 0.21).

**Conclusion, correctly scoped.** Layer-1 confidence and self-consistency signals carry a small but real signal above chance (confirmed against a proper dummy baseline), but are not sufficient to reliably drive the 4-way corrective-action decision, particularly for the rarer classes (`REPAIR`, `TRANSLATE`). This is a negative-but-legitimate finding, consistent across multiple independent methodologies — not a modeling failure. It motivates either substantially more labeled data for the minority actions, or a fundamentally different signal source (e.g. an "evidence present but unusable" feature, not yet built), rather than further tuning of this feature family.

Full script: `src/bias_policy/train_agent/main.py`.

### Phase 4 Aggregate Report

Completed.

`train_agent/aggregate_report.py` pulls B's debiasing verdict counts, C's action-label counts, D's eta-squared numbers, and E's full accuracy/macro-F1 table into one report — re-running E's evaluation live (importing directly from `main.py`) rather than hardcoding numbers, so the report can't silently drift out of sync with the model code. Output: `data/policy_formulation/phase4_aggregate_report.md`.

## Main Research Finding So Far

The passage experiment shows a major asymmetry:

- English failures are often evidence gaps. When the correct passage is supplied, about 70% of previously failing English cases recover.
- Nepali failures are usually not resolved by supplying the same evidence. Only about 14% of previously failing Nepali cases become clean `RETRIEVE` cases.

This suggests the policy cannot treat retrieval as language-neutral. For Nepali, the model often receives the correct passage but still cannot reliably extract the answer.

Phase 3 sharpens the "model still cannot use it reliably" case further. At minimum 65.8% of Nepali `STILL_FAILS` items are generation-fidelity failures, not comprehension gaps — the model can't reliably produce the correct letter even when the answer is handed to it directly, and Tiny Aya Fire's measurable bias toward option A means even the 16.4% of "passes" tied to a correct answer of A can't be fully trusted as genuine comprehension. This means `RETRIEVE` alone won't fix the majority of `STILL_FAILS` cases: the policy needs a distinct repair/verification mechanism, not a single knowledge-gap label.

That means the action policy may need to distinguish:

- `RETRIEVE`: evidence missing, retrieval likely fixes the problem
- `ABSTAIN` or `ESCALATE`: evidence present, comprehension gap confirmed, model still cannot use it reliably
- `TRANSLATE`: English recovers where Nepali does not
- a repair/verification path for generation-fidelity failures (distinct from `ABSTAIN`): near-total failures likely need a fallback action, weak-partial cases need evidence-grounded verification, and only borderline/unstable cases are candidates for consistency-based repair

Phase 4's D and E steps close the loop on that policy design: once the `TRANSLATE`/`RETRIEVE`/`REPAIR`/`ABSTAIN` labels exist (C), Layer-1 confidence/self-consistency features derived from the same resample and logprob data used everywhere else in the project (D) explain only a small share of the variance between actions, and a trained classifier built on those features tops out at macro-F1 ~0.28 against a 0.18 dummy floor (E) — a real signal, but not close to reliable, especially for the minority `REPAIR`/`TRANSLATE` classes. That is direct, quantified evidence for the project's core claim: confidence alone cannot tell you *which* corrective action a low-resource-language failure needs, only that something is wrong. Layer 2 needs either more labeled data or additional, qualitatively different signal sources beyond resample/logprob confidence.

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
    fieldity_control/
      failing_items.py             # Pull Nepali STILL_FAILS items from Phase 2
      build_fieldity_control.py    # Build control prompts (passage+question+hint)
      answer_fidelty_mcq.py        # Extract A/B/C/D from control-prompt responses
      resample_fidelity_control.py # Run 15 control resamples per item
      filter_resampled_items.py    # Exclude flagged items, classify by majority vote
      viz_resample_classification.py # Histogram of correct_count within failures
    bias_policy/
      infra_check/
        test_ollama_logprobs.py    # Pull PASS_CONFOUNDED_BY_BIAS items from Phase 3
        build_logprobs.py          # Dedupe + build logprob-probe prompts
        resample_logprobs.py       # Query Ollama logprobs at temperature 0, verify bias
      cyclic_permutation_debasing/
        build_debasing.py          # Rotate options 4 ways per item, rebuild prompts
        resample_debasing_logprobs.py # Query Ollama logprobs per rotation, track content vs. position
        aggregate_debasing.py      # Aggregate the 4 rotations per item into a stability verdict
      action_policy/
        policy_formulation.py      # Formalize TRANSLATE/RETRIEVE/REPAIR/ABSTAIN action labels
      trajectory_features/
        trajectory_freature.py     # Build entropy/agreement/logprob feature vectors per item
      train_agent/
        main.py                    # Train + validate the Layer 2 decision-tree classifier
        aggregate_report.py        # Aggregate B/C/D/E results into one Phase 4 report
    tests/
      distribution_comparision.md  # Current experiment summary and interpretation
  data/
    processed/                     # Cleaned dataset caches
    pilot1/                        # Phase 1 samples, trials, buckets, review sample
    passage1/                      # Phase 2 passage samples and retrieval comparison
    fidelity_control1/             # Phase 3 control prompts, resamples, classification
    infra_check/                   # Logprob-based bias verification + cyclic-permutation debasing
    policy_formulation/            # Action labels, trajectory features, Phase 4 aggregate report
    loader/                        # Dataset loading helpers
    data_cleaner/                  # Dataset-specific cleaners
  results/                         # Early resampling sanity checks
  phases/                          # Phase diagrams (phase1-4.png)
  gen_fidelity_correct_count_hist.png # Phase 3 correct_count histogram
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
| `data/fidelity_control1/still_fails_items.json` | 78 Nepali `STILL_FAILS` items pulled from Phase 2 |
| `data/fidelity_control1/fidelity_control_prompts.json` | Audited items + control prompts (passage, question, English hint) |
| `data/fidelity_control1/resampled_fidelity_control_output.json` | 1,170 control-prompt trials (Tiny Aya Fire) |
| `data/fidelity_control1/resampled_fidelity_control_output_qwen_model.json` | Same 1,170-trial battery run against Qwen2.5-1.5B-instruct |
| `data/fidelity_control1/filtered_resampled_items.json` | 73-item clean subset (excludes `AMBIGUOUS_ITEM`/`CORRUPTED_EVIDENCE`) |
| `data/fidelity_control1/phase3_classification.json` | Per-item generation-fidelity vs. comprehension-gap classification |
| `data/fidelity_control1/phase3_classification_summary.json` | Classification counts and percentages |
| `data/infra_check/ollama_logprobs.json` | 12 `PASS_CONFOUNDED_BY_BIAS` items pulled from Phase 3 |
| `data/infra_check/resample_ollama_logprobs.json` | Deduped, single-letter-answer logprob-probe prompts (12 unique) |
| `data/infra_check/ollama_logprob_probe_results.json` | Temperature-0 logprob probe results, deterministic bias check |
| `data/infra_check/debasic_ollama_log.json` | 48 rotated prompts (12 `PASS_CONFOUNDED_BY_BIAS` items x 4 cyclic option-orderings) |
| `data/infra_check/debasic_ollama_logprob_resampling_result.json` | Temperature-0 logprob results for all 48 rotated prompts |
| `data/infra_check/aggregated_debasing.json` | Per-item content-vs-position stability verdict across the 4 rotations (12 items) |
| `data/policy_formulation/policy_Action_determination.json` | Final formalized `TRANSLATE`/`RETRIEVE`/`REPAIR`/`ABSTAIN` action label per item (92 items) |
| `data/policy_formulation/trajectory_feature.json` | 17-feature trajectory vector (entropy, agreement, logprob-derived) per item (92 records) |
| `data/policy_formulation/phase4_aggregate_report.md` | Aggregated Phase 4 B/C/D/E results, regenerated live from the model code |
| `gen_fidelity_correct_count_hist.png` | Distribution of `correct_count` within generation-fidelity failures |
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

### Phase 3: Fidelity-Control Experiment

```bash
python -m src.fieldity_control.failing_items
# manual audit step: add "english_hint" and "audit_label" to each item in
# data/fidelity_control1/still_fails_items.json before continuing
python -m src.fieldity_control.build_fieldity_control
python -m src.fieldity_control.resample_fidelity_control
python -m src.fieldity_control.filter_resampled_items
python -m src.fieldity_control.viz_resample_classification
```

### Follow-up: Logprob-Based Bias Verification

```bash
python -m src.bias_policy.infra_check.test_ollama_logprobs
python -m src.bias_policy.infra_check.build_logprobs
python -m src.bias_policy.infra_check.resample_logprobs
```

### Cyclic-Permutation Debasing

```bash
python -m src.bias_policy.cyclic_permutation_debasing.build_debasing
python -m src.bias_policy.cyclic_permutation_debasing.resample_debasing_logprobs
python -m src.bias_policy.cyclic_permutation_debasing.aggregate_debasing
```

### Action-Policy Formalization

```bash
python -m src.bias_policy.action_policy.policy_formulation
```

### Trajectory-Metric Features

```bash
python -m src.bias_policy.trajectory_features.trajectory_freature
```

### Layer 2 Classifier + Aggregate Report

```bash
python -m src.bias_policy.train_agent.main
python -m src.bias_policy.train_agent.aggregate_report
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

1. Design and validate a repair/verification mechanism for `GENERATION_FIDELITY_ISSUE` items, tiered by `correct_count` (near-total failure vs. weak-partial vs. borderline/unstable) rather than a single fallback.
2. Verify that the plurality answer across resamples actually matches ground truth for the borderline/unstable tier before treating it as a voting-repairable case.
3. Extend cyclic-permutation debasing beyond the 12 `PASS_CONFOUNDED_BY_BIAS` items — the 13 `COMPREHENSION_GAP` items are currently trusted as `REPAIR`-eligible on classification alone, without the same content-vs-position check applied to the debased items.
4. Resolve the 11 Nepali `PARTIAL_IMPROVEMENT` items that cleared neither the with-passage nor no-passage `8/15` bar in `policy_formulation.py` — they currently receive no action label at all.
5. Cleanly separate dataset artifacts from genuine model failures.
6. Add Bengali and BanglaRQA abstention data back into the action-policy framing.
7. Write the Phase 4 prose summary (12 confounded items resolved, finalized action hierarchy, honest Layer 2 performance ceiling) using `phase4_aggregate_report.md` as the source of truth.
8. Find a signal source beyond resample/logprob confidence for the Layer 2 classifier — Phase 4E showed confidence/self-consistency features alone plateau at macro-F1 ~0.28, particularly weak on `REPAIR`/`TRANSLATE`. Candidates: an "evidence present but unusable" feature, or substantially more labeled data for the minority classes.

Later:

1. Integrate translation as an actual tool path.
2. Compare English vs Hindi as recovery targets for Nepali `TRANSLATE` candidates.
3. Evaluate other candidate Layer 2 models with bias-conditioned comparisons (Qwen2.5-1.5B-instruct was ruled out — no genuine complementary capability, just a stronger position bias toward D).
4. Evaluate whether the trained policy (once it clears a reasonable performance bar) outperforms the fixed threshold rules in `policy_formulation.py` in an end-to-end setting.

## Notes On Reliability

The current findings are strong enough to guide the next research phase, but they are not a full dataset audit.

Known caveats:

- only a subset of items received manual review
- some Nepali passages may be corrupted or misaligned
- some gold labels and distractors have documented defects
- MCQ letter extraction is intentionally simple and should be stress-tested before larger runs
- current thresholds are research heuristics, not final policy boundaries
- Tiny Aya Fire has a confirmed position bias toward option A (45.7% selected vs. 29.5% correct); any pass/fail result on an A-correct item should be treated as potentially confounded, not just Phase 3's `PASS_CONFOUNDED_BY_BIAS` items
- the temperature-0 logprob check and the cyclic-permutation debasing built on top of it only cover the 12 `PASS_CONFOUNDED_BY_BIAS` items so far — the same content-vs-position check has not yet been applied to the 13 `COMPREHENSION_GAP` items or the broader dataset to see how far position bias reaches
- Phase 3's trivial-baseline / fact-repetition control (meant to establish a fidelity floor independent of task difficulty) was designed but not fully built out — the answer-handed control task became the main focus instead
- `policy_formulation.py`'s `REPAIR` action trusts `COMPREHENSION_GAP` classification at face value for the 61 items that never went through cyclic-permutation debasing (only the 12 `PASS_CONFOUNDED_BY_BIAS` items did) — those `REPAIR` labels have not had the same content-vs-position check applied as the 4 `permutation_stable` items
- 11 Nepali `PARTIAL_IMPROVEMENT` items clear neither the with-passage nor the no-passage `8/15` bar in `policy_formulation.py` and are left without an action label in `data/policy_formulation/policy_Action_determination.json` — a known gap, not an omission of the underlying data
- the Phase 4E classifier trains on only 92 labeled items, with the rarest class (`TRANSLATE`) at just 4 examples; the reported macro-F1 (~0.28) should be read as a directional signal, not a precise estimate — the confidence interval around it is wide at this sample size
- candidate signals that fed into building the C action labels (retrieval-improvement, EN/NE recovery, fidelity classification, permutation verdict) are related to — but distinct in construction from — the D features actually used as classifier inputs; the 3- and 16-feature runs avoid using C's thresholded outputs directly, but this boundary should be checked again before adding any new feature derived from C's inputs
- the Phase 4 aggregate report (`phase4_aggregate_report.md`) re-runs E live from `main.py`, so its numbers track the current script, but the "unrestricted, ungrouped" tree variant referenced in the Phase 4 progress notes as a sanity check is not preserved as a runnable path in the shipped script — only the dummy baseline and the two grouped/nested-pruned variants are

These caveats are part of the research contribution: LRCal-Agent is explicitly trying to learn when model behavior is unsupported, unstable, or language-conditionally unreliable.
