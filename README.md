# LRCal-Agent

**Research question:** Can an LLM agent recognize when it lacks sufficient knowledge in a low-resource language, and choose an appropriate action — answer, abstain, translate, retrieve, or clarify — rather than answering confidently and wrongly?

**Core contribution:** LRCal-Agent is a lightweight, *trained* policy module (not a prompting technique) that sits on top of a base LLM and learns, from labeled trajectories, to choose the right action given a calibration signal and contextual features.

## Architecture

Two layers:

- **Layer 1 — Signal/Calibration.** Measures confidence via resampling: the same query is sent to the model N times (temperature > 0) and answer agreement/entropy is computed. This layer asks "does the model's confidence track its actual accuracy?" and is built on established calibration techniques (ECE, Brier score) rather than being itself novel.
- **Layer 2 — Policy (LRCal-Agent).** A small trained MLP with two heads (confidence + action) that takes Layer 1's signal plus trajectory-level features (tool calls made, retrieval quality, translation attempted, language/resource level) and outputs a chosen action from `{answer, abstain, translate, retrieve, clarify}`. This is the novel contribution: a *trained* policy rather than a fixed threshold rule.

Layer 1 always runs first and only measures; Layer 2 decides; the base agent executes the decision (e.g., actually performs the translation if that's the chosen action).

## Languages

- **English** — high-resource anchor
- **Bengali** — primary low-resource language (has a native, validated unanswerable-question dataset: BanglaRQA)
- **Nepali** — harder case; no native unanswerable-question resource exists, so one is constructed
- **Hindi** — optional secondary experiment only (see below), not a core language

## Core methodology — Constructs A and B

Belebele MCQ items (parallel across en/ne/bn, all normally answerable-with-passage) are repurposed: the passage is stripped, the bare question is resampled N=10-15 times per language, and per-language accuracy is measured.

- **Construct A** — near-chance accuracy in *both* English and Nepali → genuinely unanswerable without evidence, language-independent. Used as a **control/contrast**.
- **Construct B** — near-chance in Nepali, clearly above-chance in English → a **Nepali-specific knowledge gap** (the model knows the fact, but the language shift is hiding it). This is the **primary construct** — the paper's headline result, and the only construct where "translate" is ever the correct gold action.

Gold actions are defined per *(construct, agent variant)* pair — the same Construct B item is "abstain" for a no-tool agent but "translate" for a translate-capable agent.

## Dataset stack

| Dataset | Role |
|---|---|
| Belebele (en/ne/bn) | Calibration/accuracy backbone + source for Construct A/B construction. Human-translated, parallel, MCQ, no unanswerable items. |
| BanglaRQA | Native Bengali abstention data. 3,000 passages, ~14,889 QA pairs (flattened), ~24.4% unanswerable, 4 question types (factoid, confirmation, list, causal). |
| SQuAD 2.0 | English abstention data (native, standard benchmark). |
| Yunika/Nepali-QA | Small (266 items) supplementary Nepali accuracy check only — no unanswerable items, not usable for action-head training. |
| SQuAD 2.0 Hindi (optional, not yet integrated) | Only used on already-identified Construct B items, for the Hindi-bridge experiment. |

## Optional secondary experiment: Hindi bridge

Since Hindi is linguistically close to Nepali, for Construct B items we compare recovery rate when translating to Hindi vs. English. Framed precisely as *"a direct, item-level comparison of English and Hindi as recovery targets for Nepali questions"* — not as the first use of Hindi as a related language for Nepali (already done by prior work).

## Related work

**Feng et al. 2024 (EMNLP), "Teaching LLMs to Abstain across Languages via Multilingual Feedback"** (arXiv:2406.15948) is the closest prior work. It covers exactly Nepali/Bengali/Hindi and already runs experiments on Belebele. Key differentiators for LRCal-Agent:

- Their abstain ground truth is pure post-hoc correctness ("would the answer be wrong"), with no distinction between *why* it's wrong. Construct A/B explicitly separates language-independent vs. language-specific gaps — theirs can't.
- Their method (MULTI-RELATED) is prompting-only, with no training. LRCal-Agent is a trained module.
- Their action space is binary (abstain/answer). LRCal-Agent has five explicit actions.

## Base model

**Tiny Aya-Fire** (`hf.co/CohereLabs/tiny-aya-fire-GGUF:Q4_K_M`, via Ollama) — Cohere's 3.35B multilingual model, South-Asia-specialized variant. Chosen over Tiny Aya-Global after multiple replicated head-to-head resampling tests (Global showed severe, repeated script-drift failures — e.g., generating in Telugu when asked a Nepali question). Chosen over Aya-23 after confirming Aya-23's 23-language set does **not** include Nepali or Bengali. Runs locally on 16GB unified memory (MacBook Air M4).

## Translation tool

**IndicTrans2** (planned, not yet integrated) — chosen over the Google Translate API for being free, open, reproducible, and benchmarked specifically on Nepali/Hindi/Bengali.

## Validated findings (Phase 0)

- Massive English vs. Nepali/Bengali accuracy gap reproduced directly (English: consistently ~100%; Nepali/Bengali: highly variable, often near-chance) — the core calibration-gap phenomenon the project studies.
- "Confidently wrong" failure mode confirmed with real examples (e.g., Fire repeatedly answering "Nepalgunj" for Nepal's capital).
- Script-drift failure mode discovered and replicated (model generating in unrelated scripts — Telugu, Arabic — instead of the target language).
- A distinct **generation-fidelity vs. knowledge-access** distinction surfaced for Bengali specifically (e.g., "Dhaka" spelled wrong/near-miss vs. genuinely wrong city) — flagged as a real methodological nuance requiring careful, hedged claims (partial semantic recovery ≠ proven knowledge), not yet a core research pillar.
- Bengali/Nepali grammatical inflection (case suffixes) initially caused false-negative scoring bugs — fixed via a suffix-aware extraction tier in the scoring pipeline (see `fire-api-call/src.py`).

## Repo layout

```
data/
  loader/            # HF `datasets` loaders (Belebele en/ne/bn, SQuAD 2.0, BanglaRQA, Yunika)
  data_cleaner/       # Per-dataset cleaning scripts
  processed/          # Cached, cleaned JSON — the datasets loaders/cleaners produce
fire-api-call/
  src.py              # Resampling + calibration experiment harness (Ollama-backed)
resampling_test_results*.json   # Output of resampling experiments (per language)
```

## Current status

Phase 0 (setup) is essentially complete: base model validated and chosen, all four core datasets loaded/cleaned/cached to `data/processed/`, and the resampling/scoring pipeline built and debugged.

**Next milestone:** Phase 1 pilot — run the full Construct A/B filtering protocol on a 200-300 item Belebele sample to get a real p_B estimate before scaling to the full dataset.
