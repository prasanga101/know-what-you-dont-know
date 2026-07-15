import json
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from src.bias_policy.train_agent.main import (
    BASELINE_FEATURES,
    RICH_TRAJECTORY_FEATURES,
    build_feature_matrix,
    evaluate_decision_tree,
    evaluate_dummy,
    load_items,
    make_outer_splits,
)

DEBIASING_FILE = "data/infra_check/aggregated_debasing.json"
POLICY_FILE = "data/policy_formulation/policy_Action_determination.json"
TRAJECTORY_FILE = "data/policy_formulation/trajectory_feature.json"
REPORT_FILE = "data/policy_formulation/phase4_aggregate_report.md"

VERDICT_VALUES = {"permutation_stable", "position_anchored", "complex_interaction"}
TRAJECTORY_FEATURES_FOR_ETA = ["entropy", "agreement_rate", "token_probability_variance"]


def find_verdict_key(items):
    for key, value in items[0].items():
        if isinstance(value, str) and value in VERDICT_VALUES:
            return key
    raise KeyError(
        f"No field holding a debiasing verdict found in {DEBIASING_FILE}. "
        f"Available keys: {list(items[0].keys())}"
    )


def summarize_debiasing():
    with open(DEBIASING_FILE, encoding="utf-8") as file:
        items = json.load(file)
    verdict_key = find_verdict_key(items)
    counts = Counter(item[verdict_key] for item in items)
    return len(items), counts


def summarize_policy():
    with open(POLICY_FILE, encoding="utf-8") as file:
        items = json.load(file)
    counts = Counter(item["action"] for item in items)
    return len(items), counts


def eta_squared(items, feature):
    values = [item[feature] for item in items]
    overall_mean = sum(values) / len(values)
    total_variation = sum((value - overall_mean) ** 2 for value in values)

    if total_variation == 0:
        return 0.0

    by_action = {}
    for item in items:
        by_action.setdefault(item["action"], []).append(item[feature])

    between_group_variation = 0.0
    for group in by_action.values():
        group_mean = sum(group) / len(group)
        between_group_variation += len(group) * (group_mean - overall_mean) ** 2

    return between_group_variation / total_variation


def summarize_trajectory():
    with open(TRAJECTORY_FILE, encoding="utf-8") as file:
        items = json.load(file)

    scores = {
        feature: eta_squared(items, feature)
        for feature in TRAJECTORY_FEATURES_FOR_ETA
    }
    return len(items), scores


def summarize_classifier():
    items = load_items()
    y = np.asarray([item["action"] for item in items])
    groups = np.asarray([item["link"] for item in items])
    labels = sorted(set(y))
    outer_splits = make_outer_splits(y, groups)

    runs = []

    dummy_predictions, _ = evaluate_dummy(y, outer_splits)
    runs.append(("Dummy (always ABSTAIN)", dummy_predictions))

    for name, feature_names in [
        ("Decision tree, 3 features (grouped, nested-pruned)", BASELINE_FEATURES),
        ("Decision tree, 16 features (grouped, nested-pruned)", RICH_TRAJECTORY_FEATURES),
    ]:
        x = build_feature_matrix(items, feature_names)
        predictions, _ = evaluate_decision_tree(x, y, groups, labels, outer_splits)
        runs.append((name, predictions))

    rows = []
    for name, predictions in runs:
        rows.append(
            {
                "name": name,
                "accuracy": accuracy_score(y, predictions),
                "balanced_accuracy": balanced_accuracy_score(y, predictions),
                "macro_f1": f1_score(
                    y, predictions, labels=labels, average="macro", zero_division=0
                ),
            }
        )
    return rows


def build_report():
    lines = ["# Phase 4 Aggregate Report", ""]

    debias_n, debias_counts = summarize_debiasing()
    lines.append("## B: Cyclic-permutation debiasing")
    lines.append(f"Resolved {debias_n} confounded items.")
    for verdict, count in debias_counts.most_common():
        lines.append(f"- {verdict}: {count}")
    lines.append("")

    policy_n, policy_counts = summarize_policy()
    lines.append("## C: Action-policy hierarchy")
    lines.append(f"Labeled {policy_n} items.")
    for action, count in policy_counts.most_common():
        lines.append(f"- {action}: {count}")
    lines.append("")

    traj_n, eta_scores = summarize_trajectory()
    lines.append("## D: Trajectory features")
    lines.append(
        f"Built features for {traj_n} items. "
        f"Eta-squared (share of action-label variance explained):"
    )
    for feature, score in eta_scores.items():
        lines.append(f"- {feature}: {score:.4f}")
    lines.append("")

    lines.append("## E: Layer 2 classifier (out-of-fold, pooled across outer folds)")
    for row in summarize_classifier():
        lines.append(
            f"- {row['name']}: accuracy={row['accuracy']:.3f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    report = build_report()
    print(report)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nSaved report to {REPORT_FILE}")


if __name__ == "__main__":
    main()