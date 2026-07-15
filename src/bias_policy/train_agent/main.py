import json
import math
from collections import Counter

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier


INPUT_FILE = "data/policy_formulation/trajectory_feature.json"
EXPECTED_ITEM_COUNT = 92
OUTER_SPLITS = 4
INNER_SPLITS = 3
RANDOM_STATE = 42

BASELINE_FEATURES = [
    "entropy",
    "agreement_rate",
    "token_probability_variance",
]

RICH_TRAJECTORY_FEATURES = [
    "entropy",
    "agreement_rate",
    "resample_prob_A",
    "resample_prob_B",
    "resample_prob_C",
    "resample_prob_D",
    "resample_top2_margin",
    "token_probability_variance",
    "logprob_prob_A",
    "logprob_prob_B",
    "logprob_prob_C",
    "logprob_prob_D",
    "logprob_entropy",
    "logprob_top2_margin",
    "resample_logprob_top_match",
    "resample_logprob_js_divergence",
]

TREE_PARAM_GRID = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.01],
}


def load_items():
    with open(INPUT_FILE, encoding="utf-8") as file:
        items = json.load(file)

    if len(items) != EXPECTED_ITEM_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ITEM_COUNT} trajectory items, got {len(items)}"
        )

    unique_keys = {
        (item["link"], item["question_number"])
        for item in items
    }
    if len(unique_keys) != EXPECTED_ITEM_COUNT:
        raise ValueError("Trajectory data contains duplicate item keys")

    return items


def build_feature_matrix(items, feature_names):
    rows = []

    for item in items:
        row = []
        for feature in feature_names:
            value = item.get(feature)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"Invalid feature {feature!r} for item "
                    f"{item.get('item_id')}: {value!r}"
                )
            row.append(value)
        rows.append(row)

    return np.asarray(rows, dtype=float)


def make_outer_splits(y, groups):
    splitter = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    placeholder_x = np.zeros((len(y), 1))
    return list(splitter.split(placeholder_x, y, groups))


def evaluate_dummy(y, outer_splits):
    predictions = np.empty_like(y)
    placeholder_x = np.zeros((len(y), 1))

    for train_indices, test_indices in outer_splits:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(placeholder_x[train_indices], y[train_indices])
        predictions[test_indices] = model.predict(placeholder_x[test_indices])

    return predictions, []


def evaluate_decision_tree(
    x,
    y,
    groups,
    labels,
    outer_splits,
):
    predictions = np.empty_like(y)
    selected_parameters = []

    def macro_f1(estimator, x_scored, y_true):
        y_pred = estimator.predict(x_scored)
        return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    for fold_number, (train_indices, test_indices) in enumerate(
        outer_splits,
        start=1,
    ):
        x_train = x[train_indices]
        y_train = y[train_indices]
        groups_train = groups[train_indices]

        inner_splitter = StratifiedGroupKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + fold_number,
        )
        search = GridSearchCV(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            param_grid=TREE_PARAM_GRID,
            scoring=macro_f1,
            cv=inner_splitter,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(x_train, y_train, groups=groups_train)

        predictions[test_indices] = search.predict(x[test_indices])
        selected_parameters.append(search.best_params_)

    return predictions, selected_parameters


def print_evaluation(name, y, predictions, labels, selected_parameters):
    print(f"\n{'=' * 80}")
    print(name)
    print(f"{'=' * 80}")
    print(classification_report(
        y,
        predictions,
        labels=labels,
        zero_division=0,
    ))
    print("labels:", labels)
    print("confusion matrix:")
    print(confusion_matrix(y, predictions, labels=labels))

    if selected_parameters:
        print("selected tree parameters by outer fold:")
        for fold_number, parameters in enumerate(selected_parameters, start=1):
            print(f"  fold {fold_number}: {parameters}")

    return {
        "name": name,
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(
            y,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }


def print_summary(scores):
    print(f"\n{'=' * 80}")
    print("OUT-OF-FOLD SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"{'model':32s} "
        f"{'accuracy':>10s} "
        f"{'balanced':>10s} "
        f"{'macro_f1':>10s}"
    )
    for score in scores:
        print(
            f"{score['name']:32s} "
            f"{score['accuracy']:10.3f} "
            f"{score['balanced_accuracy']:10.3f} "
            f"{score['macro_f1']:10.3f}"
        )


def main():
    items = load_items()
    y = np.asarray([item["action"] for item in items])
    groups = np.asarray([item["link"] for item in items])
    labels = sorted(set(y))
    outer_splits = make_outer_splits(y, groups)

    print("items:", len(items))
    print("unique source links:", len(set(groups)))
    print("class counts:", dict(Counter(y)))
    print("outer validation: StratifiedGroupKFold")
    print("selection metric: macro-F1")

    evaluations = []

    dummy_predictions, dummy_parameters = evaluate_dummy(y, outer_splits)
    evaluations.append(print_evaluation(
        "Dummy: most frequent action",
        y,
        dummy_predictions,
        labels,
        dummy_parameters,
    ))

    for name, feature_names in [
        ("Decision tree: original 3 features", BASELINE_FEATURES),
        ("Decision tree: rich trajectory features", RICH_TRAJECTORY_FEATURES),
    ]:
        x = build_feature_matrix(items, feature_names)
        predictions, selected_parameters = evaluate_decision_tree(
            x,
            y,
            groups,
            labels,
            outer_splits,
        )
        evaluations.append(print_evaluation(
            name,
            y,
            predictions,
            labels,
            selected_parameters,
        ))

    print_summary(evaluations)


if __name__ == "__main__":
    main()