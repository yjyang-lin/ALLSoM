from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tabpfn import TabPFNClassifier


FAMILIES = ("FMO", "SULT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate separate FMO/SULT TabPFN SoM classifiers"
    )
    parser.add_argument("--feature_csv", type=Path, default="tabpfn/FMO_SULT_site_features.csv")
    parser.add_argument("--metrics_csv", type=Path, default="tabpfn/FMO_SULT_tabpfn_metrics.csv")
    parser.add_argument("--prediction_csv", type=Path, default="tabpfn/FMO_SULT_tabpfn_predictions.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "som_all_acc": accuracy_score(labels, predictions),
        "som_all_precision": precision_score(labels, predictions, zero_division=0),
        "som_all_recall": recall_score(labels, predictions, zero_division=0),
        "som_all_f1": f1_score(labels, predictions, zero_division=0),
        "som_all_mcc": matthews_corrcoef(labels, predictions),
        "som_all_spec": tn / (tn + fp) if tn + fp else 0.0,
        "som_all_jaccard": jaccard_score(labels, predictions, zero_division=0),
        "som_all_auroc": roc_auc_score(labels, probabilities),
        "som_all_auprc": average_precision_score(labels, probabilities),
    }


def calculate_topk(test: pd.DataFrame) -> dict:
    hits = {1: 0, 2: 0, 3: 0}
    eligible = 0

    for _, group in test.groupby("record_id", sort=False):
        labels = group["label"].to_numpy(dtype=np.int64)
        if not np.any(labels == 1):
            continue
        eligible += 1
        probabilities = group["som_probability"].to_numpy(dtype=float)
        order = np.argsort(-probabilities, kind="mergesort")
        ranked_labels = labels[order]
        for k in hits:
            if np.any(ranked_labels[: min(k, len(ranked_labels))] == 1):
                hits[k] += 1

    return {
        "top1": hits[1] / eligible if eligible else 0.0,
        "top2": hits[2] / eligible if eligible else 0.0,
        "top3": hits[3] / eligible if eligible else 0.0,
        "topk_records": eligible,
    }


def validate_no_leakage(frame: pd.DataFrame, family: str) -> None:
    family_frame = frame.loc[frame["family"].eq(family)]
    train_molecules = set(
        family_frame.loc[family_frame["split"].eq("train"), "molecule_id"]
    )
    test_molecules = set(
        family_frame.loc[family_frame["split"].eq("test"), "molecule_id"]
    )
    overlap = train_molecules & test_molecules
    if overlap:
        raise ValueError(f"{family}: {len(overlap)} molecules leak across train/test")


def positive_probability(classifier, matrix: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(matrix)
    classes = np.asarray(classifier.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices) != 1:
        raise ValueError(f"Cannot locate positive class in {classes.tolist()}")
    return probabilities[:, positive_indices[0]]


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.feature_csv)
    required = {
        "record_id", "family", "enzyme", "molecule_id", "split",
        "atom_index", "label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    feature_columns = sorted(
        column for column in frame.columns if column.startswith("feature_")
    )
    if len(feature_columns) != 192:
        raise ValueError(f"Expected 192 feature columns, got {len(feature_columns)}")
    if not frame["split"].isin(["train", "test"]).all():
        raise ValueError("split must contain only train/test")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("label must contain only 0/1")

    metric_rows = []
    prediction_frames = []

    for family in FAMILIES:
        validate_no_leakage(frame, family)
        family_frame = frame.loc[frame["family"].eq(family)].copy()
        train = family_frame.loc[family_frame["split"].eq("train")].copy()
        test = family_frame.loc[family_frame["split"].eq("test")].copy()
        if train.empty or test.empty:
            raise ValueError(f"{family}: empty train or test set")
        if train["label"].nunique() != 2 or test["label"].nunique() != 2:
            raise ValueError(f"{family}: train and test must both contain 0/1 labels")

        classifier = TabPFNClassifier(
            device=args.device,
            random_state=args.seed,
            balance_probabilities=False,
        )
        classifier.fit(
            train[feature_columns].to_numpy(dtype=np.float32),
            train["label"].to_numpy(dtype=np.int64),
        )
        probabilities = positive_probability(
            classifier, test[feature_columns].to_numpy(dtype=np.float32)
        )
        test["som_probability"] = probabilities
        test["som_prediction"] = (probabilities >= 0.5).astype(np.int64)

        metrics = calculate_metrics(
            test["label"].to_numpy(dtype=np.int64), probabilities
        )
        metrics.update(calculate_topk(test))
        metric_rows.append({
            "family": family,
            "train_sites": len(train),
            "test_sites": len(test),
            **metrics,
        })
        prediction_frames.append(test.drop(columns=feature_columns))

    metrics_output = pd.DataFrame(metric_rows)
    predictions_output = pd.concat(prediction_frames, ignore_index=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.to_csv(args.metrics_csv, index=False)
    predictions_output.to_csv(args.prediction_csv, index=False)

    print(metrics_output.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Metrics: {args.metrics_csv}")
    print(f"Predictions: {args.prediction_csv}")


if __name__ == "__main__":
    main()
