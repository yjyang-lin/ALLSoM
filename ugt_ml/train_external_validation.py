from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
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
from xgboost import XGBClassifier


RDLogger.DisableLog("rdApp.*")

REQUIRED_COLUMNS = {"standardized_SMILES", "uniprot_id", "label"}
MORGAN_BITS = 512
MORGAN_RADIUS = 2
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_BITS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF/XGBoost UGT classifiers and run external validation."
    )
    parser.add_argument("--train-csv", type=Path, default="metab_data/UGT/UGT_train.csv")
    parser.add_argument("--external-csv", type=Path, default="inference/UGT/UGT_ex_val.csv")
    parser.add_argument(
        "--model",
        choices=("rf", "xgboost", "both"),
        default="both",
        help="Model family to train (default: both).",
    )
    parser.add_argument("--output-dir", type=Path, default="ugt_ml/results")
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["standardized_SMILES"] = frame["standardized_SMILES"].astype(str).str.strip()
    frame["uniprot_id"] = frame["uniprot_id"].astype(str).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not frame["label"].isin((0, 1)).all():
        raise ValueError(f"{path} must contain only 0 or 1 in the label column")
    return frame


def fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse standardized_SMILES: {smiles}")

    maccs = MACCSkeys.GenMACCSKeys(mol)
    morgan = MORGAN_GENERATOR.GetFingerprint(mol)
    maccs_array = np.zeros(maccs.GetNumBits(), dtype=np.uint8)
    morgan_array = np.zeros(MORGAN_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(maccs, maccs_array)
    DataStructs.ConvertToNumpyArray(morgan, morgan_array)
    return np.concatenate((maccs_array, morgan_array))


def build_feature_cache(*frames: pd.DataFrame) -> dict[str, np.ndarray]:
    smiles_values = pd.concat(
        [frame["standardized_SMILES"] for frame in frames], ignore_index=True
    ).unique()
    return {smiles: fingerprint(smiles) for smiles in smiles_values}


def feature_matrix(frame: pd.DataFrame, cache: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([cache[smiles] for smiles in frame["standardized_SMILES"]])


def make_estimator(model_name: str):
    # Keep library defaults, except for the requested reproducible RF seed.
    if model_name == "rf":
        return RandomForestClassifier(random_state=0)
    if model_name == "xgboost":
        return XGBClassifier()
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(y_true: np.ndarray, y_probability: np.ndarray) -> dict[str, float]:
    y_pred = (y_probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "specificity": specificity,
        "jaccard": jaccard_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_probability),
        "auprc": average_precision_score(y_true, y_probability),
    }


def train_and_predict(
    model_name: str,
    train_frame: pd.DataFrame,
    external_frame: pd.DataFrame,
    cache: dict[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    train_subtypes = set(train_frame["uniprot_id"])
    external_subtypes = set(external_frame["uniprot_id"])
    unknown = sorted(external_subtypes - train_subtypes)
    if unknown:
        raise ValueError(
            f"External set contains subtypes absent from the training set: {', '.join(unknown)}"
        )

    models: dict[str, object] = {}
    probabilities = np.full(len(external_frame), np.nan, dtype=float)

    for subtype in sorted(external_subtypes):
        train_mask = train_frame["uniprot_id"].eq(subtype)
        external_mask = external_frame["uniprot_id"].eq(subtype)
        train_part = train_frame.loc[train_mask]
        external_part = external_frame.loc[external_mask]

        estimator = make_estimator(model_name)
        estimator.fit(
            feature_matrix(train_part, cache),
            train_part["label"].to_numpy(),
        )
        probabilities[external_mask.to_numpy()] = estimator.predict_proba(
            feature_matrix(external_part, cache)
        )[:, 1]
        models[subtype] = estimator

    if np.isnan(probabilities).any():
        raise RuntimeError("Some external validation samples did not receive prediction probabilities")
    return models, probabilities


def main() -> None:
    args = parse_args()
    train_frame = read_dataset(args.train_csv)
    external_frame = read_dataset(args.external_csv)
    feature_cache = build_feature_cache(train_frame, external_frame)
    model_names = ("rf", "xgboost") if args.model == "both" else (args.model,)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    prediction_output = external_frame.copy()

    for model_name in model_names:
        models, probabilities = train_and_predict(
            model_name, train_frame, external_frame, feature_cache
        )
        metrics = evaluate(external_frame["label"].to_numpy(), probabilities)
        metric_rows.append({"model": model_name, **metrics})
        prediction_output[f"{model_name}_probability"] = probabilities
        prediction_output[f"{model_name}_prediction"] = (probabilities >= 0.5).astype(int)

        joblib.dump(
            {
                "models": models,
                "feature": {
                    "maccs_bits": 167,
                    "morgan_bits": MORGAN_BITS,
                    "morgan_radius": MORGAN_RADIUS,
                },
            },
            args.output_dir / f"{model_name}_subtype_models.joblib",
        )

    metrics_output = pd.DataFrame(metric_rows)
    metrics_output.to_csv(args.output_dir / "external_metrics.csv", index=False)
    prediction_output.to_csv(
        args.output_dir / "external_predictions.csv", index=False
    )

    print(
        f"Train rows: {len(train_frame)}; external rows: {len(external_frame)}; "
        f"features: 167 MACCS + {MORGAN_BITS} Morgan = {167 + MORGAN_BITS}"
    )
    print(metrics_output.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
