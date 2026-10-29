from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


sys.path.insert(0, ".")

from featurize.mol_featurize import (
    build_graph,
    build_node_feature_cache,
    prepare_molecule_and_label,
)


RDLogger.DisableLog("rdApp.*")

FMO_SMARTS = (
    "[N;X3;+0;!$(N-C(=O));!$(N-S(=O)=O)]",
    "[nH0;+0]",
    "[S;X2;+0]",
    "[S;X3;+0](=O)",
)
SULT_SMARTS = (
    "[O;H1;+0;$([O]-[C,c]);!$([O]-C=O)]",
    "[N;H1,H2;+0;!$(N-C(=O));!$(N-S(=O)=O)]",
)
FAMILY_PATTERNS = {
    "FMO": tuple(Chem.MolFromSmarts(value) for value in FMO_SMARTS),
    "SULT": tuple(Chem.MolFromSmarts(value) for value in SULT_SMARTS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw node + enzyme one-hot FMO/SULT site features"
    )
    parser.add_argument("--input_csv", type=Path, default="tabpfn/enzyme_FMO_SULT.csv")
    parser.add_argument("--output_csv", type=Path, default="tabpfn/node_onehot/FMO_SULT_node_onehot_features.csv")
    parser.add_argument("--feature_workers", type=int, default=4)
    return parser.parse_args()


def enzyme_family(enzyme: str) -> str:
    value = str(enzyme).strip().upper()
    if value.startswith("FMO"):
        return "FMO"
    if value.startswith("SULT"):
        return "SULT"
    raise ValueError(f"Unsupported enzyme outside FMO/SULT: {enzyme}")


def canonical_smiles(smiles: str, row_index: int) -> str:
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise ValueError(f"Row {row_index}: invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def candidate_labels(molecule: Chem.Mol, family: str, row_index: int) -> list:
    candidates = {
        atom_index
        for pattern in FAMILY_PATTERNS[family]
        for match in molecule.GetSubstructMatches(pattern)
        for atom_index in match
    }
    mapped = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() == 1
    }
    outside = mapped - candidates
    if outside:
        raise ValueError(
            f"Row {row_index}: mapped SoM outside {family} SMARTS: {sorted(outside)}"
        )
    if not candidates:
        raise ValueError(f"Row {row_index}: no {family} SMARTS candidate atoms")
    return [
        1 if atom_index in mapped else 0 if atom_index in candidates else "mask"
        for atom_index in range(molecule.GetNumAtoms())
    ]


def parse_records(frame: pd.DataFrame):
    records = []
    descriptor_items = {}

    for row_index, row in frame.iterrows():
        enzyme = str(row["Enzyme"]).strip()
        family = enzyme_family(enzyme)
        split = str(row["split"]).strip().lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Row {row_index}: split must be train/test, got {split}")

        mapped_molecule = Chem.MolFromSmiles(str(row["SoM"]).strip())
        if mapped_molecule is None:
            raise ValueError(f"Row {row_index}: invalid SoM SMILES")
        labels = candidate_labels(mapped_molecule, family, row_index)
        graph_molecule, graph_labels, clean_smiles, descriptor_smiles = (
            prepare_molecule_and_label(mapped_molecule, labels)
        )
        descriptor_items[descriptor_smiles] = clean_smiles
        records.append({
            "record_id": int(row_index),
            "source": str(row.get("Source", "")),
            "family": family,
            "enzyme": enzyme,
            "name": str(row.get("Name", "")),
            "smiles": str(row["SMILES"]),
            "molecule_id": canonical_smiles(row["SMILES"], row_index),
            "split": split,
            "substrate_label": int(row["label"]),
            "graph_molecule": graph_molecule,
            "graph_labels": graph_labels,
            "descriptor_smiles": descriptor_smiles,
        })
    return records, descriptor_items


def build_output(records, feature_cache, enzymes):
    output_rows = []
    enzyme_to_index = {enzyme: index for index, enzyme in enumerate(enzymes)}
    onehot_columns = [f"enzyme_onehot_{enzyme}" for enzyme in enzymes]
    node_columns = None

    for record in records:
        descriptor_smiles = record["descriptor_smiles"]
        if descriptor_smiles not in feature_cache:
            raise RuntimeError(
                f"Feature generation failed for record {record['record_id']}"
            )
        graph, _ = build_graph(
            record["graph_molecule"],
            feature_cache[descriptor_smiles],
            record["graph_labels"],
        )
        node_features = graph.x.numpy()
        labels = graph.y_node.numpy()
        if node_columns is None:
            node_columns = [
                f"node_feature_{index:02d}" for index in range(node_features.shape[1])
            ]
        elif len(node_columns) != node_features.shape[1]:
            raise RuntimeError("Inconsistent raw node feature dimensions")

        onehot = np.zeros(len(enzymes), dtype=np.float32)
        onehot[enzyme_to_index[record["enzyme"]]] = 1.0
        atom_symbols = [
            atom.GetSymbol() for atom in record["graph_molecule"].GetAtoms()
        ]

        for atom_index in np.flatnonzero(labels >= 0):
            row = {
                "record_id": record["record_id"],
                "source": record["source"],
                "family": record["family"],
                "enzyme": record["enzyme"],
                "name": record["name"],
                "smiles": record["smiles"],
                "molecule_id": record["molecule_id"],
                "split": record["split"],
                "substrate_label": record["substrate_label"],
                "atom_index": int(atom_index),
                "atom_symbol": atom_symbols[atom_index],
                "label": int(labels[atom_index]),
            }
            row.update(zip(node_columns, node_features[atom_index].tolist()))
            row.update(zip(onehot_columns, onehot.tolist()))
            output_rows.append(row)

    return pd.DataFrame(output_rows), node_columns, onehot_columns


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    required = {"Enzyme", "SMILES", "SoM", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    enzymes = sorted(frame["Enzyme"].str.strip().unique())
    records, descriptor_items = parse_records(frame)
    feature_cache, errors = build_node_feature_cache(
        descriptor_items, num_workers=args.feature_workers
    )
    if errors:
        raise RuntimeError(f"Molecular feature errors: {errors}")

    output, node_columns, onehot_columns = build_output(
        records, feature_cache, enzymes
    )
    output = output.sort_values(["family", "record_id", "atom_index"]).reset_index(
        drop=True
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, float_format="%.8g")

    print(f"Raw node dimension: {len(node_columns)}")
    print(f"Enzyme one-hot dimension: {len(onehot_columns)}")
    print(f"Combined dimension: {len(node_columns) + len(onehot_columns)}")
    print(
        output.groupby(["family", "split"])
        .agg(sites=("label", "size"), som=("label", "sum"))
        .assign(non_som=lambda value: value["sites"] - value["som"])
        .to_string()
    )
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
