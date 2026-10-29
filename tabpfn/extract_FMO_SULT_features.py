from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader


sys.path[:0] = [".", "model"]

from featurize.mol_featurize import (
    build_graph,
    build_node_feature_cache,
    prepare_molecule_and_label,
)
from train import SOMModel
from utils import load_protein_dict, som_collate


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
FEATURE_DIM = 192
FEATURE_COLUMNS = [f"feature_{index:03d}" for index in range(FEATURE_DIM)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 192D pretrained fused features for FMO/SULT sites"
    )
    parser.add_argument("--input_csv", type=Path, default="tabpfn/enzyme_FMO_SULT.csv")
    parser.add_argument("--output_csv", type=Path, default="tabpfn/FMO_SULT_site_features.csv")
    parser.add_argument("--checkpoint", type=Path, default="runs_pretrain/split_enzyme_cluster/exp_1/best.pt")
    parser.add_argument("--config", type=Path, default="runs_pretrain/split_enzyme_cluster/exp_1/config.json")
    parser.add_argument("--protein_pkl", type=Path, default="features/protein/protein_metab.pkl")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
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


def candidate_labels(
    mapped_molecule: Chem.Mol,
    family: str,
    row_index: int,
) -> list:
    candidates = {
        atom_index
        for pattern in FAMILY_PATTERNS[family]
        for match in mapped_molecule.GetSubstructMatches(pattern)
        for atom_index in match
    }
    mapped = {
        atom.GetIdx()
        for atom in mapped_molecule.GetAtoms()
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
        for atom_index in range(mapped_molecule.GetNumAtoms())
    ]


def parse_records(frame: pd.DataFrame, protein_dict: dict):
    parsed = []
    descriptor_items = {}

    for row_index, row in frame.iterrows():
        enzyme = str(row["Enzyme"]).strip()
        family = enzyme_family(enzyme)
        split = str(row["split"]).strip().lower()
        if split not in {"train", "test"}:
            raise ValueError(f"Row {row_index}: split must be train/test, got {split}")
        if enzyme not in protein_dict:
            raise KeyError(f"Row {row_index}: protein features missing for {enzyme}")

        mapped_molecule = Chem.MolFromSmiles(str(row["SoM"]).strip())
        if mapped_molecule is None:
            raise ValueError(f"Row {row_index}: invalid SoM SMILES")
        labels = candidate_labels(mapped_molecule, family, row_index)
        graph_molecule, graph_labels, clean_smiles, descriptor_smiles = (
            prepare_molecule_and_label(mapped_molecule, labels)
        )
        descriptor_items[descriptor_smiles] = clean_smiles
        parsed.append({
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
    return parsed, descriptor_items


def build_samples(parsed, feature_cache, protein_dict, max_prot_len):
    samples = []
    metadata = {}

    for item in parsed:
        descriptor_smiles = item["descriptor_smiles"]
        if descriptor_smiles not in feature_cache:
            raise RuntimeError(
                f"Feature generation failed for record {item['record_id']}"
            )
        graph, _ = build_graph(
            item["graph_molecule"],
            feature_cache[descriptor_smiles],
            item["graph_labels"],
        )
        graph.y = graph.y_node.clone().long()
        graph.candidate_mask = graph.y.ge(0)

        protein = protein_dict[item["enzyme"]]["features"][:max_prot_len]
        protein = torch.as_tensor(protein, dtype=torch.float32)
        token = str(item["record_id"])
        samples.append((graph, protein, token, item["substrate_label"]))
        metadata[token] = {
            key: item[key]
            for key in (
                "record_id", "source", "family", "enzyme", "name", "smiles",
                "molecule_id", "split", "substrate_label",
            )
        }
        metadata[token]["atom_symbols"] = [
            atom.GetSymbol() for atom in item["graph_molecule"].GetAtoms()
        ]
    return samples, metadata


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    node_dim: int,
    protein_dim: int,
    edge_dim: int,
    device: torch.device,
) -> SOMModel:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    head_hidden = [
        int(value) for value in str(config["head_hidden"]).split(",") if value.strip()
    ]
    if not head_hidden or head_hidden[0] != 64:
        raise ValueError(f"Expected first head dimension 64, got {head_hidden}")

    model = SOMModel(
        node_dim,
        protein_dim,
        int(config["gnn_hidden"]),
        head_hidden,
        float(config["dropout"]),
        int(config["prot_kernel"]),
        edge_dim,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def extract_rows(model, loader, metadata, device):
    output_rows = []
    captured = {}

    def capture_192d(_module, inputs, _output):
        captured["features"] = inputs[0].detach()

    # Capture g3 at the input of head.net = Linear(192,64), LayerNorm, ...
    # g3 has already been conditioned on the enzyme protein through two FiLM blocks.
    hook = model.head.net[0].register_forward_hook(capture_192d)
    try:
        for batch, protein, protein_mask, tokens, _substrate_labels in loader:
            batch = batch.to(device)
            protein = protein.to(device)
            protein_mask = protein_mask.to(device)
            captured.clear()
            model(batch, protein, protein_mask)
            features = captured.get("features")
            if (
                features is None
                or features.ndim != 2
                or features.shape[1] != FEATURE_DIM
            ):
                raise RuntimeError(
                    "Failed to capture 192D enzyme-conditioned graph features"
                )

            features = features.float().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()
            candidates = batch.candidate_mask.detach().cpu().numpy().astype(bool)
            pointers = batch.ptr.detach().cpu().tolist()

            for graph_index, token in enumerate(tokens):
                start, end = pointers[graph_index], pointers[graph_index + 1]
                meta = metadata[token]
                for local_index in np.flatnonzero(candidates[start:end]):
                    feature = features[start + local_index]
                    row = {
                        "record_id": meta["record_id"],
                        "source": meta["source"],
                        "family": meta["family"],
                        "enzyme": meta["enzyme"],
                        "name": meta["name"],
                        "smiles": meta["smiles"],
                        "molecule_id": meta["molecule_id"],
                        "split": meta["split"],
                        "substrate_label": meta["substrate_label"],
                        "atom_index": int(local_index),
                        "atom_symbol": meta["atom_symbols"][local_index],
                        "label": int(labels[start + local_index]),
                    }
                    row.update(zip(FEATURE_COLUMNS, feature.tolist()))
                    output_rows.append(row)
    finally:
        hook.remove()
    return output_rows


def main() -> None:
    args = parse_args()
    required = {"Enzyme", "SMILES", "SoM", "label", "split"}
    frame = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    protein_dict = load_protein_dict(str(args.protein_pkl))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parsed, descriptor_items = parse_records(frame, protein_dict)
    feature_cache, errors = build_node_feature_cache(
        descriptor_items, num_workers=args.feature_workers
    )
    if errors:
        raise RuntimeError(f"Molecular feature errors: {errors}")

    samples, metadata = build_samples(
        parsed, feature_cache, protein_dict, int(config["max_prot_len"])
    )
    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=som_collate,
        drop_last=False,
    )

    first_graph, first_protein, _, _ = samples[0]
    node_dim = int(first_graph.x.shape[1])
    protein_dim = int(first_protein.shape[1])
    edge_dim = int(first_graph.edge_attr.shape[1])
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    model = load_model(
        args.checkpoint, args.config, node_dim, protein_dim, edge_dim, device
    )

    rows = extract_rows(model, loader, metadata, device)
    output = pd.DataFrame(rows)
    output = output.sort_values(["family", "record_id", "atom_index"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, float_format="%.8g")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Feature dimension: {len(FEATURE_COLUMNS)}")
    print(
        output.groupby(["family", "split"])
        .agg(sites=("label", "size"), som=("label", "sum"))
        .assign(non_som=lambda value: value["sites"] - value["som"])
        .to_string()
    )
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
