import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Dataset


sys.path.insert(0, ".")
sys.path.insert(0, "model")

from featurize.mol_featurize import (
    build_graph,
    build_node_feature_cache,
    prepare_molecule_and_label,
)
from train import SOMModel, build_metrics, som_substrate_loss
from utils import (
    _safe_div,
    compute_topk_hit_per_graph,
    format_metrics,
    load_protein_dict,
    som_collate,
)


AOX_SMARTS = (
    "[$([cH]1:[n]:[a]:[a]:[a]:[a]:1),"
    "$([cH]1:[a]:[a]:[n]:[a]:[a]:1),"
    "$([cH]1:[n]:[a]:[a]:[a]:1)]",
)
AOX_PATTERNS = tuple(Chem.MolFromSmarts(value) for value in AOX_SMARTS)


def aox_labels(mol: Chem.Mol, substrate_label: int) -> list:
    candidates = {
        atom_idx
        for pattern in AOX_PATTERNS
        for match in mol.GetSubstructMatches(pattern)
        for atom_idx in match
    }
    mapped = {
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1
    }
    if substrate_label == 0:
        return [0 if i in candidates else "mask" for i in range(mol.GetNumAtoms())]
    if not mapped:
        return ["candidate" if i in candidates else "mask" for i in range(mol.GetNumAtoms())]
    return [
        1 if i in mapped else 0 if i in candidates else "mask"
        for i in range(mol.GetNumAtoms())
    ]


class ExternalDataset(Dataset):
    def __init__(self, samples, protein_dict, max_prot_len):
        self.samples = samples
        self.protein_dict = protein_dict
        self.max_prot_len = max_prot_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        graph, uid, substrate_label = self.samples[index]
        graph = graph.clone()
        graph.y = graph.y_node.clone().long()
        # AOX: -1 is outside the chemical candidate set; -2 remains a candidate.
        graph.candidate_mask = graph.y.ne(-1)
        protein = torch.as_tensor(
            self.protein_dict[uid]["features"][: self.max_prot_len],
            dtype=torch.float32,
        )
        return graph, protein, uid, substrate_label


def build_dataset(csv_path, protein_dict, max_prot_len, num_workers):
    frame = pd.read_csv(csv_path)
    required = {"uniprot_id", "som_smiles", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    parsed = []
    descriptor_items = {}
    for row_index, row in frame.iterrows():
        uid = str(row["uniprot_id"]).strip()
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"row {row_index}: label must be 0 or 1")
        if uid not in protein_dict:
            raise KeyError(f"row {row_index}: protein features not found for {uid}")
        mol = Chem.MolFromSmiles(str(row["som_smiles"]).strip())
        if mol is None:
            raise ValueError(f"row {row_index}: invalid som_smiles")
        labels = aox_labels(mol, label)
        graph_mol, node_labels, clean_smiles, descriptor_smiles = (
            prepare_molecule_and_label(mol, labels)
        )
        parsed.append((graph_mol, node_labels, descriptor_smiles, uid, label))
        descriptor_items[descriptor_smiles] = clean_smiles

    feature_cache, errors = build_node_feature_cache(descriptor_items, num_workers)
    samples = []
    for row_index, (mol, labels, descriptor_smiles, uid, label) in enumerate(parsed):
        if descriptor_smiles not in feature_cache:
            raise RuntimeError(
                f"row {row_index}: feature generation failed: "
                f"{errors.get(descriptor_smiles, 'unknown error')}"
            )
        graph, _ = build_graph(mol, feature_cache[descriptor_smiles], labels)
        samples.append((graph, uid, label))
    return ExternalDataset(samples, protein_dict, max_prot_len), frame


def load_model(checkpoint_path, config, dimensions, device):
    node_dim, protein_dim, edge_dim = dimensions
    model = SOMModel(
        node_dim,
        protein_dim,
        int(config["gnn_hidden"]),
        [int(x) for x in str(config["head_hidden"]).split(",") if x.strip()],
        float(config["dropout"]),
        int(config["prot_kernel"]),
        edge_dim,
    ).to(device)
    del model.ec_head
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    return model


@torch.inference_mode()
def evaluate(models, loader, device):
    node_logits_all, node_labels_all, node_substrate_all = [], [], []
    substrate_logits_all, substrate_labels_all = [], []
    som_loss_sum = substrate_loss_sum = 0.0
    topk = {f"top{k}": [0, 0] for k in (1, 2, 3)}

    for batch, protein, protein_mask, _uids, substrate_labels in loader:
        batch = batch.to(device)
        protein = protein.to(device)
        protein_mask = protein_mask.to(device)
        substrate_labels = substrate_labels.to(device)

        probabilities = torch.stack(
            [torch.sigmoid(model(batch, protein, protein_mask)) for model in models]
        ).mean(dim=0)
        probabilities = probabilities.clamp(1e-7, 1 - 1e-7)
        logits = torch.logit(probabilities)
        node_labels = batch.y.float().view(-1)
        som_loss, substrate_loss, substrate_logits, substrate_valid = (
            som_substrate_loss(
                batch,
                logits,
                node_labels,
                substrate_labels,
                training=False,
            )
        )
        som_loss_sum += som_loss.item()
        substrate_loss_sum += substrate_loss.item()

        node_valid = node_labels.ge(0)
        node_logits_all.append(logits[node_valid].cpu())
        node_labels_all.append(node_labels[node_valid].cpu())
        node_substrate_all.append(substrate_labels[batch.batch][node_valid].cpu())
        substrate_logits_all.append(substrate_logits[substrate_valid].cpu())
        substrate_labels_all.append(substrate_labels[substrate_valid].cpu())

        result = compute_topk_hit_per_graph(batch, logits, substrate_labels)
        for key, (hits, count) in result.items():
            topk[key][0] += hits
            topk[key][1] += count

    node_logits = torch.cat(node_logits_all)
    node_labels = torch.cat(node_labels_all)
    node_substrate = torch.cat(node_substrate_all)
    substrate_logits = torch.cat(substrate_logits_all)
    substrate_labels = torch.cat(substrate_labels_all)

    metrics = build_metrics(
        node_logits, node_labels, node_substrate,
        substrate_logits, substrate_labels,
    )
    metrics["som_loss"] = som_loss_sum / len(loader)
    metrics["substrate_loss"] = substrate_loss_sum / len(loader)
    metrics["ec_loss"] = 0.0
    metrics["loss"] = metrics["som_loss"]
    metrics.update(
        {f"som_sub_{key}": _safe_div(hits, count)
         for key, (hits, count) in topk.items()}
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="AOX external five-model ensemble evaluation")
    parser.add_argument("--csv", default="inference/AOX/AOX_ex_val.csv")
    parser.add_argument("--protein-pkl", default="features/protein/protein_metab.pkl")
    parser.add_argument(
        "--model-dir",
        default="runs_finetune/AOX_transfer/split_enzyme_cluster/exp_1/smiles",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4, help="CDK feature workers")
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_paths = [Path(args.model_dir) / f"fold_{i}/best.pt" for i in range(1, 6)]
    config_paths = [path.with_name("config.json") for path in checkpoint_paths]
    for path in checkpoint_paths + config_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    configs = [json.loads(path.read_text()) for path in config_paths]
    architecture_keys = ("max_prot_len", "gnn_hidden", "head_hidden", "dropout", "prot_kernel")
    for key in architecture_keys:
        if len({str(config[key]) for config in configs}) != 1:
            raise ValueError(f"fold configs disagree on {key}")

    protein_dict = load_protein_dict(args.protein_pkl)
    dataset, frame = build_dataset(
        args.csv, protein_dict, int(configs[0]["max_prot_len"]), args.num_workers
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=som_collate, drop_last=False,
    )
    graph0, protein0, _, _ = dataset[0]
    edge_dim = int(graph0.edge_attr.shape[1]) if graph0.edge_attr.numel() else 0
    dimensions = (int(graph0.x.shape[1]), int(protein0.shape[1]), edge_dim)
    device = torch.device(args.device)
    models = [load_model(path, config, dimensions, device)
              for path, config in zip(checkpoint_paths, configs)]

    print(
        f"External AOX validation: molecules={len(dataset)}, "
        f"positive={int((frame.label == 1).sum())}, negative={int((frame.label == 0).sum())}, "
        f"models={len(models)}, device={device}"
    )
    print(format_metrics(evaluate(models, loader, device)))


if __name__ == "__main__":
    main()
