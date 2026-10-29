# -*- coding: utf-8 -*-
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Dataset

sys.path[:0] = [".", "model"]

from featurize.mol_featurize import build_graph, build_node_feature_cache, prepare_molecule_and_label
from train import SOMModel, build_metrics, graph_max_logits, masked_som_loss
from utils import _safe_div, compute_topk_hit_per_graph, format_metrics, load_protein_dict, som_collate

sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.numeric"] = np.core.numeric
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def cyp_labels(mol: Chem.Mol, substrate_label: int) -> List:
    mapped = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1}
    if substrate_label == 0:
        return [0] * mol.GetNumAtoms()
    if not mapped:
        return ["mask"] * mol.GetNumAtoms()
    return [int(index in mapped) for index in range(mol.GetNumAtoms())]


class CYPExternalDataset(Dataset):
    def __init__(self, samples, protein_dict, max_prot_len):
        self.samples, self.protein_dict = samples, protein_dict
        self.max_prot_len = max_prot_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        graph, uid, substrate_label = self.samples[index]
        graph = graph.clone()
        graph.y = graph.y_node.clone().long()
        graph.candidate_mask = torch.ones_like(graph.y, dtype=torch.bool)
        protein = torch.as_tensor(self.protein_dict[uid]["features"][:self.max_prot_len],
                                  dtype=torch.float32)
        return graph, protein, uid, substrate_label


def build_dataset(csv_path, protein_dict, max_prot_len, feature_workers):
    frame = pd.read_csv(csv_path)
    parsed, descriptor_items = [], {}
    for row in frame.itertuples(index=False):
        uid, substrate_label = str(row.uniprot_id).strip(), int(row.label)
        mol = Chem.MolFromSmiles(str(row.som_smiles).strip())
        graph_mol, labels, clean_smiles, descriptor_smiles = prepare_molecule_and_label(
            mol, cyp_labels(mol, substrate_label))
        parsed.append((graph_mol, labels, descriptor_smiles, uid, substrate_label))
        descriptor_items[descriptor_smiles] = clean_smiles

    feature_cache, _ = build_node_feature_cache(descriptor_items, num_workers=feature_workers)
    samples = []
    for mol, labels, descriptor_smiles, uid, substrate_label in parsed:
        graph, _ = build_graph(mol, feature_cache[descriptor_smiles], labels)
        samples.append((graph, uid, substrate_label))
    return CYPExternalDataset(samples, protein_dict, max_prot_len), frame


def load_checkpoints(model_dir):
    return [torch.load(model_dir / f"exp_{seed + 1}" / "best.pt", map_location="cpu")
            for seed in range(3)]


def build_models(checkpoints, node_dim, protein_dim, edge_dim, device):
    models = []
    for checkpoint in checkpoints:
        config = checkpoint["metadata"]
        head_hidden = [int(value) for value in config["head_hidden"].split(",")]
        model = SOMModel(node_dim, protein_dim, config["gnn_hidden"], head_hidden,
                         config["dropout"], config["prot_kernel"], edge_dim).to(device)
        del model.ec_head
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)
    return models


@torch.inference_mode()
def evaluate(models: List[nn.Module], loader: DataLoader,
             device: torch.device) -> tuple[Dict[str, float], np.ndarray]:
    node_logits_all, node_labels_all, node_substrate_all = [], [], []
    substrate_logits_all, substrate_labels_all, substrate_probs_all = [], [], []
    losses = {key: torch.zeros((), device=device)
              for key in ("loss", "substrate_loss", "som_loss")}
    topk = {f"top{k}": [0, 0] for k in (1, 2, 3)}

    for batch, protein, protein_mask, _uids, substrate_labels in loader:
        batch, protein = batch.to(device), protein.to(device)
        protein_mask, substrate_labels = protein_mask.to(device), substrate_labels.to(device)
        node_probs = torch.stack([
            torch.sigmoid(model(batch, protein, protein_mask)) for model in models
        ]).mean(dim=0)
        node_logits = torch.logit(node_probs.clamp(1e-7, 1 - 1e-7))
        node_labels = batch.y.float().view(-1)
        substrate_logits, substrate_valid = graph_max_logits(batch, node_logits)
        substrate_probs = torch.sigmoid(substrate_logits[substrate_valid])
        substrate_loss = F.binary_cross_entropy_with_logits(
            substrate_logits[substrate_valid], substrate_labels[substrate_valid])
        som_loss = masked_som_loss(node_logits, node_labels)
        loss = 0.5 * substrate_loss + 0.5 * som_loss

        for key, value in (("loss", loss), ("substrate_loss", substrate_loss),
                           ("som_loss", som_loss)):
            losses[key] += value
        node_logits_all.append(node_logits.cpu())
        node_labels_all.append(node_labels.cpu())
        node_substrate_all.append(substrate_labels[batch.batch].cpu())
        substrate_logits_all.append(substrate_logits[substrate_valid].cpu())
        substrate_labels_all.append(substrate_labels[substrate_valid].cpu())
        substrate_probs_all.append(substrate_probs.cpu())

        for key, (hits, valid) in compute_topk_hit_per_graph(
                batch, node_logits, substrate_labels).items():
            topk[key][0] += hits
            topk[key][1] += valid

    metrics = build_metrics(torch.cat(node_logits_all), torch.cat(node_labels_all),
                            torch.cat(node_substrate_all), torch.cat(substrate_logits_all),
                            torch.cat(substrate_labels_all))
    metrics.update({key: value.item() / len(loader) for key, value in losses.items()})
    metrics.update({f"som_sub_{key}": _safe_div(hits, valid)
                    for key, (hits, valid) in topk.items()})
    return metrics, torch.cat(substrate_probs_all).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--csv", default="inference/CYP/CYP_ex_val.csv")
    parser.add_argument("--protein_pkl", default="features/protein/protein_metab.pkl")
    parser.add_argument("--model_dir",
                        default="runs_finetune/CYP/split_enzyme_cluster_to_split_smiles_group")
    parser.add_argument("--output_csv",
                        default="inference/CYP/CYP_external_validation_result.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_workers", type=int, default=4)
    args = parser.parse_args()

    device, model_dir = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"), Path(args.model_dir)
    checkpoints = load_checkpoints(model_dir)
    config = checkpoints[0]["metadata"]
    protein_dict = load_protein_dict(args.protein_pkl)
    dataset, frame = build_dataset(args.csv, protein_dict, config["max_prot_len"],
                                   args.feature_workers)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=True, collate_fn=som_collate, drop_last=False)

    graph, protein, _, _ = dataset[0]
    edge_dim = int(graph.edge_attr.shape[1]) if graph.edge_attr.numel() else 0
    models = build_models(checkpoints, int(graph.x.shape[1]), int(protein.shape[1]),
                          edge_dim, device)
    metrics, substrate_probs = evaluate(models, loader, device)

    positive = int(frame["label"].eq(1).sum())
    known_som = sum(any(atom.GetAtomMapNum() == 1
                        for atom in Chem.MolFromSmiles(smiles).GetAtoms())
                    for smiles in frame.loc[frame["label"].eq(1), "som_smiles"])
    print(f"External CYP validation: samples={len(dataset)}, positive={positive}, "
          f"negative={len(dataset) - positive}, som_known_positive={known_som}, "
          f"models={len(models)}, device={device}")
    print(format_metrics(metrics))

    output = frame.copy()
    output["substrate_probability"] = substrate_probs
    output["substrate_correct"] = ((substrate_probs >= 0.5)
                                     == output["label"].to_numpy()).astype(int)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
