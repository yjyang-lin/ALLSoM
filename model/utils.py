import os
import pickle
import random
import sys
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    jaccard_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.numeric"] = np.core.numeric

EC_NUM_CLASSES = 6
METRIC_KEYS = (
    "acc", "precision", "recall", "f1", "mcc",
    "spec", "jaccard", "auroc", "auprc",
)


def _normalize_ec_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return []
        return [x.strip() for x in value.split(";") if x.strip()]
    if isinstance(value, (list, tuple, set)):
        return [x for item in value for x in _normalize_ec_list(item)]
    return _normalize_ec_list(str(value))


def _get_ec_nums(item: Any) -> List[str]:
    if item is None:
        return []
    for key in ("ec_num", "ec_nums", "ec", "ecs"):
        if isinstance(item, dict) and key in item:
            return _normalize_ec_list(item[key])
        if hasattr(item, key):
            return _normalize_ec_list(getattr(item, key))
    return []


def build_ec_first_targets(
    uids: List[str],
    prot_dict: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.zeros(len(uids), EC_NUM_CLASSES, dtype=torch.float32)
    valid = torch.zeros(len(uids), dtype=torch.bool)

    for i, uid in enumerate(uids):
        for ec in _get_ec_nums(prot_dict.get(str(uid))):
            first = ec.split(".", 1)[0].strip()
            if first.isdigit() and 1 <= int(first) <= EC_NUM_CLASSES:
                targets[i, int(first) - 1] = 1.0
                valid[i] = True

    return targets.to(device, non_blocking=True), valid.to(device, non_blocking=True)


def seed_all(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def is_special_enzyme(uid: str) -> bool:
    return str(uid).strip().upper().startswith(("UGT", "AOX"))


@torch.no_grad()
def compute_metrics_sklearn(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    if logits.numel() == 0:
        return {key: 0.0 for key in METRIC_KEYS}

    probs = torch.sigmoid(logits).detach().float().cpu().numpy()
    y = labels.detach().long().cpu().numpy()
    pred = (probs >= 0.5).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2

    return {
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y, pred)) if two_classes else 0.0,
        "spec": _safe_div(tn, tn + fp),
        "jaccard": float(jaccard_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, probs)) if two_classes else 0.0,
        "auprc": float(average_precision_score(y, probs)) if two_classes else 0.0,
    }


@torch.no_grad()
def compute_topk_hit_per_graph(
    batch: Batch,
    node_scores: torch.Tensor,
    substrate_labels: torch.Tensor,
    k_list: List[int] = [1, 2, 3],
    return_tensors: bool = False,
) -> Dict[str, tuple[Any, Any]]:
    scores, atom_mask = to_dense_batch(
        node_scores.detach().view(-1), batch.batch, fill_value=-torch.inf
    )
    labels, _ = to_dense_batch(
        batch.y.detach().view(-1).long(), batch.batch, fill_value=-1
    )
    candidates, _ = to_dense_batch(
        batch.candidate_mask.detach().view(-1).bool(),
        batch.batch,
        fill_value=False,
    )
    valid = atom_mask & candidates & labels.ge(0)
    eligible = substrate_labels.detach().gt(0.5) & (valid & labels.eq(1)).any(dim=1)
    count = eligible.sum()

    max_k = min(max(k_list), scores.shape[1])
    ranked = scores.masked_fill(~valid, -torch.inf).topk(max_k, dim=1).indices
    positives = labels.gather(1, ranked).eq(1).cumsum(dim=1).gt(0)
    result = {
        f"top{k}": ((positives[:, min(k, max_k) - 1] & eligible).sum(), count)
        for k in k_list
    }
    if return_tensors:
        return result
    return {
        key: (int(hits.item()), int(total.item()))
        for key, (hits, total) in result.items()
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    losses = ("loss", "substrate_loss", "som_loss", "ec_loss")
    
    parts = [
        " ".join(f"{k}={metrics[k]:.5f}" for k in losses if k in metrics)
    ]

    for prefix in ("substrate", "som_sub", "som_all"):
        keys = list(METRIC_KEYS)
        if prefix == "som_sub":
            keys += ["top1", "top2", "top3"]
        values = [
            f"{key}={metrics[f'{prefix}_{key}']:.5f}"
            for key in keys
            if f"{prefix}_{key}" in metrics
        ]
        if values:
            parts.append(f"{prefix}[{' '.join(values)}]")

    return "\n".join(part for part in parts if part)


def _norm_split_name(value: Any) -> str:
    value = "" if value is None else str(value).strip().lower()
    return "valid" if value == "val" else value


def load_mol_payload(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"Bad molecule pkl format: {path}")
    return payload


def load_protein_dict(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Bad protein pkl format: {path}")
    return data


class SOMDataset(Dataset):
    def __init__(
        self,
        mol_payload: Dict[str, Any],
        prot_dict: Dict[str, Any],
        split_mode: str = "split_scaffold",
        split_name: str = "train",
        max_prot_len: int = 1024,
    ):
        if split_mode not in {
            "split_scaffold", "split_smiles_group", "split_enzyme_cluster"
        }:
            raise ValueError(f"Invalid split_mode: {split_mode}")

        self.prot_dict = prot_dict
        self.max_prot_len = max_prot_len
        split_name = _norm_split_name(split_name)
        self.samples = []

        for record in mol_payload["data"]:
            relations = record.get("protein_records", [])
            for relation in relations:
                split = (
                    relation.get(split_mode, "")
                    if split_mode == "split_enzyme_cluster"
                    else record.get(split_mode, "")
                )
                if _norm_split_name(split) == split_name:
                    self.samples.append((record, relation))

        if not self.samples:
            raise ValueError(
                f"No enzyme-molecule pairs for split={split_name}, "
                f"field={split_mode}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        record, relation = self.samples[idx]
        graph: Data = copy.copy(record["graph"])
        graph.y = torch.as_tensor(record["node_label"], dtype=torch.long)

        uid = str(relation["uniprot_id"]).strip()
        substrate_label = int(relation["substrate_label"])
        if substrate_label not in (0, 1):
            raise ValueError(f"Invalid substrate_label: {substrate_label}")
        if uid not in self.prot_dict:
            raise KeyError(f"uniprot_id={uid} not found in protein pkl")

        graph.candidate_mask = graph.y.ne(-1) if is_special_enzyme(uid) else torch.ones_like(graph.y, dtype=torch.bool)

        protein = self.prot_dict[uid]["features"]
        if isinstance(protein, torch.Tensor):
            protein = protein.detach().cpu()
            if protein.dtype != torch.float32:
                protein = protein.float()
            protein = protein[:self.max_prot_len]
        else:
            protein = torch.as_tensor(protein[:self.max_prot_len], dtype=torch.float32)

        return graph, protein, uid, substrate_label


# Keep protein tensors unpadded during collation and defer padding until they
# are moved to GPU. This avoids expensive CPU-side construction of a large
# padded tensor for every batch and reduces host-side memory-copy overhead.
# This implementation is intended for the current num_workers=0 pipeline.
def som_collate(batch_list):
    graphs, proteins, uids, substrate_labels = zip(*batch_list)
    batch = Batch.from_data_list(list(graphs))
    state = {}

    def build(device):
        if state:
            return
        lengths = [x.size(0) for x in proteins]
        B, Lmax, D = len(proteins), max(lengths), proteins[0].size(1)
        protein_pad = torch.zeros(B, Lmax, D, dtype=proteins[0].dtype, device=device)
        for i, x in enumerate(proteins):
            protein_pad[i, :x.size(0)].copy_(x)
        lengths = torch.tensor(lengths, device=device)
        protein_mask = torch.arange(Lmax, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        state["pad"], state["mask"] = protein_pad, protein_mask

    class LazyTensor:
        def __init__(self, key):
            self.key = key

        def to(self, device, non_blocking=False):
            build(device)
            return state[self.key]

    return batch, LazyTensor("pad"), LazyTensor("mask"), list(uids), torch.tensor(substrate_labels, dtype=torch.float32)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v, 1 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        state = model.state_dict()
        self.backup = {k: state[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            state[k].copy_(v)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        state = model.state_dict()
        for k, v in self.backup.items():
            state[k].copy_(v)
        self.backup = {}


@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 1e-4
    best: float = -1e18
    bad_epochs: int = 0
    best_state: Optional[Dict[str, Any]] = None
    warmup: int = 3
    _step_count: int = 0

    def step(self, metric: float, model: nn.Module) -> bool:
        self._step_count += 1

        if metric > self.best + self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        elif self._step_count > self.warmup:
            self.bad_epochs += 1

        return (
            self._step_count > self.warmup
            and self.bad_epochs >= self.patience
        )

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state, strict=True)
