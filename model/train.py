# -*- coding: utf-8 -*-
import os
import argparse
import json
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

from muon import SingleDeviceMuonWithAuxAdam
from utils import (
    EMA,
    EarlyStopping,
    SOMDataset,
    _safe_div,
    build_ec_first_targets,
    compute_metrics_sklearn,
    compute_topk_hit_per_graph,
    format_metrics,
    load_mol_payload,
    load_protein_dict,
    seed_all,
    som_collate,
)

sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.numeric"] = np.core.numeric
warnings.filterwarnings("ignore")

EC_NUM_CLASSES = 6

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.1,
        norm: bool = True,
    ):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.extend([nn.GELU(), nn.Dropout(dropout)])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProteinEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, prot_pad: torch.Tensor, prot_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PlainCNNBlock1D(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5, dropout: float = 0.1):
        super().__init__()
        assert k % 2 == 1
        self.norm = nn.LayerNorm(c_in)
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Identity() if c_in == c_out else nn.Linear(c_in, c_out)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask_f = None if mask is None else mask.unsqueeze(-1).to(x.dtype)
        if mask_f is not None:
            x = x * mask_f

        residual = self.skip(x)
        h = self.norm(x)
        if mask_f is not None:
            h = h * mask_f

        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        x = residual + self.drop(self.act(h))
        return x if mask_f is None else x * mask_f


class ProteinCNNEncoder(ProteinEncoder):
    def __init__(self, c_in: int, c_hidden: int = 256, k: int = 5, dropout: float = 0.1):
        super().__init__(out_dim=c_hidden)
        self.c_hidden = c_hidden
        self.proj = nn.Linear(c_in, c_hidden)
        self.norm0 = nn.LayerNorm(c_hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            PlainCNNBlock1D(c_hidden, c_hidden, k=k, dropout=dropout) for _ in range(2)
        ])

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


    def encode_input(self, prot_pad: torch.Tensor, prot_mask: torch.Tensor) -> torch.Tensor:
        mask_f = prot_mask.unsqueeze(-1).to(dtype=prot_pad.dtype)
        x = self.drop(self.act(self.norm0(self.proj(prot_pad))))
        return x * mask_f

    def forward(self, prot_pad: torch.Tensor, prot_mask: torch.Tensor, return_multi: bool = False):
        x = self.encode_input(prot_pad, prot_mask)
        pooled, feats = {}, {}
        for i, block in enumerate(self.blocks, start=1):
            x = block(x, prot_mask)
            pooled[i] = self.masked_mean(x, prot_mask)
            feats[i] = x
        return (pooled, feats) if return_multi else pooled[2]


class _GMFBlock(nn.Module):
    def __init__(self, hidden=128, dropout=0.1, heads=4):
        super().__init__()
        self.dropout = dropout
        self.local_conv = GINEConv(nn=nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        ), edge_dim=hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.local_conv(x, edge_index, edge_attr)
        x = self.norm1(x + F.dropout(h, self.dropout, self.training))

        dense_x, mask = to_dense_batch(x, batch)
        h, _ = self.attn(dense_x, dense_x, dense_x, key_padding_mask=~mask,
                         need_weights=False)
        x = self.norm2(x + F.dropout(h[mask], self.dropout, self.training))

        return self.norm3(x + self.ffn(x))


class MolGraphEncoder(nn.Module):
    def __init__(self, x_in: int, edge_dim: int = 0, hidden: int = 256, num_layers: int = 3,
                 dropout: float = 0.1, gat_heads: int = 4):
        super().__init__()
        if edge_dim <= 0:
            raise ValueError(f"GMF(GINE) requires edge_attr with edge_dim > 0, but got {edge_dim}")

        self.hidden = hidden
        self.num_layers = num_layers
        self.node_proj = nn.Identity() if x_in == hidden else nn.Linear(x_in, hidden)
        self.edge_encoder = nn.Linear(edge_dim, hidden)
        self.blocks = nn.ModuleList([
            _GMFBlock(hidden=hidden, dropout=dropout, heads=gat_heads) for _ in range(num_layers)
        ])

    def encode_input(self, x: torch.Tensor, edge_attr: Optional[torch.Tensor]):
        if edge_attr is None or edge_attr.numel() == 0:
            raise ValueError("GMF(GINE) needs non-empty edge_attr.")
        return self.node_proj(x), self.edge_encoder(edge_attr)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            raise ValueError("GMF encoder needs `batch` to separate graphs.")

        x, edge_attr = self.encode_input(x, edge_attr)
        for block in self.blocks:
            x = block(x, edge_index, edge_attr, batch)
        return x


class SOMModel(nn.Module):
    def __init__(self, node_in_dim: int, prot_in_dim: int, gnn_hidden: int = 256,
                 head_hidden: List[int] = [128, 64], dropout: float = 0.1,
                 prot_kernel: int = 11, edge_dim: int = 0):
        super().__init__()
        self.hidden = gnn_hidden
        self.gnn = MolGraphEncoder(node_in_dim, edge_dim=edge_dim, hidden=gnn_hidden,
                                   num_layers=3, dropout=dropout)
        self.prot_enc = ProteinCNNEncoder(prot_in_dim, c_hidden=gnn_hidden, k=prot_kernel,
                                          dropout=dropout)
        self.film_mlps = nn.ModuleDict({
            "c1_to_g2": self._make_film_mlp(gnn_hidden, gnn_hidden * 2, dropout),
            "c2_to_g3": self._make_film_mlp(gnn_hidden, gnn_hidden * 2, dropout),
        })
        self.head = MLP(gnn_hidden, head_hidden, 1, dropout=dropout, norm=True)
        self.ec_head = MLP(gnn_hidden, head_hidden, EC_NUM_CLASSES, dropout=dropout, norm=True)

        for mlp in self.film_mlps.values():
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    @staticmethod
    def _make_film_mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim * 2), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(in_dim * 2, out_dim))

    @staticmethod
    def _film_graph(x: torch.Tensor, cond: torch.Tensor, film: nn.Module,
                    batch_index: torch.Tensor) -> torch.Tensor:
        gamma_raw, beta = film(cond).chunk(2, dim=-1)
        gamma, beta = (1.0 + gamma_raw)[batch_index], beta[batch_index]
        return gamma * x + beta

    def forward(self, batch: Batch, prot_pad: torch.Tensor, prot_mask: torch.Tensor,
                return_ec_logits: bool = False):
        edge_attr = getattr(batch, "edge_attr", None)
        node_x, edge_h = self.gnn.encode_input(batch.x, edge_attr)
        prot_x = self.prot_enc.encode_input(prot_pad, prot_mask)

        g1 = self.gnn.blocks[0](node_x, batch.edge_index, edge_h, batch.batch)
        c1 = self.prot_enc.blocks[0](prot_x, prot_mask)
        c1_pool = self.prot_enc.masked_mean(c1, prot_mask)

        g2_in = self._film_graph(g1, c1_pool, self.film_mlps["c1_to_g2"], batch.batch)
        c2 = self.prot_enc.blocks[1](c1, prot_mask)
        g2 = self.gnn.blocks[1](g2_in, batch.edge_index, edge_h, batch.batch)
        c2_pool = self.prot_enc.masked_mean(c2, prot_mask)

        g3_in = self._film_graph(g2, c2_pool, self.film_mlps["c2_to_g3"], batch.batch)
        g3 = self.gnn.blocks[2](g3_in, batch.edge_index, edge_h, batch.batch)

        logits = self.head(g3).view(-1)
        if not return_ec_logits:
            return logits

        ec_logits = self.ec_head(c2_pool)
        return logits, ec_logits


def graph_max_logits(
    batch: Batch,
    node_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dense, atom_mask = to_dense_batch(
        node_logits, batch.batch, fill_value=-torch.inf
    )
    candidate_mask, _ = to_dense_batch(
        batch.candidate_mask.bool(), batch.batch, fill_value=False
    )
    valid = atom_mask & candidate_mask
    graph_valid = valid.any(dim=1)
    graph_logits = dense.masked_fill(~valid, -torch.inf).max(dim=1).values
    return graph_logits, graph_valid


def masked_som_loss(node_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels >= 0
    if not valid.any():
        return node_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(node_logits[valid], labels[valid])


def som_substrate_loss(
    batch: Batch,
    node_logits: torch.Tensor,
    node_labels: torch.Tensor,
    substrate_labels: torch.Tensor,
    neg_pos_ratio: float = 2.0,
    training: bool = True,
    hard_frac: float = 0.5,
):
    substrate_logits, substrate_valid = graph_max_logits(batch, node_logits)
    zero = node_logits.sum() * 0.0
    if substrate_valid.any():
        sub_logits = substrate_logits[substrate_valid]
        sub_labels = substrate_labels[substrate_valid]
        loss_substrate = (
            sub_labels * F.relu(-sub_logits)
            + (1.0 - sub_labels) * F.relu(sub_logits)
        ).mean()
    else:
        loss_substrate = zero

    if not training:
        return (masked_som_loss(node_logits, node_labels),
                loss_substrate, substrate_logits, substrate_valid)

    valid = node_labels >= 0
    if not valid.any():
        return zero, loss_substrate, substrate_logits, substrate_valid

    logits, y = node_logits[valid], node_labels[valid]
    pos_logits, neg_logits = logits[y == 1], logits[y == 0]

    if not pos_logits.numel():
        loss_som = F.softplus(neg_logits).mean()
    elif not neg_logits.numel():
        loss_som = F.softplus(-pos_logits).mean()
    else:
        neg_loss = F.softplus(neg_logits)
        n_keep = min(neg_logits.numel(), round(neg_pos_ratio * pos_logits.numel()))
        n_hard = round(hard_frac * n_keep)
        n_random = n_keep - n_hard

        hard_idx = torch.topk(neg_loss.detach(), n_hard).indices
        remain = torch.ones(neg_logits.numel(), dtype=torch.bool, device=logits.device)
        remain[hard_idx] = False
        remain_idx = remain.nonzero(as_tuple=True)[0]
        random_idx = remain_idx[
            torch.randperm(remain_idx.numel(), device=logits.device)[:n_random]
        ]

        neg_logits = neg_logits[torch.cat([hard_idx, random_idx])]
        pos_loss = F.softplus(-pos_logits).mean()
        neg_loss = F.softplus(neg_logits).mean()
        loss_som = 0.5 * pos_loss + 0.5 * neg_loss

    return loss_som, loss_substrate, substrate_logits, substrate_valid


def compute_ec_loss(
    ec_logits: torch.Tensor,
    uids: List[str],
    prot_dict: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    targets, valid = build_ec_first_targets(uids, prot_dict, device)
    if not valid.any():
        return ec_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(ec_logits[valid], targets[valid])


def build_metrics(
    node_logits: torch.Tensor,
    node_labels: torch.Tensor,
    node_substrate: torch.Tensor,
    substrate_logits: torch.Tensor,
    substrate_labels: torch.Tensor,
) -> Dict[str, float]:
    out = {}

    for prefix, logits, labels in (
        ("substrate", substrate_logits, substrate_labels),
        ("som_sub", node_logits[node_substrate.eq(1)], node_labels[node_substrate.eq(1)]),
        ("som_all", node_logits, node_labels),
    ):
        out.update({
            f"{prefix}_{key}": value
            for key, value in compute_metrics_sklearn(logits, labels).items()
        })

    return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prot_dict: Dict[str, Any],
    ec_loss_weight: float,
    neg_pos_ratio: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    node_logits_all, node_labels_all, node_substrate_all = [], [], []
    substrate_logits_all, substrate_labels_all = [], []
    losses = {k: torch.zeros((), device=device)
              for k in ("loss", "substrate_loss", "som_loss", "ec_loss")}
    topk = {key: torch.zeros(2, dtype=torch.long, device=device)
            for key in ("top1", "top2", "top3")}

    for batch, prot_pad, prot_mask, uids, substrate_labels in loader:
        batch = batch.to(device, non_blocking=True)
        prot_pad, prot_mask = prot_pad.to(device, non_blocking=True), prot_mask.to(device, non_blocking=True)
        substrate_labels = substrate_labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            node_logits, ec_logits = model(batch, prot_pad, prot_mask, return_ec_logits=True)
            node_labels = batch.y.float().view(-1)

            loss_som, loss_substrate, substrate_logits, substrate_valid = som_substrate_loss(
                batch, node_logits, node_labels, substrate_labels,
                neg_pos_ratio=neg_pos_ratio, training=training)

            loss_ec = compute_ec_loss(ec_logits, uids, prot_dict, device)
            loss = loss_som + ec_loss_weight * loss_ec

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

        for k, v in (("loss", loss), ("substrate_loss", loss_substrate),
                     ("som_loss", loss_som), ("ec_loss", loss_ec)):
            losses[k] += v.detach()

        node_substrate = substrate_labels[batch.batch]
        node_valid = node_labels >= 0
        node_logits_all.append(node_logits[node_valid].detach())
        node_labels_all.append(node_labels[node_valid].detach())
        node_substrate_all.append(node_substrate[node_valid].detach())
        substrate_logits_all.append(substrate_logits[substrate_valid].detach())
        substrate_labels_all.append(substrate_labels[substrate_valid].detach())

        batch_topk = compute_topk_hit_per_graph(
            batch,
            node_logits,
            substrate_labels,
            return_tensors=True,
        )
        for key, (hits, total) in batch_topk.items():
            topk[key] += torch.stack((hits, total))

    out = build_metrics(
        torch.cat(node_logits_all),
        torch.cat(node_labels_all),
        torch.cat(node_substrate_all),
        torch.cat(substrate_logits_all),
        torch.cat(substrate_labels_all),
    )
    out.update({key: value.item() / len(loader) for key, value in losses.items()})
    out.update({
        f"som_sub_{key}": _safe_div(counts[0].item(), counts[1].item())
        for key, counts in topk.items()
    })
    return out


def build_loaders(
    mol_pkl: str,
    protein_pkl: str,
    split_mode: str,
    max_prot_len: int,
    batch_size: int,
    seed: int,
):
    mol_payload = load_mol_payload(mol_pkl)
    prot_dict = load_protein_dict(protein_pkl)

    ds_train = SOMDataset(mol_payload, prot_dict, split_mode, "train", max_prot_len)
    ds_valid = SOMDataset(mol_payload, prot_dict, split_mode, "valid", max_prot_len)
    ds_test = SOMDataset(mol_payload, prot_dict, split_mode, "test", max_prot_len)

    g = torch.Generator()
    g.manual_seed(seed)
    loader_kwargs = dict(
        pin_memory=True,
        collate_fn=som_collate,
        drop_last=False,
        generator=g,
    )

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, **loader_kwargs)
    loader_valid = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=0, **loader_kwargs)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, **loader_kwargs)

    g0, prot0, _uid0, _substrate0 = ds_train[0]
    node_in_dim = int(g0.x.shape[1])
    prot_in_dim = int(prot0.shape[1])

    edge_dim = 0
    if hasattr(g0, "edge_attr") and g0.edge_attr is not None and g0.edge_attr.numel() > 0:
        edge_dim = int(g0.edge_attr.shape[1])

    return loader_train, loader_valid, loader_test, node_in_dim, prot_in_dim, edge_dim, prot_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--mol_pkl", type=str, default="features/molecule/pretrain_features1.pkl")
    ap.add_argument("--protein_pkl", type=str, default="features/protein/protein_pretrain.pkl")
    ap.add_argument("--split_mode", type=str, default="split_scaffold",
                    choices=["split_smiles_group", "split_enzyme_cluster", "split_scaffold"])

    ap.add_argument("--max_prot_len", type=int, default=1024)

    ap.add_argument("--gnn_hidden", type=int, default=192)
    ap.add_argument("--head_hidden", type=str, default="64,32")
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--prot_kernel", type=int, default=5)

    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ec_loss_weight", type=float, default=1.0)
    ap.add_argument("--neg_pos_ratio", type=float, default=2.0)

    ap.add_argument("--early_patience", type=int, default=3)
    ap.add_argument("--early_metric", default="som_all_auprc",
                    choices=["substrate_auprc", "substrate_mcc",
                             "som_sub_auprc", "som_sub_mcc",
                             "som_all_auprc", "som_all_mcc"])

    ap.add_argument("--save_dir", type=str, default="runs_pretrain/split_scaffold")
    ap.add_argument("--run_name", type=str, default="exp_4")

    args = ap.parse_args()

    seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    loader_train, loader_valid, loader_test, node_in_dim, prot_in_dim, edge_dim, prot_dict = build_loaders(
        args.mol_pkl,
        args.protein_pkl,
        args.split_mode,
        args.max_prot_len,
        args.batch_size,
        args.seed,
    )

    head_hidden = [int(x) for x in args.head_hidden.split(",") if x.strip()]

    model = SOMModel(
        node_in_dim=node_in_dim,
        prot_in_dim=prot_in_dim,
        gnn_hidden=args.gnn_hidden,
        head_hidden=head_hidden,
        dropout=args.dropout,
        prot_kernel=args.prot_kernel,
        edge_dim=edge_dim,
    ).to(device)

    backbone_prefixes = ("gnn.", "prot_enc.", "film_mlps.")
    muon_params, adamw_params = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(backbone_prefixes) and p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    param_groups = []

    if adamw_params:
        param_groups.append(dict(
            params=adamw_params, lr=args.lr, betas=(0.9, 0.95),
            eps=1e-10, weight_decay=args.weight_decay, use_muon=False,
        ))

    if muon_params:
        param_groups.append(dict(
            params=muon_params, lr=args.lr, momentum=0.95,
            weight_decay=args.weight_decay, use_muon=True,
        ))

    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    ema = EMA(model, decay=0.995)

    def lr_lambda(epoch_idx: int):
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,)
    early = EarlyStopping(patience=args.early_patience, min_delta=1e-4,)

    best_ckpt = os.path.join(run_dir, "best.pt")
    last_ckpt = os.path.join(run_dir, "last.pt")

    print("==== Run Info ====")
    print(f"run_dir: {run_dir}")
    print(f"split_mode: {args.split_mode}")
    print(f"dims: node_in={node_in_dim} prot_in={prot_in_dim}")
    print(f"device: {device}")
    print("optimizer: SingleDeviceMuonWithAuxAdam")
    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adamw_params):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        train_m = run_epoch(
            model, loader_train, device, prot_dict,
            args.ec_loss_weight, args.neg_pos_ratio, optimizer, ema,
        )

        ema.apply(model)

        val_m = run_epoch(
            model, loader_valid, device, prot_dict,
            args.ec_loss_weight, args.neg_pos_ratio,
        )
        test_m = run_epoch(
            model, loader_test, device, prot_dict,
            args.ec_loss_weight, args.neg_pos_ratio,
        )

        key = args.early_metric
        score = float(val_m.get(key, 0.0))
        improved = score > early.best + early.min_delta
        stop = early.step(score, model)

        if improved:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, best_ckpt)

        ema.restore(model)

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": {
                k: v.detach().cpu().clone()
                for k, v in ema.shadow.items()
            },
            "optimizer": optimizer.state_dict(),
        }, last_ckpt)

        dt = time.time() - t0

        print(f"\n[Epoch {epoch:02d}] time={dt:.1f}s lr={curr_lr:.2e}")

        for name, metrics in (
            ("train", train_m),
            ("val", val_m),
            ("test", test_m),
        ):
            text = format_metrics(metrics).replace("\n", "\n    ")
            print(f"  {name}:\n    {text}")

        print(
            f"  early({key})={score:.5f} best={early.best:.5f} "
            f"bad={early.bad_epochs}/{early.patience}"
        )

        if stop:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Restoring best EMA model "
                f"(best {key}={early.best:.4f})."
            )
            break

        scheduler.step()

    early.restore(model)

    final_val = run_epoch(
        model, loader_valid, device, prot_dict,
        args.ec_loss_weight, args.neg_pos_ratio,
    )
    final_test = run_epoch(
        model, loader_test, device, prot_dict,
        args.ec_loss_weight, args.neg_pos_ratio,
    )

    print("==== Final (Best Restored) ====")

    for name, metrics in (
        ("val", final_val),
        ("test", final_test),
    ):
        text = format_metrics(metrics).replace("\n", "\n    ")
        print(f"{name}:\n    {text}")

    print(f"Saved best checkpoint: {best_ckpt}")
    print(f"Saved last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
