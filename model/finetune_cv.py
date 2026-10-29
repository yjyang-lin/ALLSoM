import argparse
import json
import os
import statistics
import time
from collections import Counter

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from muon import SingleDeviceMuonWithAuxAdam
from train import SOMModel, run_epoch
from utils import SOMDataset, format_metrics, load_mol_payload, load_protein_dict, seed_all, som_collate


TRANSFER_PREFIXES = ("gnn.", "prot_enc.", "film_mlps.", "head.")


def load_pretrained(model, ckpt_path, device):
    """Load the pretrained encoders, protein conditioning, and SoM head."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_state = checkpoint.get("model", checkpoint)
    candidate_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key.startswith(TRANSFER_PREFIXES)
    }
    expected = {
        key for key in model.state_dict() if key.startswith(TRANSFER_PREFIXES)
    }
    transfer_state = {
        key: value for key, value in candidate_state.items() if key in expected
    }
    missing = sorted(expected - set(transfer_state))
    if missing:
        raise RuntimeError(
            "Pretrained weights do not match: "
            f"missing={missing}"
        )
    model.load_state_dict(transfer_state, strict=False)


def _label(record):
    relations = record.get("protein_records", [])
    return int(relations[0]["substrate_label"])


def _group(record, split_mode):
    smiles = record.get("clean_smiles") or record.get("smiles")
    if not smiles:
        raise ValueError(f"Record sdf_idx={record.get('sdf_idx')} is missing SMILES")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if split_mode == "smiles":
        return canonical
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    # Acyclic molecules have an empty Murcko scaffold; keeping each canonical
    # SMILES as its own group avoids putting all acyclic compounds in one fold.
    return scaffold or f"NO_SCAFFOLD::{canonical}"


def make_folds(records, split_mode, n_splits, seed):
    y = np.asarray([_label(record) for record in records])
    groups = np.asarray([_group(record, split_mode) for record in records])
    if len(set(groups)) < n_splits:
        raise ValueError(
            f"Only {len(set(groups))} distinct groups are available; "
            f"cannot perform {n_splits}-fold cross-validation"
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(records)), y, groups)), y, groups


def make_dataset(records, indices, prot_dict, max_prot_len):
    dataset = SOMDataset.__new__(SOMDataset)
    dataset.prot_dict = prot_dict
    dataset.max_prot_len = max_prot_len
    dataset.samples = [
        (records[i], relation)
        for i in indices
        for relation in records[i].get("protein_records", [])
    ]
    if not dataset.samples:
        raise ValueError("The five-fold split produced an empty dataset")
    return dataset


def make_loaders(records, train_idx, test_idx, prot_dict, args, seed):
    datasets = [make_dataset(records, idx, prot_dict, args.max_prot_len)
                for idx in (train_idx, test_idx)]
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True,
                  drop_last=False, collate_fn=som_collate, generator=generator)
    loaders = [DataLoader(ds, shuffle=(i == 0), **common) for i, ds in enumerate(datasets)]
    graph, protein, _, _ = datasets[0][0]
    edge_dim = int(graph.edge_attr.shape[1]) if getattr(graph, "edge_attr", None) is not None and graph.edge_attr.numel() else 0
    return (*loaders, int(graph.x.shape[1]), int(protein.shape[1]), edge_dim)


def run_fold(args, fold, records, train_idx, test_idx, prot_dict):
    fold_seed = args.seed + fold
    seed_all(fold_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.save_root, args.split_mode, f"fold_{fold}")
    os.makedirs(run_dir, exist_ok=True)
    loaders = make_loaders(records, train_idx, test_idx, prot_dict, args, fold_seed)
    train_loader, test_loader, node_dim, prot_dim, edge_dim = loaders

    model = SOMModel(
        node_in_dim=node_dim,
        prot_in_dim=prot_dim,
        gnn_hidden=args.gnn_hidden,
        head_hidden=[int(x) for x in args.head_hidden.split(",") if x.strip()],
        dropout=args.dropout,
        prot_kernel=args.prot_kernel,
        edge_dim=edge_dim,
    ).to(device)
    load_pretrained(model, args.pretrain_ckpt, device)

    muon_params, adamw_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = (
                muon_params
                if name.startswith(("gnn.", "prot_enc.", "film_mlps."))
                and parameter.ndim >= 2
                else adamw_params
            )
            target.append(parameter)

    param_groups = []
    if adamw_params:
        param_groups.append(dict(
            params=adamw_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=args.weight_decay,
            use_muon=False,
        ))
    if muon_params:
        param_groups.append(dict(
            params=muon_params,
            lr=args.lr,
            momentum=0.95,
            weight_decay=args.weight_decay,
            use_muon=True,
        ))
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    best_path = os.path.join(run_dir, "best.pt")

    config = vars(args).copy()
    config.update(fold=fold, fold_seed=fold_seed, n_train=len(train_idx),
                  n_test=len(test_idx))
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            prot_dict=prot_dict,
            ec_loss_weight=0.0,
            neg_pos_ratio=args.neg_pos_ratio,
            optimizer=optimizer,
        )
        print(f"[fold {fold}/{args.n_splits}][epoch {epoch:02d}] {time.time()-started:.1f}s "
              f"train_loss={train_metrics.get('loss', 0):.5f}")

    model_state = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith("ec_head.")
    }
    torch.save({"epoch": args.epochs, "model": model_state,
                "optimizer": optimizer.state_dict()}, best_path)
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        prot_dict=prot_dict,
        ec_loss_weight=0.0,
        neg_pos_ratio=args.neg_pos_ratio,
    )
    print("test:\n    " + format_metrics(test_metrics).replace("\n", "\n    "))
    return {"test": test_metrics}


def main():
    parser = argparse.ArgumentParser(description="AOX grouped five-fold cross-validation")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mol_pkl", default="features/molecule/AOX_features.pkl")
    parser.add_argument("--protein_pkl", default="features/protein/protein_metab.pkl")
    parser.add_argument("--split_mode", choices=["smiles", "scaffold"], default="smiles")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_prot_len", type=int, default=1024)
    parser.add_argument("--gnn_hidden", type=int, default=192)
    parser.add_argument("--head_hidden", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--prot_kernel", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--neg_pos_ratio", type=float, default=2.0)
    parser.add_argument("--pretrain_ckpt", default="runs_pretrain/split_enzyme_cluster/exp_1/best.pt")
    parser.add_argument("--save_root", default="runs_finetune/AOX_transfer/split_enzyme_cluster/exp_1")
    args = parser.parse_args()

    if not os.path.exists(args.pretrain_ckpt):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrain_ckpt}")

    payload, prot_dict = load_mol_payload(args.mol_pkl), load_protein_dict(args.protein_pkl)
    records = payload["data"]
    folds, labels, groups = make_folds(records, args.split_mode, args.n_splits, args.seed)
    print(f"AOX records={len(records)}, isoforms={sorted({r['uniprot_id'] for x in records for r in x['protein_records']})}, "
          f"groups={len(set(groups))}, labels={dict(Counter(labels))}")

    results = []
    for fold_no, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\nfold {fold_no}: train={len(train_idx)} test={len(test_idx)}")
        results.append(run_fold(args, fold_no, records, train_idx, test_idx, prot_dict))

    print("\n5-FOLD TEST SUMMARY (mean ± sample std)")
    for key in results[0]["test"]:
        values = [result["test"][key] for result in results]
        print(f"  {key}: {statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}")


if __name__ == "__main__":
    main()
