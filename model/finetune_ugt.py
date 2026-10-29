import argparse
import csv
import multiprocessing as mp
import os
import statistics
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from muon import SingleDeviceMuonWithAuxAdam
from train import SOMModel, build_loaders, build_metrics, som_substrate_loss
from utils import (EMA, EarlyStopping, _safe_div, compute_topk_hit_per_graph,
                   format_metrics, seed_all)

warnings.filterwarnings("ignore")

PRETRAIN_SPLITS = ("split_smiles_group", "split_enzyme_cluster", "split_scaffold")
FINETUNE_SPLITS = ("split_smiles_group", "split_scaffold")
SEEDS = (0, 1, 2)
TRANSFER_PREFIXES = ("gnn.", "prot_enc.", "film_mlps.", "head.")


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              neg_pos_ratio: float, optimizer: Optional[torch.optim.Optimizer] = None,
              ema: Optional[EMA] = None) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    node_logits_all, node_labels_all, node_substrate_all = [], [], []
    substrate_logits_all, substrate_labels_all = [], []
    losses = {key: torch.zeros((), device=device)
              for key in ("loss", "substrate_loss", "som_loss")}
    topk = {f"top{k}": [0, 0] for k in (1, 2, 3)}

    for batch, prot_pad, prot_mask, _uids, substrate_labels in loader:
        batch = batch.to(device, non_blocking=True)
        prot_pad = prot_pad.to(device, non_blocking=True)
        prot_mask = prot_mask.to(device, non_blocking=True)
        substrate_labels = substrate_labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            node_logits = model(batch, prot_pad, prot_mask)
            node_labels = batch.y.float().view(-1)

            som_loss, substrate_loss, substrate_logits, substrate_valid = (
                som_substrate_loss(
                    batch,
                    node_logits,
                    node_labels,
                    substrate_labels,
                    neg_pos_ratio=neg_pos_ratio,
                    training=training,
                )
            )
            loss = som_loss

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                if ema is not None:
                    ema.update(model)

        for key, value in (
            ("loss", loss),
            ("substrate_loss", substrate_loss),
            ("som_loss", som_loss),
        ):
            losses[key] += value.detach()

        node_substrate = substrate_labels[batch.batch]
        node_valid = node_labels >= 0
        node_logits_all.append(node_logits[node_valid].detach().cpu())
        node_labels_all.append(node_labels[node_valid].detach().cpu())
        node_substrate_all.append(
            node_substrate[node_valid].detach().cpu())
        substrate_logits_all.append(
            substrate_logits[substrate_valid].detach().cpu())
        substrate_labels_all.append(
            substrate_labels[substrate_valid].detach().cpu())

        for key, (hits, count) in compute_topk_hit_per_graph(
                batch, node_logits.detach(), substrate_labels).items():
            topk[key][0] += hits
            topk[key][1] += count

    metrics = build_metrics(
        torch.cat(node_logits_all),
        torch.cat(node_labels_all),
        torch.cat(node_substrate_all),
        torch.cat(substrate_logits_all),
        torch.cat(substrate_labels_all),
    )
    metrics.update({
        key: value.item() / len(loader)
        for key, value in losses.items()
    })
    metrics.update({
        f"som_sub_{key}": _safe_div(hits, count)
        for key, (hits, count) in topk.items()
    })
    return metrics


def load_pretrained(model: nn.Module, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    state = {key: value for key, value in state.items()
             if key.startswith(TRANSFER_PREFIXES)}
    model.load_state_dict(state, strict=False)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    muon_params, adamw_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("gnn.", "prot_enc.", "film_mlps.")) and parameter.ndim >= 2:
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)

    return SingleDeviceMuonWithAuxAdam([
        dict(params=adamw_params, lr=lr, betas=(0.9, 0.95), eps=1e-10,
             weight_decay=weight_decay, use_muon=False),
        dict(params=muon_params, lr=lr, momentum=0.95,
             weight_decay=weight_decay, use_muon=True),
    ])


def make_jobs(args):
    pairs = (
        [
            (pretrain_split, finetune_split)
            for pretrain_split in PRETRAIN_SPLITS
            for finetune_split in FINETUNE_SPLITS
        ]
        if args.run_all
        else [(args.pretrain_split, args.split_mode)]
    )
    jobs = []
    for pretrain_split, finetune_split in pairs:
        combination = f"{pretrain_split}_to_{finetune_split}"
        for seed in SEEDS:
            run_name = f"exp_{seed + 1}"
            jobs.append({
                "pretrain_split": pretrain_split, "finetune_split": finetune_split,
                "combination": combination, "seed": seed, "run_name": run_name,
                "pretrain_ckpt": os.path.join(args.pretrain_root, pretrain_split,
                                                run_name, "best.pt"),
            })
    return jobs


def train_one(args, job):
    seed_all(job["seed"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.save_root, job["combination"], job["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    loaders = build_loaders(args.mol_pkl, args.protein_pkl, job["finetune_split"],
                            args.max_prot_len, args.batch_size, job["seed"])
    train_loader, val_loader, test_loader, node_dim, prot_dim, edge_dim, _ = loaders

    model = SOMModel(node_dim, prot_dim, args.gnn_hidden,
                     [int(value) for value in args.head_hidden.split(",")],
                     args.dropout, args.prot_kernel, edge_dim).to(device)
    del model.ec_head
    load_pretrained(model, job["pretrain_ckpt"], device)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    ema = EMA(model, decay=0.995)
    early = EarlyStopping(args.early_patience, min_delta=1e-4)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    best_epoch = 0
    metadata = {
        **job,
        **{
            key: getattr(args, key)
            for key in (
                "mol_pkl", "protein_pkl", "max_prot_len", "gnn_hidden",
                "head_hidden", "dropout", "prot_kernel", "epochs",
                "batch_size", "lr", "weight_decay", "neg_pos_ratio",
                "early_patience", "early_metric",
            )
        },
    }

    print("==== Run Info ====")
    print(f"combination: {job['combination']}")
    print(f"seed: {job['seed']}")
    print(f"pretrain: {job['pretrain_ckpt']}")
    print(f"save: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, device, args.neg_pos_ratio, optimizer, ema
        )
        ema.apply(model)
        val_metrics = run_epoch(model, val_loader, device, args.neg_pos_ratio)
        test_metrics = run_epoch(model, test_loader, device, args.neg_pos_ratio)

        score = float(val_metrics[args.early_metric])
        improved = score > early.best + early.min_delta
        stop = early.step(score, model)
        if improved:
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "metadata": metadata}, best_path)

        ema.restore(model)
        ema_state = {key: value.detach().cpu().clone() for key, value in ema.shadow.items()}
        torch.save({"epoch": epoch, "model": model.state_dict(), "ema": ema_state,
                    "optimizer": optimizer.state_dict(), "metadata": metadata}, last_path)

        print(
            f"\n[Epoch {epoch:02d}] time={time.time() - started:.1f}s "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for name, metrics in (("train", train_metrics), ("val", val_metrics),
                              ("test", test_metrics)):
            text = format_metrics(metrics).replace("\n", "\n    ")
            print(f"  {name}:\n    {text}")
        print(
            f"  early({args.early_metric})={score:.5f} "
            f"best={early.best:.5f} "
            f"bad={early.bad_epochs}/{early.patience}"
        )
        if stop:
            break

    early.restore(model)
    val_metrics = run_epoch(model, val_loader, device, args.neg_pos_ratio)
    test_metrics = run_epoch(model, test_loader, device, args.neg_pos_ratio)
    print("\n==== Final (Best Restored) ====")
    print("val:\n    " + format_metrics(val_metrics).replace("\n", "\n    "))
    print("test:\n    " + format_metrics(test_metrics).replace("\n", "\n    "))
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved last checkpoint: {last_path}")

    return {**job, "best_epoch": best_epoch, "val": val_metrics, "test": test_metrics}


def run_job(args, job):
    log_path = os.path.join(args.save_root, job["combination"], job["run_name"], "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            return train_one(args, job)


def write_summary(args, results):
    metric_names = list(results[0]["val"])
    rows = []

    for result in sorted(results, key=lambda x: (x["combination"], x["seed"])):
        for section in ("val", "test"):
            rows.append({
                "combination": result["combination"], "pretrain_split": result["pretrain_split"],
                "finetune_split": result["finetune_split"], "section": section,
                "statistic": "seed", "seed": result["seed"],
                "best_epoch": result["best_epoch"], "pretrain_ckpt": result["pretrain_ckpt"],
                **result[section]
            })

    combinations = sorted({result["combination"] for result in results})
    for combination in combinations:
        group = [r for r in results if r["combination"] == combination]
        for section in ("val", "test"):
            for statistic, function in (("mean", statistics.mean), ("std", statistics.stdev)):
                row = {
                    "combination": combination, "pretrain_split": group[0]["pretrain_split"],
                    "finetune_split": group[0]["finetune_split"], "section": section,
                    "statistic": statistic, "seed": "", "best_epoch": "", "pretrain_ckpt": "",
                }
                row.update({metric: function([r[section][metric] for r in group])
                            for metric in metric_names})
                rows.append(row)

    folder = args.save_root if args.run_all else os.path.join(
        args.save_root, results[0]["combination"])
    path = os.path.join(folder, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fields = ["combination", "pretrain_split", "finetune_split", "section",
                  "statistic", "seed", "best_epoch", "pretrain_ckpt", *metric_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune UGT models")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--mol_pkl", default="features/molecule/UGT_features.pkl")
    parser.add_argument("--protein_pkl", default="features/protein/protein_metab.pkl")
    parser.add_argument("--finetune_split", dest="split_mode", choices=FINETUNE_SPLITS,
                        default="split_smiles_group")
    parser.add_argument("--pretrain_split", choices=PRETRAIN_SPLITS,
                        default="split_enzyme_cluster")
    parser.add_argument("--max_prot_len", type=int, default=1024)
    parser.add_argument("--gnn_hidden", type=int, default=192)
    parser.add_argument("--head_hidden", default="64,32")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--prot_kernel", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--neg_pos_ratio", type=float, default=2.0)
    parser.add_argument("--early_patience", type=int, default=3)
    parser.add_argument("--early_metric", default="som_all_auprc",
                        choices=("substrate_auprc", "substrate_mcc", "som_sub_auprc",
                                 "som_sub_mcc", "som_all_auprc", "som_all_mcc"))
    parser.add_argument("--pretrain_root", default="runs_pretrain")
    parser.add_argument("--save_root", default="runs_finetune/UGT")
    args = parser.parse_args()

    jobs = make_jobs(args)
    workers = min(args.workers, len(jobs))
    print(f"jobs={len(jobs)} workers={workers} gpu={args.gpu}")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(run_job, [args] * len(jobs), jobs, chunksize=1))

    print(f"Summary CSV: {write_summary(args, results)}")


if __name__ == "__main__":
    main()
