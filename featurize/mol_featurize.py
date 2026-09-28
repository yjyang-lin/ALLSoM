# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
import pickle
import subprocess
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors
from torch_geometric.data import Data
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

JAVA_CLASS_DIR = "CDK"
JAVA_DEP_JAR = "CDK/atom_descriptor.jar"
JAVA_MAIN_CLASS = "CDKDescriptors"
JAVA_XMX = "3g"
JAVA_BATCH_SIZE = 50
JAVA_FEATURE_DIM = 5

RING_CATEGORIES = [(3, 4), (5, 7), (8, 11), (12, 999)]
HBA_SMARTS = "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
HBD_SMARTS = "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]"
HBA_PATTERN = Chem.MolFromSmarts(HBA_SMARTS)
HBD_PATTERN = Chem.MolFromSmarts(HBD_SMARTS)

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default

def one_of_k_encoding_unk(value: Any, allowable_set: List[Any]) -> List[bool]:
    value = value if value in allowable_set else allowable_set[-1]
    return [value == item for item in allowable_set]

def add_failed(failed, sdf_idx, smiles="", mapped_smiles="", enzyme_id="", reason="", detail=""):
    failed.append({
        "sdf_idx": sdf_idx,
        "smiles": smiles,
        "mapped_smiles": mapped_smiles,
        "enzyme_id": enzyme_id,
        "reason": reason,
        "detail": detail,
    })

def iter_chunks(items: List[Tuple[str, str]], chunk_size: int) -> Iterable[Dict[str, str]]:
    for start in range(0, len(items), chunk_size):
        yield dict(items[start:start + chunk_size])

def prepare_molecule_and_label(sdf_mol: Chem.Mol, som_labels: List[Any]):
    source = Chem.Mol(sdf_mol)
    n = source.GetNumAtoms()
    if len(som_labels) != n:
        raise ValueError(f"Number of SoM labels ({len(som_labels)}) does not match the number of atoms ({n})")

    label_map = {0: 0, 1: 1, "candidate": -2, "mask": -1}
    try:
        labels = np.asarray([label_map[x] for x in som_labels], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError("SoM labels may only be 0, 1, candidate, or mask") from exc

    for atom in source.GetAtoms():
        atom.SetAtomMapNum(0)

    clean_smiles = Chem.MolToSmiles(source, canonical=False, isomericSmiles=True)
    if not source.HasProp("_smilesAtomOutputOrder"):
        raise ValueError("Missing _smilesAtomOutputOrder")

    order = [int(i) for i in literal_eval(source.GetProp("_smilesAtomOutputOrder"))]
    if len(order) != n or sorted(order) != list(range(n)):
        raise ValueError(f"Invalid atom order: {order}")

    graph_mol = Chem.MolFromSmiles(clean_smiles)
    if graph_mol is None or graph_mol.GetNumAtoms() != n:
        raise ValueError("Atom count changed after SMILES conversion")
    if [source.GetAtomWithIdx(i).GetSymbol() for i in order] != [
        atom.GetSymbol() for atom in graph_mol.GetAtoms()
    ]:
        raise ValueError("Atom order mismatch after SMILES conversion")

    descriptor_mol = Chem.Mol(graph_mol)
    for atom in descriptor_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    return (
        graph_mol,
        labels[order],
        clean_smiles,
        Chem.MolToSmiles(descriptor_mol, canonical=False, isomericSmiles=True),
    )

def compute_hba_hbd_flags(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    num_atoms = mol.GetNumAtoms()
    hba_flags = np.zeros(num_atoms, dtype=np.float32)
    hbd_flags = np.zeros(num_atoms, dtype=np.float32)

    for match in mol.GetSubstructMatches(HBA_PATTERN, uniquify=True):
        for atom_idx in match:
            hba_flags[atom_idx] = 1.0

    for match in mol.GetSubstructMatches(HBD_PATTERN, uniquify=True):
        for atom_idx in match:
            hbd_flags[atom_idx] = 1.0

    return hba_flags, hbd_flags

def compute_ring_flags(mol: Chem.Mol) -> np.ndarray:
    atom_rings = mol.GetRingInfo().AtomRings()
    flags = []
    for atom in mol.GetAtoms():
        ring_sizes = [len(ring) for ring in atom_rings if atom.GetIdx() in ring]
        flags.append(
            [int(any(low <= size <= high for size in ring_sizes)) for low, high in RING_CATEGORIES]
        )
    return np.asarray(flags, dtype=np.float32)

def compute_rdkit_numeric_features(mol: Chem.Mol) -> np.ndarray:
    num_atoms = mol.GetNumAtoms()

    try:
        logp_contrib = [safe_float(item[0]) for item in Crippen._GetAtomContribs(mol)]
    except Exception:
        logp_contrib = [0.0] * num_atoms

    try:
        asa_contrib = [safe_float(v) / 4.0 for v in rdMolDescriptors._CalcLabuteASAContribs(mol)[0]]
    except Exception:
        asa_contrib = [0.0] * num_atoms

    try:
        tpsa_contrib = [safe_float(v) / 10.0 for v in rdMolDescriptors._CalcTPSAContribs(mol)]
    except Exception:
        tpsa_contrib = [0.0] * num_atoms

    hba_flags, hbd_flags = compute_hba_hbd_flags(mol)

    rows = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        rows.append(
            [
                logp_contrib[idx],
                asa_contrib[idx],
                tpsa_contrib[idx],
                float(hba_flags[idx]),
                float(hbd_flags[idx]),
            ]
        )

    return np.asarray(rows, dtype=np.float32)

def compute_atom_categorical_features(mol: Chem.Mol) -> np.ndarray:
    ring_flags = compute_ring_flags(mol)
    rows = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        row = (
            one_of_k_encoding_unk(
                atom.GetSymbol(),
                ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "X"],
            )
            + one_of_k_encoding_unk(atom.GetTotalDegree(), [1, 2, 3, 4])
            + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3])
            + one_of_k_encoding_unk(atom.GetTotalValence(), [1, 2, 3, 4, 5, 6])
            + one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
            + one_of_k_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                ],
            )
            + one_of_k_encoding_unk(
                atom.GetChiralTag(),
                [
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                ],
            )
            + [int(atom.GetIsAromatic())]
            + ring_flags[idx].astype(int).tolist()
        )
        rows.append(row)

    return np.asarray(rows, dtype=np.float32)

# Java/CDK atom features

def parse_java_map_line(value: str) -> Dict[int, int]:
    try:
        return {int(k): int(v) for k, v in literal_eval(value.replace("=", ":")).items()}
    except Exception as exc:
        raise ValueError(f"cannot parse mapped_index line: {value}") from exc

def align_java_descriptor(
    mapped_index_line: str,
    descriptor_line: str,
    num_graph_atoms: int,
) -> np.ndarray:
    """
    The descriptor array returned by Java includes explicit H atoms, while mappedIndex
    only contains graph atoms with positive atom-map numbers. Align the Java atom
    indices with graph atom order using maps 1..N.
    """
    mapped_index = parse_java_map_line(mapped_index_line)

    try:
        values = literal_eval(descriptor_line)
    except Exception as exc:
        raise ValueError(f"cannot parse descriptor line: {descriptor_line}") from exc

    expected_maps = set(range(1, num_graph_atoms + 1))
    actual_maps = set(mapped_index.keys())

    missing_maps = expected_maps - actual_maps
    if missing_maps:
        raise ValueError(
            f"mapped_index missing graph atom maps: {sorted(missing_maps)}; "
            f"mapped_index={mapped_index}"
        )

    unexpected_maps = actual_maps - expected_maps
    if unexpected_maps:
        raise ValueError(
            f"mapped_index contains unexpected maps: {sorted(unexpected_maps)}; "
            f"mapped_index={mapped_index}"
        )

    aligned = np.zeros(num_graph_atoms, dtype=np.float32)

    for graph_atom_idx in range(num_graph_atoms):
        map_num = graph_atom_idx + 1
        jar_atom_idx = mapped_index[map_num]
        if jar_atom_idx < 0 or jar_atom_idx >= len(values):
            raise ValueError(
                f"mapped_index value out of descriptor range: "
                f"map_num={map_num}, jar_atom_idx={jar_atom_idx}, len(values)={len(values)}"
            )
        aligned[graph_atom_idx] = safe_float(values[jar_atom_idx])

    return aligned

def compute_java_atom_features_from_fields(
    mapped_index_line: str,
    descriptor_fields: List[str],
    num_graph_atoms: int,
) -> np.ndarray:
    if len(descriptor_fields) != JAVA_FEATURE_DIM:
        raise ValueError(
            f"expected {JAVA_FEATURE_DIM} Java descriptor fields, got {len(descriptor_fields)}"
        )

    features = np.stack(
        [
            align_java_descriptor(mapped_index_line, descriptor_fields[0], num_graph_atoms) / 5.0,
            align_java_descriptor(mapped_index_line, descriptor_fields[1], num_graph_atoms),
            align_java_descriptor(mapped_index_line, descriptor_fields[2], num_graph_atoms),
            align_java_descriptor(mapped_index_line, descriptor_fields[3], num_graph_atoms) / 3.0,
            align_java_descriptor(mapped_index_line, descriptor_fields[4], num_graph_atoms) / 5.0,
        ],
        axis=1,
    ).astype(np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

def build_java_feature_cache_batch(
    descriptor_items: Dict[str, str],
    java_xmx: str = JAVA_XMX,
    batch_size: int = JAVA_BATCH_SIZE,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    cache, errors = {}, {}
    descriptor_smiles_list = list(descriptor_items.keys())

    if not descriptor_smiles_list:
        return cache, errors

    input_text = "".join(
        f"{idx}\t{smi}\n"
        for idx, smi in enumerate(descriptor_smiles_list)
    )

    result = subprocess.run(
        [
            "java",
            f"-Xmx{java_xmx}",
            "-cp",
            f"{JAVA_CLASS_DIR}:{JAVA_DEP_JAR}",
            JAVA_MAIN_CLASS,
            "--batch",
            str(batch_size),
        ],
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Java/CDK batch failed: returncode={result.returncode}; "
            f"stderr={result.stderr}; stdout_head={result.stdout[:1000]}"
        )

    seen = set()

    for line in result.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        try:
            idx = int(parts[0])
            descriptor_smiles = descriptor_smiles_list[idx]
        except Exception:
            continue

        seen.add(idx)

        if parts[1] != "OK":
            errors[descriptor_smiles] = parts[2] if len(parts) > 2 else "unknown Java error"
            continue

        if len(parts) != 8:
            errors[descriptor_smiles] = f"bad Java output field count: {len(parts)}"
            continue

        mol = Chem.MolFromSmiles(descriptor_smiles)
        if mol is None:
            errors[descriptor_smiles] = f"bad descriptor_smiles: {descriptor_smiles}"
            continue

        try:
            cache[descriptor_smiles] = compute_java_atom_features_from_fields(
                mapped_index_line=parts[2],
                descriptor_fields=parts[3:8],
                num_graph_atoms=mol.GetNumAtoms(),
            )
        except Exception as exc:
            errors[descriptor_smiles] = str(exc)

    for idx, descriptor_smiles in enumerate(descriptor_smiles_list):
        if idx not in seen:
            errors[descriptor_smiles] = "missing from Java stdout batch output"

    return cache, errors

def build_java_feature_cache_batch_parallel(
    descriptor_items: Dict[str, str],
    num_workers: int,
    java_xmx: str = JAVA_XMX,
    batch_size: int = JAVA_BATCH_SIZE,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    cache: Dict[str, np.ndarray] = {}
    errors: Dict[str, str] = {}

    items = list(descriptor_items.items())
    if not items:
        return cache, errors

    num_workers = max(1, min(int(num_workers), len(items)))
    num_chunks = min(len(items), max(num_workers * 8, num_workers))
    chunk_size = max(1, (len(items) + num_chunks - 1) // num_chunks)
    chunks = list(iter_chunks(items, chunk_size))

    print(
        f"Computing Java/CDK features with parallel batches: "
        f"num_workers={num_workers}, chunks={len(chunks)}, "
        f"unique molecules={len(items)}, chunk_size≈{chunk_size}"
    )

    with tqdm(
        total=len(items),
        desc=f"Java/CDK features ({num_workers} workers)",
        unit="mol",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    build_java_feature_cache_batch,
                    chunk,
                    java_xmx,
                    batch_size,
                ): chunk
                for chunk in chunks
            }

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                chunk_size_done = len(chunk)

                try:
                    sub_cache, sub_errors = future.result()
                except Exception as exc:
                    for descriptor_smiles in chunk.keys():
                        errors[descriptor_smiles] = repr(exc)

                    pbar.update(chunk_size_done)
                    pbar.set_postfix(
                        {
                            "ok": len(cache),
                            "err": len(errors),
                            "failed_chunk": chunk_size_done,
                        }
                    )
                    continue

                cache.update(sub_cache)
                errors.update(sub_errors)

                pbar.update(chunk_size_done)
                pbar.set_postfix(
                    {
                        "ok": len(cache),
                        "err": len(errors),
                    }
                )

    return cache, errors

# Node feature assembly

def compute_node_features(clean_smiles: str, java_features: np.ndarray) -> np.ndarray:
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        raise ValueError(f"bad clean_smiles: {clean_smiles}")

    java_features = np.asarray(java_features, dtype=np.float32)
    if java_features.shape != (mol.GetNumAtoms(), JAVA_FEATURE_DIM):
        raise ValueError(
            f"java_features shape {java_features.shape} incompatible with "
            f"num_atoms={mol.GetNumAtoms()}, dim={JAVA_FEATURE_DIM}"
        )

    rdkit_numeric = compute_rdkit_numeric_features(mol)
    categorical = compute_atom_categorical_features(mol)

    features = np.concatenate([rdkit_numeric, java_features, categorical], axis=1).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

def build_node_feature_cache(
    descriptor_items: Dict[str, str],
    num_workers: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    cache: Dict[str, np.ndarray] = {}
    errors: Dict[str, str] = {}

    java_workers = max(1, int(num_workers))
    print(
        f"Computing Java/CDK features with parallel batches: "
        f"num_workers={java_workers}, unique molecules={len(descriptor_items)}"
    )

    java_feature_cache, java_feature_errors = build_java_feature_cache_batch_parallel(
        descriptor_items=descriptor_items,
        num_workers=java_workers,
    )

    for descriptor_smiles, clean_smiles in tqdm(
        descriptor_items.items(),
        desc="Assemble node features",
        unit="mol",
    ):
        try:
            mol = Chem.MolFromSmiles(clean_smiles)
            if mol is None:
                raise ValueError(f"bad clean_smiles: {clean_smiles}")

            if descriptor_smiles in java_feature_cache:
                java_features = java_feature_cache[descriptor_smiles]
            else:
                err = java_feature_errors.get(descriptor_smiles, "unknown Java/CDK error")
                print(
                    f"[WARN] Java/CDK features failed, use zeros. "
                    f"smiles={clean_smiles}; error={err}"
                )
                java_features = np.zeros((mol.GetNumAtoms(), JAVA_FEATURE_DIM), dtype=np.float32)

            cache[descriptor_smiles] = compute_node_features(
                clean_smiles=clean_smiles,
                java_features=java_features,
            )
        except Exception as exc:
            errors[descriptor_smiles] = str(exc)

    return cache, errors

# Edge features and graph construction

def bond_ring_category_features(bond: Chem.Bond) -> List[int]:
    bond_idx = bond.GetIdx()
    bond_rings = bond.GetOwningMol().GetRingInfo().BondRings()
    ring_sizes = [len(ring) for ring in bond_rings if bond_idx in ring]
    return [int(any(low <= size <= high for size in ring_sizes)) for low, high in RING_CATEGORIES]

def bond_features(bond: Chem.Bond) -> np.ndarray:
    features = (
        one_of_k_encoding_unk(
            bond.GetBondType(),
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ],
        )
        + one_of_k_encoding_unk(
            bond.GetStereo(),
            [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
            ],
        )
        + [int(bond.GetIsConjugated())]
        + bond_ring_category_features(bond)
    )
    return np.asarray(features, dtype=np.float32)

def build_graph(
    mol: Chem.Mol,
    node_features: np.ndarray,
    node_label: np.ndarray,
) -> Tuple[Data, torch.Tensor]:
    num_atoms = mol.GetNumAtoms()
    node_features = np.asarray(node_features, dtype=np.float32)
    node_label = np.asarray(node_label, dtype=np.int64)

    if node_features.ndim != 2 or node_features.shape[0] != num_atoms:
        raise ValueError(f"node_features shape {node_features.shape} incompatible with {num_atoms} atoms")
    if node_label.shape != (num_atoms,):
        raise ValueError(f"node_label shape {node_label.shape} incompatible with {num_atoms} atoms")

    edge_pairs: List[Tuple[int, int]] = []
    edge_rows: List[np.ndarray] = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        row = bond_features(bond)
        edge_pairs.extend([(begin_idx, end_idx), (end_idx, begin_idx)])
        edge_rows.extend([row, row])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.asarray(edge_rows, dtype=np.float32), dtype=torch.float)
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), dtype=torch.float),
            size=(num_atoms, num_atoms),
        ).coalesce()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 12), dtype=torch.float)
        adj = torch.sparse_coo_tensor(
            indices=torch.empty((2, 0), dtype=torch.long),
            values=torch.empty((0,), dtype=torch.float),
            size=(num_atoms, num_atoms),
        ).coalesce()

    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_node=torch.tensor(node_label, dtype=torch.long),
        num_nodes=num_atoms,
    )
    return graph, adj

def validate_alignment(mol: Chem.Mol, node_features: np.ndarray, node_label: np.ndarray) -> None:
    num_atoms = mol.GetNumAtoms()
    if node_features.shape[0] != num_atoms:
        raise ValueError(f"node_features rows {node_features.shape[0]} != num_atoms {num_atoms}")
    if node_label.shape[0] != num_atoms:
        raise ValueError(f"node_label length {node_label.shape[0]} != num_atoms {num_atoms}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "fine_tune", "inference"], default="train")
    parser.add_argument("--in_sdf", default="pretrain_data/pretrain_data_split.sdf")
    parser.add_argument("--out_pkl", default="features/molecule/pretrain_features.pkl")
    parser.add_argument("--out_failed_csv", default="pretrain_data/failed_records.csv")
    parser.add_argument("--smiles_field", default="SMILES")
    parser.add_argument("--mapped_smiles_field", default="MappedSMILES")
    parser.add_argument("--som_field", default="SOM_AtomIdx")
    parser.add_argument("--protein_field", default="Protein_Substrate_Records")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_pkl) or ".", exist_ok=True)
    sdf_mols = list(Chem.SDMolSupplier(args.in_sdf, removeHs=True))
    parsed, failed, descriptor_items = [], [], {}

    for sdf_idx, mol in enumerate(tqdm(sdf_mols, desc="Parse SDF", unit="mol")):
        if mol is None:
            add_failed(failed, sdf_idx, reason="bad_sdf_mol")
            continue

        smiles = mol.GetProp(args.smiles_field).strip() if mol.HasProp(args.smiles_field) else ""
        mapped_smiles = (
            mol.GetProp(args.mapped_smiles_field).strip()
            if mol.HasProp(args.mapped_smiles_field) else ""
        )

        try:
            if not smiles:
                raise ValueError(f"Missing {args.smiles_field}")

            som_labels = (
                json.loads(mol.GetProp(args.som_field))
                if mol.HasProp(args.som_field)
                else ["mask"] * mol.GetNumAtoms()
            )
            if not isinstance(som_labels, list):
                raise ValueError(f"{args.som_field} must be a JSON list")

            raw = json.loads(mol.GetProp(args.protein_field))
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"{args.protein_field} must be a non-empty JSON list")

            splits = {}
            if args.mode in ("train", "fine_tune"):
                for field, key in (
                    ("SmilesGroupSplit", "split_smiles_group"),
                    ("ScaffoldSplit", "split_scaffold"),
                ):
                    if not mol.HasProp(field):
                        raise ValueError(f"Missing {field}")
                    splits[key] = mol.GetProp(field).strip()

            graph_mol, node_label, clean_smiles, descriptor_smiles = prepare_molecule_and_label(
                mol, som_labels
            )
        except Exception as exc:
            add_failed(
                failed, sdf_idx, smiles, mapped_smiles,
                reason="molecule_parse_failed", detail=str(exc),
            )
            continue

        relations, seen = [], {}
        for item in raw:
            try:
                enzyme = str(item["uniprot_id"]).strip()
                if not enzyme:
                    raise ValueError("Empty uniprot_id")

                label = item.get("substrate_label")
                if args.mode != "inference":
                    label = int(label)
                    if label not in (0, 1):
                        raise ValueError(f"Invalid substrate_label: {label}")
                elif label is not None:
                    label = int(label)

                relation = {"uniprot_id": enzyme, "substrate_label": label}
                if "enzyme_cluster" in item:
                    relation["enzyme_cluster"] = str(item["enzyme_cluster"])
                if args.mode == "train":
                    if "EnzymeClusterSplit" not in item:
                        raise ValueError("Missing EnzymeClusterSplit")
                    relation["split_enzyme_cluster"] = str(item["EnzymeClusterSplit"])

                signature = json.dumps(relation, sort_keys=True, ensure_ascii=False)
                if enzyme in seen and seen[enzyme] != signature:
                    raise ValueError("Conflicting records for the same protein")
                if enzyme not in seen:
                    seen[enzyme] = signature
                    relations.append(relation)
            except Exception as exc:
                add_failed(
                    failed, sdf_idx, smiles, mapped_smiles,
                    str(item.get("uniprot_id", "")),
                    "protein_record_failed", str(exc),
                )

        if not relations:
            continue

        parsed.append({
            "sdf_idx": sdf_idx,
            "smiles": smiles,
            "mapped_smiles": mapped_smiles,
            "clean_smiles": clean_smiles,
            "descriptor_smiles": descriptor_smiles,
            "graph_mol": graph_mol,
            "node_label": node_label,
            "relations": relations,
            **splits,
        })
        descriptor_items[descriptor_smiles] = clean_smiles

    node_cache, feature_errors = build_node_feature_cache(
        descriptor_items, num_workers=args.num_workers
    )

    records = []
    for item in tqdm(parsed, desc="Build graphs", unit="mol"):
        descriptor_smiles = item["descriptor_smiles"]
        if descriptor_smiles not in node_cache:
            add_failed(
                failed, item["sdf_idx"], item["smiles"], item["mapped_smiles"],
                reason="node_feature_failed",
                detail=feature_errors.get(descriptor_smiles, "missing node features"),
            )
            continue

        try:
            features = node_cache[descriptor_smiles]
            validate_alignment(item["graph_mol"], features, item["node_label"])
            graph, adj = build_graph(item["graph_mol"], features, item["node_label"])
            if torch.isnan(graph.x).any() or (
                graph.edge_attr.numel() and torch.isnan(graph.edge_attr).any()
            ):
                raise ValueError("graph contains NaN")
        except Exception as exc:
            add_failed(
                failed, item["sdf_idx"], item["smiles"], item["mapped_smiles"],
                reason="graph_build_failed", detail=str(exc),
            )
            continue

        base = {
            "sdf_idx": item["sdf_idx"],
            "smiles": item["smiles"],
            "mapped_smiles": item["mapped_smiles"],
            "clean_smiles": item["clean_smiles"],
            "graph": graph,
            "adj": adj,
            "node_label": item["node_label"],
        }
        if args.mode in ("train", "fine_tune"):
            base["split_smiles_group"] = item["split_smiles_group"]
            base["split_scaffold"] = item["split_scaffold"]

        base["protein_records"] = item["relations"]
        records.append(base)

    payload = {
        "meta": {
            "mode": args.mode,
            "in_sdf": args.in_sdf,
            "node_feature_dim": int(records[0]["graph"].x.shape[1]) if records else None,
            "edge_feature_dim": int(records[0]["graph"].edge_attr.shape[1]) if records else None,
            "n_records": len(records),
            "n_protein_records": sum(len(r["protein_records"]) for r in records),
            "n_failed": len(failed),
        },
        "data": records,
    }

    with open(args.out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    if failed:
        os.makedirs(os.path.dirname(args.out_failed_csv) or ".", exist_ok=True)
        with open(args.out_failed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sdf_idx", "smiles", "mapped_smiles",
                    "enzyme_id", "reason", "detail",
                ],
            )
            writer.writeheader()
            writer.writerows(failed)
    elif os.path.exists(args.out_failed_csv):
        os.remove(args.out_failed_csv)

    print(
        f"molecules={len(parsed)} records={len(records)} "
        f"failed={len(failed)} saved={args.out_pkl}"
    )
    if failed:
        print(f"failed_csv={args.out_failed_csv}")

if __name__ == "__main__":
    main()
