import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset

sys.path[:0] = [".", "model"]

from featurize.mol_featurize import build_graph, build_node_feature_cache, prepare_molecule_and_label
from train import SOMModel
from utils import load_protein_dict, som_collate


sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.numeric"] = np.core.numeric
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

MODEL_DIRS = {
    "CYP": Path("runs_finetune/CYP/split_enzyme_cluster_to_split_smiles_group"),
    "UGT": Path("runs_finetune/UGT/split_enzyme_cluster_to_split_smiles_group"),
    "AOX": Path("runs_finetune/AOX_transfer/split_enzyme_cluster/exp_1/smiles"),
}
RULE_COLUMNS = (
    "Reaction name", "SMIRKS", "Priority level", "Name of rule subset", "Score atom map",
)
PRIORITY_WEIGHTS = {"common": 1.0, "uncommon": 0.2}
MIN_PRODUCT_CARBON_ATOMS = 4
UGT_SMARTS = (
    "[$([OX2H][A;!#1;!$([C,N,P,S]=O)])]",
    "[$([OX2H]a)]",
    "[$([OX2H1][CX3]=O)]",
    "[$([#7;!R0]),$([NX3,NX4]);!$([#7][C,S]=[O,S,N]);!$(N=O)]",
)
UGT_PATTERNS = tuple(Chem.MolFromSmarts(smarts) for smarts in UGT_SMARTS)
GEMINAL_DIOL_TO_CARBONYL = AllChem.ReactionFromSmarts(
    "[C;X4:1]([O;H1;D1:2])([O;H1;D1:3])>>[C:1](=[O:2])"
)


class MoleculeDataset(Dataset):
    def __init__(self, samples, protein_dict, max_protein_length):
        self.samples = samples
        self.protein_dict = protein_dict
        self.max_protein_length = max_protein_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        graph, uid, _molecule, _row_index = self.samples[index]
        graph = graph.clone()
        graph.y = graph.y_node.clone().long()
        graph.candidate_mask = torch.ones_like(graph.y, dtype=torch.bool)
        protein = torch.as_tensor(
            self.protein_dict[uid]["features"][: self.max_protein_length],
            dtype=torch.float32,
        )
        return graph, protein, uid, 1


def enzyme_class(uid):
    uid = str(uid).strip().upper()
    for family in ("CYP", "UGT", "AOX"):
        if uid.startswith(family):
            return family
    raise ValueError(f"Unsupported enzyme: {uid}")


def load_rules(path):
    rules = {family: [] for family in MODEL_DIRS}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RULE_COLUMNS:
            raise ValueError(f"Reaction-rule columns must be: {list(RULE_COLUMNS)}")

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed reaction-rule row {line_number}")

            name = row["Reaction name"].strip()
            smirks = row["SMIRKS"].strip()
            priority = row["Priority level"].strip()
            subset = row["Name of rule subset"].strip()
            score_atom_map_text = row["Score atom map"].strip()
            if subset not in rules or priority not in PRIORITY_WEIGHTS:
                raise ValueError(f"Invalid reaction-rule metadata at row {line_number}")

            score_atom_map = int(score_atom_map_text) if score_atom_map_text else None
            reaction = AllChem.ReactionFromSmarts(smirks)
            if reaction is None:
                raise ValueError(f"Invalid SMIRKS at row {line_number}: {name}")
            reaction.Initialize()
            rules[subset].append(
                {
                    "name": name,
                    "smirks": smirks,
                    "priority": priority,
                    "weight": PRIORITY_WEIGHTS[priority],
                    "reaction": reaction,
                    "query": reaction.GetReactantTemplate(0),
                    "score_atom_map": score_atom_map if subset == "CYP" else None,
                    "site_query_indices": reaction_site_query_indices(
                        reaction.GetReactantTemplate(0)
                    ) if subset == "UGT" else (),
                }
            )
    return rules


def reaction_site_query_indices(query):
    """Return the O/N atom directly conjugated by each UGT reaction rule."""
    hetero_atoms = [
        atom.GetIdx()
        for atom in query.GetAtoms()
        if atom.GetAtomicNum() in {7, 8}
    ]
    hydrogen_bearing = [
        index
        for index in hetero_atoms
        if any(
            neighbor.GetAtomicNum() == 1
            for neighbor in query.GetAtomWithIdx(index).GetNeighbors()
        )
    ]
    return tuple(hydrogen_bearing or hetero_atoms)


def checkpoint_paths(model_dir):
    paths = sorted(model_dir.glob("exp_*/best.pt"))
    if not paths:
        paths = sorted(model_dir.glob("fold_*/best.pt"))
    if not paths:
        raise FileNotFoundError(f"No best.pt found under {model_dir}")
    return paths


def load_models(model_dir, node_dim, protein_dim, edge_dim, device):
    checkpoints = [torch.load(path, map_location="cpu") for path in checkpoint_paths(model_dir)]
    models = []
    for checkpoint in checkpoints:
        config = checkpoint["metadata"]
        hidden = [int(value) for value in config["head_hidden"].split(",")]
        model = SOMModel(
            node_dim, protein_dim, config["gnn_hidden"], hidden,
            config["dropout"], config["prot_kernel"], edge_dim,
        ).to(device)
        del model.ec_head
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)
    return models, int(checkpoints[0]["metadata"]["max_prot_len"])


def build_dataset(frame, protein_dict, max_protein_length, workers):
    parsed = []
    descriptor_items = {}
    for row_index, row in frame.iterrows():
        uid = str(row["enzyme"]).strip()
        molecule = Chem.MolFromSmiles(str(row["smiles"]).strip())
        if molecule is None:
            raise ValueError(f"Invalid SMILES at CSV row {row_index + 2}")
        graph_molecule, labels, clean_smiles, descriptor_smiles = prepare_molecule_and_label(
            molecule, [0] * molecule.GetNumAtoms()
        )
        parsed.append((row_index, uid, graph_molecule, labels, descriptor_smiles))
        descriptor_items[descriptor_smiles] = clean_smiles

    feature_cache, errors = build_node_feature_cache(descriptor_items, num_workers=workers)
    if errors:
        first = next(iter(errors.items()))
        raise RuntimeError(f"Feature generation failed for {first[0]}: {first[1]}")

    samples = []
    for row_index, uid, molecule, labels, descriptor_smiles in parsed:
        graph, _ = build_graph(molecule, feature_cache[descriptor_smiles], labels)
        samples.append((graph, uid, molecule, row_index))
    return MoleculeDataset(samples, protein_dict, max_protein_length)


def valid_product_fragments(product):
    try:
        Chem.SanitizeMol(product)
        product = Chem.RemoveHs(product)
        fragments = Chem.GetMolFrags(product, asMols=True, sanitizeFrags=True)
    except Exception:
        return []
    return [
        fragment
        for fragment in fragments
        if (
            sum(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())
            > MIN_PRODUCT_CARBON_ATOMS
        )
    ]
def normalize_geminal_diols(molecule):
    """Represent aldehyde/ketone hydrates as their conventional carbonyl form."""
    normalized = Chem.Mol(molecule)
    while True:
        outcomes = GEMINAL_DIOL_TO_CARBONYL.RunReactants((normalized,))
        if not outcomes:
            return normalized
        normalized = outcomes[0][0]
        try:
            Chem.SanitizeMol(normalized)
            normalized = Chem.RemoveHs(normalized)
        except Exception:
            return molecule


def ring_n_dealkylation_product(rule, match, molecule):
    """Open a cyclic N-C bond without splitting or duplicating the molecule."""
    if rule["name"].lower().replace("_", " ") != "n-dealkylation":
        return False, None
    mapped_atoms = {
        atom.GetAtomMapNum(): match[atom.GetIdx()]
        for atom in rule["query"].GetAtoms()
        if atom.GetAtomMapNum()
    }
    nitrogen_index = mapped_atoms.get(1)
    carbon_index = mapped_atoms.get(2)
    if nitrogen_index is None or carbon_index is None:
        return False, None
    bond = molecule.GetBondBetweenAtoms(nitrogen_index, carbon_index)
    if bond is None or not bond.IsInRing():
        return False, None

    editable = Chem.RWMol(molecule)
    editable.RemoveBond(nitrogen_index, carbon_index)
    if len(Chem.GetMolFrags(editable.GetMol())) != 1:
        return True, None
    oxygen_index = editable.AddAtom(Chem.Atom(8))
    editable.AddBond(carbon_index, oxygen_index, Chem.BondType.DOUBLE)
    product = editable.GetMol()
    try:
        product.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(product)
        Chem.AssignStereochemistry(product, cleanIt=True, force=True)
    except Exception:
        return True, None
    return True, product


def route_annotation(rule, match, molecule):
    """Identify routes that produce redundant single-step metabolites."""
    mapped_atoms = {
        atom.GetAtomMapNum(): match[atom.GetIdx()]
        for atom in rule["query"].GetAtoms()
        if atom.GetAtomMapNum()
    }
    name = rule["name"].lower().replace("_", " ")

    if name == "aliphatic hydroxylation":
        carbon_index = mapped_atoms.get(1)
        if carbon_index is None:
            return None, (), None
        carbon = molecule.GetAtomWithIdx(carbon_index)
        keys = []
        for neighbor in carbon.GetNeighbors():
            symbol = neighbor.GetSymbol()
            bond = molecule.GetBondBetweenAtoms(carbon_index, neighbor.GetIdx())
            if symbol == "N" and bond is not None and bond.IsInRing():
                continue
            if symbol in {"N", "O", "S"}:
                keys.append((symbol, carbon_index))
            elif symbol in {"F", "Cl", "Br"}:
                keys.append(("X", carbon_index))
        return "hydroxylation", tuple(sorted(keys)), carbon_index

    if name == "n-dealkylation":
        carbon_index = mapped_atoms.get(2)
        nitrogen_index = mapped_atoms.get(1)
        if carbon_index is None or nitrogen_index is None:
            return None, (), carbon_index
        bond = molecule.GetBondBetweenAtoms(nitrogen_index, carbon_index)
        if bond is not None and bond.IsInRing():
            return None, (), carbon_index
        return "dealkylation", (("N", carbon_index),), carbon_index

    if name == "o-dealkylation of methylenedioxyphenyl":
        carbon_index = mapped_atoms.get(2)
        return "dealkylation", (("O", carbon_index),), carbon_index

    if name == "oxidative ether cleavage to one alcohol and one aldehyde/ketone":
        carbon_index = mapped_atoms.get(3)
        return "dealkylation", (("O", carbon_index),), carbon_index

    if name == "s-dealkylation":
        carbon_index = mapped_atoms.get(2)
        return "dealkylation", (("S", carbon_index),), carbon_index

    if name == "oxidative dehalogenation alkyl":
        carbon_index = mapped_atoms.get(1)
        return "dealkylation", (("X", carbon_index),), carbon_index

    return None, (), None


def generate_products(molecule, probabilities, rules, allowed_site_indices=None):
    parent_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    heavy_atom_count = molecule.GetNumAtoms()
    reaction_molecule = Chem.AddHs(molecule)
    products = {}
    allowed_sites = None if allowed_site_indices is None else set(allowed_site_indices)

    for rule in rules:
        matches = reaction_molecule.GetSubstructMatches(rule["query"], uniquify=True)
        for match in matches:
            matched_heavy_atoms = sorted(
                atom_index for atom_index in set(match) if atom_index < heavy_atom_count
            )
            if not matched_heavy_atoms:
                continue
            if allowed_sites is not None:
                reaction_sites = sorted({
                    match[index]
                    for index in rule.get("site_query_indices", ())
                    if index < len(match) and match[index] < heavy_atom_count
                } & allowed_sites)
                if not reaction_sites:
                    continue
            else:
                reaction_sites = matched_heavy_atoms
            route_type, route_keys, route_carbon = route_annotation(rule, match, reaction_molecule)
            score_atom_index = next((
                match[atom.GetIdx()]
                for atom in rule["query"].GetAtoms()
                if atom.GetAtomMapNum() == rule.get("score_atom_map")
            ), None)
            if score_atom_index is not None and score_atom_index < heavy_atom_count:
                som_probability = float(probabilities[score_atom_index])
            elif route_carbon is not None:
                som_probability = float(probabilities[route_carbon])
            else:
                som_probability = max(
                    float(probabilities[atom_index])
                    for atom_index in reaction_sites
                )
            score = som_probability * rule["weight"]

            is_ring_dealkylation, ring_product = ring_n_dealkylation_product(rule, match, molecule)
            if is_ring_dealkylation:
                outcomes = ((ring_product,),) if ring_product is not None else ()
            else:
                protected = Chem.Mol(reaction_molecule)
                matched_set = set(match)
                for atom in protected.GetAtoms():
                    if atom.GetIdx() not in matched_set:
                        atom.SetBoolProp("_protected", True)
                try:
                    outcomes = rule["reaction"].RunReactants((protected,))
                except Exception:
                    continue

            for outcome in outcomes:
                for raw_product in outcome:
                    for product in valid_product_fragments(raw_product):
                        product_smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
                        product = Chem.MolFromSmiles(product_smiles)
                        if product is None:
                            continue
                        product = normalize_geminal_diols(product)
                        product_smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
                        if product_smiles == parent_smiles:
                            continue
                        record = {
                            "product_smiles": product_smiles,
                            "reaction_name": rule["name"],
                            "priority": rule["priority"],
                            "matched_atom_indices_0based": ";".join(map(str, matched_heavy_atoms)),
                            "som_probability": som_probability,
                            "product_score": score,
                            "_has_explicit_score_atom": rule.get("score_atom_map")
                            is not None,
                            "_hydroxylation_keys": set(route_keys)
                            if route_type == "hydroxylation"
                            else set(),
                            "_dealkylation_keys": set(route_keys)
                            if route_type == "dealkylation"
                            else set(),
                        }
                        previous = products.get(product_smiles)
                        if previous is None:
                            products[product_smiles] = record
                        else:
                            hydroxylation_keys = previous["_hydroxylation_keys"] | record["_hydroxylation_keys"]
                            dealkylation_keys = previous["_dealkylation_keys"] | record["_dealkylation_keys"]
                            if (
                                score > previous["product_score"]
                                or score == previous["product_score"]
                                and record["_has_explicit_score_atom"]
                                and not previous["_has_explicit_score_atom"]
                            ):
                                products[product_smiles] = record
                            products[product_smiles]["_hydroxylation_keys"] = hydroxylation_keys
                            products[product_smiles]["_dealkylation_keys"] = dealkylation_keys

    dealkylation_keys = set().union(*(record["_dealkylation_keys"] for record in products.values()))
    products = {
        smiles: record
        for smiles, record in products.items()
        if record["_dealkylation_keys"]
        or not (record["_hydroxylation_keys"] & dealkylation_keys)
    }

    return sorted(
        products.values(),
        key=lambda row: (
            -row["product_score"], not row["_has_explicit_score_atom"], row["product_smiles"]
        ),
    )


@torch.inference_mode()
def predict_and_generate(models, loader, dataset, frame, rules, device, top_n, min_score):
    rows = []
    sample_offset = 0
    for batch, protein, protein_mask, _uids, _labels in loader:
        batch = batch.to(device)
        protein = protein.to(device)
        protein_mask = protein_mask.to(device)
        probabilities = torch.stack(
            [torch.sigmoid(model(batch, protein, protein_mask)) for model in models]
        ).mean(dim=0).cpu()
        batch = batch.cpu()

        for local_index in range(batch.num_graphs):
            _graph, uid, molecule, row_index = dataset.samples[sample_offset + local_index]
            start = int(batch.ptr[local_index])
            end = int(batch.ptr[local_index + 1])
            allowed_sites = None
            if enzyme_class(uid) == "UGT":
                allowed_sites = {
                    atom_index
                    for pattern in UGT_PATTERNS
                    for match in molecule.GetSubstructMatches(pattern)
                    for atom_index in match
                }
            candidates = generate_products(
                molecule, probabilities[start:end], rules, allowed_site_indices=allowed_sites
            )
            candidates = [candidate for candidate in candidates if candidate["product_score"] >= min_score]
            if top_n > 0:
                candidates = candidates[:top_n]
            source = frame.loc[row_index]
            for rank, candidate in enumerate(candidates, start=1):
                rows.append(
                    {
                        "input_row": int(row_index) + 2,
                        "name": source.get("name", ""),
                        "enzyme": uid,
                        "parent_smiles": Chem.MolToSmiles(
                            molecule, canonical=True, isomericSmiles=True
                        ),
                        "rank": rank,
                        **candidate,
                    }
                )
        sample_offset += batch.num_graphs
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="product_generation/case_study_molecules.csv")
    parser.add_argument("--output", default="product_generation/case_study_products.csv")
    parser.add_argument("--rules", default="product_generation/gloryx_reactionrules.csv")
    parser.add_argument("--protein_pkl", default="features/protein/protein_metab.pkl")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--feature_workers", type=int, default=4)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    frame = pd.read_csv(args.csv)
    required = {"smiles", "enzyme"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input CSV must contain columns: {sorted(required)}")

    frame = frame.copy()
    frame["_canonical_smiles"] = frame["smiles"].map(
        lambda value: Chem.MolToSmiles(
            Chem.MolFromSmiles(str(value).strip()),
            canonical=True,
            isomericSmiles=True,
        )
    )
    frame = frame.drop_duplicates(subset=["enzyme", "_canonical_smiles"], keep="first")
    frame["enzyme_class"] = frame["enzyme"].map(enzyme_class)
    protein_dict = load_protein_dict(args.protein_pkl)
    reaction_rules = load_rules(args.rules)
    all_rows = []

    for family, subset in frame.groupby("enzyme_class", sort=False):
        model_dir = MODEL_DIRS[family]
        paths = checkpoint_paths(model_dir)
        first_checkpoint = torch.load(paths[0], map_location="cpu")
        max_protein_length = int(first_checkpoint["metadata"]["max_prot_len"])
        dataset = build_dataset(subset, protein_dict, max_protein_length, args.feature_workers)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
            pin_memory=device.type == "cuda", collate_fn=som_collate, drop_last=False,
        )
        graph, protein, _uid, _label = dataset[0]
        edge_dim = int(graph.edge_attr.shape[1]) if graph.edge_attr.numel() else 0
        models, _ = load_models(
            model_dir, int(graph.x.shape[1]), int(protein.shape[1]), edge_dim, device
        )
        family_rows = predict_and_generate(
            models, loader, dataset, frame, reaction_rules[family],
            device, args.top_n, args.min_score,
        )
        all_rows.extend(family_rows)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    columns = [
        "input_row", "name", "enzyme", "parent_smiles",
        "rank", "product_smiles", "reaction_name", "priority",
        "matched_atom_indices_0based", "som_probability", "product_score",
    ]
    output = pd.DataFrame(all_rows, columns=columns)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.6f")
    print(f"Generated {len(output)} ranked products from {len(frame)} inputs: {output_path}")


if __name__ == "__main__":
    main()
