import os
import pickle
import logging
import argparse
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import esm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_LEN = 1024  # Maximum sequence length for processing

def parse_ec_num(x) -> List[str]:
    """
    Parse EC numbers from csv cell.

    Examples:
        NaN / "" -> []
        "1.1.1.1" -> ["1.1.1.1"]
        "1.1.1.1;1.1.1.192" -> ["1.1.1.1", "1.1.1.192"]
    """
    if pd.isna(x):
        return []

    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "none", "null"}:
        return []

    ec_list = [ec.strip() for ec in x.split(";") if ec.strip()]
    return ec_list

def read_data(input_path: str, sep: str = ",") -> Tuple[np.ndarray, Dict[str, str], Dict[str, List[str]]]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    df = pd.read_csv(input_path, sep=sep)
    df.columns = df.columns.str.lower()

    required_cols = {"uniprot_id", "sequence"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV file must contain columns {required_cols}, missing: {missing_cols}")

    if "ec_num" not in df.columns:
        logger.warning(
            "CSV file does not contain column 'ec_num'. "
            "EC numbers will be saved as empty lists."
        )
        df["ec_num"] = ""

    df["uniprot_id"] = df["uniprot_id"].astype(str)
    df["sequence"] = df["sequence"].astype(str)

    entities = df["uniprot_id"].unique()
    logger.info(f"{len(entities)} unique entities found")

    id_to_seq = dict(zip(df["uniprot_id"], df["sequence"]))

    id_to_ec: Dict[str, List[str]] = {}
    for uid, ec in zip(df["uniprot_id"], df["ec_num"]):
        ec_list = parse_ec_num(ec)

        if uid not in id_to_ec:
            id_to_ec[uid] = []

        id_to_ec[uid].extend(ec_list)

    for uid in id_to_ec:
        id_to_ec[uid] = sorted(set(id_to_ec[uid]))

    return entities, id_to_seq, id_to_ec

def load_esm2(device: torch.device):
    """
    Load ESM-2 model + alphabet + batch_converter
    """
    logger.info(f"Loading ESM-2 (esm2_t33_650M_UR50D) on device={device} ...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

@torch.inference_mode()
def get_emb_esm2(seq: str, model, batch_converter, device: torch.device,
                 repr_layer: int) -> torch.Tensor:
    """
    Return torch.Tensor with shape (L, embed_dim), where L = len(seq) (<= MAX_LEN).
    This returns per-residue representations, excluding BOS/EOS.
    """
    # batch_converter expects list of (label, seq)
    data: List[Tuple[str, str]] = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = out["representations"][repr_layer]  # (B, T, C)

    # tokens include BOS + sequence + EOS (T = L + 2)
    # remove BOS/EOS => (L, C)
    return reps[0, 1:-1, :]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="pretrain_data/uniprot_sequences.csv",
                        help="Path to input.csv")
    parser.add_argument("--out_dir", default="features/protein",
                        help="Directory to save output pkl")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id to use (default: 3). Set <0 to run on CPU.")
    parser.add_argument("--feature_name", default="protein_pretrain",
                        help="Output filename prefix")
    parser.add_argument("--repr_layer", type=int, default=33,
                        help="Which layer to extract representations from (default: 33 for esm2_t33_650M)")
    args = parser.parse_args()

    entities, id_to_seq, id_to_ec = read_data(args.input_dir, sep=",")
    entities = list(map(str, entities))

    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU was requested.")

        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU {args.gpu}. MAX_LEN={MAX_LEN}")
    else:
        device = torch.device("cpu")
        logger.warning("Running on CPU (this will be very slow).")

    logger.info(f"Input: {args.input_dir}")

    final_out_path = os.path.join(args.out_dir, f"{args.feature_name}.pkl")
    logger.info(f"Output: {final_out_path}")

    model, alphabet, batch_converter = load_esm2(device)

    out_dict = {}

    tqdm.monitor_interval = 0

    pbar = tqdm(total=len(entities), desc=("GPU" if args.gpu >= 0 else "CPU"))
    last_update = 0

    for i, uid in enumerate(entities, start=1):
        seq = str(id_to_seq[uid]).strip().upper()
        seq_trunc = seq[:MAX_LEN]
        L = len(seq_trunc)

        emb = get_emb_esm2(seq_trunc, model, batch_converter, device, args.repr_layer)

        embed_np = emb.detach().float().cpu().numpy().astype(np.float32)

        # safety: ensure first dim == L
        if embed_np.shape[0] != L:
            embed_np = embed_np[:L, :]

        out_dict[uid] = {
            "sequence": seq_trunc,
            "length": L,
            "features": embed_np,
            "ec_num": id_to_ec.get(uid, []),
        }

        if i % 1000 == 0 or i == len(entities):
            pbar.update(i - last_update)
            last_update = i

    pbar.close()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(final_out_path, "wb") as f:
        pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Saved features for {len(out_dict)} entities to {final_out_path}")

if __name__ == "__main__":
    main()