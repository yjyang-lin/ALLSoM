# ALLSoM

The SoM datasets used in this study will be released immediately upon acceptance of the manuscript!

ALLSoM is a transfer learning framework for predicting SoMs across major drug-metabolizing enzyme families. It combines molecular graph features with ESM-2 representations of enzyme sequences and transfers knowledge learned from enzymatic reactions to SoM prediction for CYP, UGT, AOX, FMO, and SULT. Atom-level SoM probabilities also support substrate assessment and candidate metabolite generation.

Web server: [https://lmmd.ecust.edu.cn/ALLSoM/](https://lmmd.ecust.edu.cn/ALLSoM/)

## Supported enzymes

- **CYP:** CYP1A2, CYP2A6, CYP2B6, CYP2C8, CYP2C9, CYP2C19, CYP2D6, CYP2E1, and CYP3A4
- **UGT:** UGT1A1, UGT1A3, UGT1A4, UGT1A7, UGT1A8, UGT1A9, UGT1A10, and UGT2B7
- **AOX:** AOX1
- **FMO and SULT:** prediction with pretrained atom representations and TabPFN

## Repository structure

```text
ALLSoM/
├── CDK/                   # CDK electronic descriptor calculation
├── featurize/             # Molecular and protein feature extraction
├── features/              # Molecular and protein representations
├── inference/             # External validation for CYP, UGT, and AOX
├── metab_data/            # CYP, UGT, and AOX data
├── model/                 # Pretraining and CYP, UGT, and AOX transfer learning
├── pretrain_data/         # Enzymatic reaction data for pretraining
├── product_generation/    # Rule-based candidate metabolite generation
├── runs_finetune/         # CYP, UGT, and AOX model checkpoints
├── runs_pretrain/         # Pretrained model checkpoints
├── tabpfn/                # FMO and SULT prediction with TabPFN
└── ugt_ml/                # Random forest and XGBoost models for UGT substrate prediction
```

## Requirements

The required software environment is specified in [`environment.yml`](https://github.com/yjyang-lin/ALLSoM/blob/main/environment.yml). Molecular featurization also requires OpenJDK 21 and the CDK descriptor program included in `CDK/`.

Compile the CDK descriptor program from the project root:

```bash
javac -cp CDK/atom_descriptor.jar CDK/CDKDescriptors.java
```

FMO and SULT classification uses a separate TabPFN environment. The tested versions and commands are provided in [`tabpfn/README.md`](tabpfn/README.md).

## Data preparation and feature extraction

Known SoM atoms are identified by atom map number `:1`, and substrate labels use `1` for substrates and `0` for non-substrates.

Generate ESM-2 residue representations:

```bash
python featurize/protein_esm2.py --input_dir pretrain_data/uniprot_sequences.csv --out_dir features/protein --feature_name protein_pretrain --gpu 0
python featurize/protein_esm2.py --input_dir metab_data/enzyme_seq.csv --out_dir features/protein --feature_name protein_metab --gpu 0
```

Generate molecular graph features:

```bash
python featurize/mol_featurize.py --mode train --in_sdf pretrain_data/pretrain_data_split.sdf --out_pkl features/molecule/pretrain_features.pkl
python featurize/mol_featurize.py --mode fine_tune --in_sdf metab_data/CYP/CYP_som_split.sdf --out_pkl features/molecule/CYP_features.pkl
python featurize/mol_featurize.py --mode fine_tune --in_sdf metab_data/UGT/UGT_som_split.sdf --out_pkl features/molecule/UGT_features.pkl
python featurize/mol_featurize.py --mode fine_tune --in_sdf metab_data/AOX/AOX_som.sdf --out_pkl features/molecule/AOX_features.pkl
```

## Pretraining

The following command trains one model using the enzyme-based split. The study used seeds 0, 1, and 2, stored as `exp_1`, `exp_2`, and `exp_3`, respectively.

```bash
python model/train.py --gpu 0 --mol_pkl features/molecule/pretrain_features.pkl --protein_pkl features/protein/protein_pretrain.pkl --split_mode split_enzyme_cluster --seed 0 --save_dir runs_pretrain/split_enzyme_cluster --run_name exp_1
```

Molecule- and scaffold-based pretraining are selected with `--split_mode split_smiles_group` and `--split_mode split_scaffold`.

## Transfer learning

Fine-tune the three enzyme-based pretrained models for CYP and UGT:

```bash
python model/finetune_cyp.py --gpu 0 --pretrain_split split_enzyme_cluster --finetune_split split_smiles_group
python model/finetune_ugt.py --gpu 0 --pretrain_split split_enzyme_cluster --finetune_split split_smiles_group
```

Use `--finetune_split split_scaffold` for scaffold-based evaluation.

Fine-tune AOX with grouped five-fold cross-validation:

```bash
python model/finetune_cv.py --gpu 0 --split_mode smiles --pretrain_ckpt runs_pretrain/split_enzyme_cluster/exp_1/best.pt --save_root runs_finetune/AOX_transfer/split_enzyme_cluster/exp_1
```

Repeat the AOX command for the three pretrained checkpoints. Scaffold-based cross-validation is selected with `--split_mode scaffold`.

## External validation

External validation scripts average predictions across the available model checkpoints.

```bash
python inference/CYP/cyp_external_validation.py --gpu 0
python inference/UGT/ugt_external_validation.py --gpu 0
python inference/AOX/aox_external_validation.py --gpu 0
```

The default input files are `inference/CYP/CYP_ex_val.csv`, `inference/UGT/UGT_ex_val.csv`, and `inference/AOX/AOX_ex_val.csv`. Alternative data and model locations can be supplied through the command-line arguments of each script.

UGT substrate baselines using random forest and XGBoost are implemented in `ugt_ml/`:

```bash
python ugt_ml/train_external_validation.py
```

## FMO and SULT prediction

Pretrained atom representations are extracted with the main ALLSoM environment and classified with TabPFN:

```bash
python tabpfn/extract_FMO_SULT_features.py
python tabpfn/run_FMO_SULT_tabpfn.py
```

## Candidate metabolite generation

Candidate metabolites are generated using metabolic reaction rules from [GLORYx](https://github.com/christinadebruynkops/GLORYx) and ranked using predicted SoM probabilities:

```bash
python product_generation/generate_products.py --csv product_generation/case_study_molecules.csv --output product_generation/case_study_products.csv --rules product_generation/gloryx_reactionrules.csv --gpu 0
```
