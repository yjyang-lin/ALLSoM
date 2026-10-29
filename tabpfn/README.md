# TabPFN workflow

Feature extraction uses the main ALLSoM environment:

```bash
python tabpfn/extract_FMO_SULT_features.py
```

Classification uses a separate environment with the following versions:

- Python 3.11.16
- TabPFN 9.0.0
- PyTorch 2.14.0+cu130
- CUDA 13.0

Run the FMO and SULT classifiers from the project root:

```bash
python tabpfn/run_FMO_SULT_tabpfn.py
```

The `node_onehot/` directory contains the corresponding baseline using the original atom features and one-hot enzyme isoform encodings.
