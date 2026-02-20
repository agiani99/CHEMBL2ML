# CHEMBL2ML

Streamlit web app that fetches ChEMBL bioactivity data for a target (via UniProt ID), computes molecular descriptors, and trains a simple predictive model (regression, with an optional classification fallback).

## What it does

- Input: a **UniProt ID** (e.g. `P00533`).
- Fetches:
  - HGNC gene symbol (via genenames.org)
  - ChEMBL target, assays, and activities (via ChEMBL API)
- Builds a dataset with:
  - `pchembl_value` as the main label
  - RDKit physicochemical descriptors (e.g. MW, LogP, HBD/HBA, TPSA, rings, rotatable bonds)
  - RDKit fragment descriptors (`fr_*`)
  - Optional ErG-style features using `FixedPharmacophoreAnalyzer` from `erg_calc_fragments_topo.py`
- Trains:
  - Regression: Random Forest (default) or XGBoost (if installed)
  - Optional fallback: binary classification (Active if `pChEMBL ≥ 6.5`) with optional SMOTE
- Exports:
  - Pickled model + preprocessing objects
  - Predictions CSV
  - Top-features JSON
  - Full dataset CSV

## Requirements

Core:
- Python 3.9+
- `streamlit`, `pandas`, `numpy`, `requests`, `scikit-learn`, `matplotlib`, `seaborn`

Optional:
- RDKit (required to compute descriptors; app will stop without it)
- `xgboost` (enables XGBoost option)
- `optuna` (enables hyperparameter search)
- `imbalanced-learn` (enables SMOTE for classification fallback)

## Install

Example (pip):

```bash
pip install streamlit pandas numpy requests scikit-learn matplotlib seaborn
```

RDKit installation depends on your platform (common options are conda-forge or a prebuilt wheel).

## Run

From this folder:

```bash
streamlit run .\chembl_ml_app.py
```

## Notes / troubleshooting

- ChEMBL API calls are rate-limited in the app; large targets may take time.
- If you see runtime errors when starting the app, check that the selected Python environment has the required packages installed.
- The ErG analyzer is loaded from `erg_calc_fragments_topo.py` and is optional.

