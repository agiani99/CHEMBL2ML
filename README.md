# ChEMBL ML Predictor (Streamlit)

Streamlit web app that fetches ChEMBL bioactivity data for a target (via UniProt ID), computes molecular descriptors, and trains a simple predictive model (regression, with an optional classification fallback).

## What it does

- Input: a **UniProt ID** (e.g. `Q9UNA0`).
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

## Read a generated PKL from `output`

The analysis pipeline in `erg_eda_enhanced_with_optuna.py` writes per-target pickle files under the run output folder, for example:

```text
output/erg_analysis_inhibition_activation_YYYYMMDD_HHMMSS/models/inhibition/AKT1_results.pkl
```

You can load one of those files with:

```python
import pickle
from pathlib import Path

pkl_path = Path("output/erg_analysis_inhibition_activation_YYYYMMDD_HHMMSS/models/inhibition/AKT1_results.pkl")

with pkl_path.open("rb") as handle:
  results = pickle.load(handle)

print(results["target"])
print(results["n_samples"])
print(results["regression"].keys())
print(results["classification"].keys())
```

Typical top-level keys in the loaded object are:

- `target`
- `threshold`
- `n_samples`
- `regression`
- `classification`
- `plots`

Example access to a saved regression model result:

```python
rf_regression = results["regression"].get("random_forest")

if rf_regression:
  print("R2:", rf_regression["r2"])
  print("RMSE:", rf_regression["rmse"])
  print("Top features:", rf_regression["top_features"][:5])
```

## Notes / troubleshooting

- ChEMBL API calls are rate-limited in the app; large targets may take time.
- If you see runtime errors when starting the app, check that the selected Python environment has the required packages installed.
- The ErG analyzer is loaded from `erg_calc_fragments_topo.py` and is optional.
