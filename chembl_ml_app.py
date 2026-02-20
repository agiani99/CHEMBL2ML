"""
ChEMBL ML Predictor - Web Application
=====================================
Upload a UniProt ID and train a predictive model on ChEMBL bioactivity data
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import json
import time
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# ML and data processing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef
# Optional SMOTE
SMOTE = None
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# RDKit for molecular descriptors
Chem = None
Descriptors = None
AllChem = None
Fragments = None
Crippen = None
Lipinski = None
MoleculeDescriptors = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, Fragments, Crippen, Lipinski
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.error("⚠️ RDKit not available. Install with: pip install rdkit")

# Try to import the ErG analyzer from local module
FixedPharmacophoreAnalyzer: Any = None
try:
    # Workspace provides erg_calc_fragments_topo.py
    from erg_calc_fragments_topo import FixedPharmacophoreAnalyzer
    ERG_ANALYZER_AVAILABLE = True
except Exception:
    ERG_ANALYZER_AVAILABLE = False
    st.warning("ErG analyzer not available. Place erg_calc_fragments.py in the same folder to add ErG columns.")

# XGBoost and Optuna
XGBRegressor: Any = None
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

optuna: Any = None
TPESampler: Any = None
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ChEMBL ML Predictor",
    page_icon="🧬",
    layout="wide"
)

# Supported activity value types (fetch once for all, filter later)
ALL_VALUE_TYPES = ['IC50', 'EC50', 'AC50', 'DC50', 'Ki', 'Kd']

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'full_data' not in st.session_state:
    st.session_state.full_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'erg_analyzer' not in st.session_state and ERG_ANALYZER_AVAILABLE and FixedPharmacophoreAnalyzer is not None:
    try:
        st.session_state.erg_analyzer = FixedPharmacophoreAnalyzer()
    except Exception:
        st.session_state.erg_analyzer = None

# ============================================================================
# CHEMBL API FUNCTIONS
# ============================================================================

def safe_float(x):
    """Convert to float or return None."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def get_hgnc_from_uniprot(uniprot_id):
    """Retrieve HGNC gene symbol from UniProt ID"""
    try:
        url = f"https://rest.genenames.org/fetch/uniprot_ids/{uniprot_id}"
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            docs = data.get('response', {}).get('docs', [])
            if docs:
                return docs[0].get('symbol')
        return None
    except Exception as e:
        st.error(f"Error fetching HGNC: {e}")
        return None

def get_chembl_target_by_uniprot(uniprot_id):
    """Get ChEMBL target ID from UniProt ID"""
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/target.json"
        params = {'target_components__accession': uniprot_id}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            targets = data.get('targets', [])
            if targets:
                return targets[0]['target_chembl_id']
        return None
    except Exception as e:
        st.error(f"Error fetching ChEMBL target: {e}")
        return None

def get_chembl_assays(target_chembl_id, assay_types=['B', 'F'], confidence_levels=[8, 9]):
    """Get biochemical assays for target"""
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/assay.json"
        params = {
            'target_chembl_id': target_chembl_id,
            'limit': 1000
        }
        response = requests.get(url, params=params, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            assays = data.get('assays', [])
            
            # Filter by assay type and confidence
            filtered_assays = []
            for assay in assays:
                assay_type = assay.get('assay_type')
                confidence = assay.get('confidence_score')
                
                if (assay_type in assay_types and 
                    confidence in confidence_levels):
                    filtered_assays.append(assay['assay_chembl_id'])
            
            return filtered_assays
        return []
    except Exception as e:
        st.error(f"Error fetching assays: {e}")
        return []

def get_activities_for_assays(assay_ids, value_types=['IC50', 'EC50', 'AC50', 'DC50', 'Ki', 'Kd']):
    """Get activities from assays with specified value types"""
    all_activities = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, assay_id in enumerate(assay_ids):
        status_text.text(f"Fetching activities from assay {idx+1}/{len(assay_ids)}")
        progress_bar.progress((idx + 1) / len(assay_ids))
        
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
            params = {
                'assay_chembl_id': assay_id,
                'pchembl_value__isnull': 'false',
                'limit': 1000
            }
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                activities = data.get('activities', [])
                
                for act in activities:
                    if (act.get('standard_type') in value_types and
                        act.get('pchembl_value') is not None):
                        all_activities.append(act)
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            st.warning(f"Error fetching activities for {assay_id}: {e}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return all_activities

def get_molecule_data(molecule_chembl_id):
    """Get molecule SMILES and properties"""
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_chembl_id}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            structures = data.get('molecule_structures', {})
            props = data.get('molecule_properties', {})
            
            return {
                'smiles': structures.get('canonical_smiles'),
                'mw': safe_float(props.get('full_mwt')),
                'alogp': safe_float(props.get('alogp')),
                'psa': safe_float(props.get('psa'))
            }
        return None
    except:
        return None

# ============================================================================
# MOLECULAR DESCRIPTOR FUNCTIONS
# ============================================================================

def calculate_rdkit_descriptors(smiles):
    """Calculate RDKit fr_* and physicochemical descriptors"""
    if not RDKIT_AVAILABLE:
        return {}

    if Chem is None or Descriptors is None or Fragments is None or Crippen is None or Lipinski is None:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Physicochemical descriptors
        descriptors = {
            'MW': Descriptors.MolWt(mol),  # type: ignore[attr-defined]
            'LogP': Crippen.MolLogP(mol),  # type: ignore[attr-defined]
            'HBD': Lipinski.NumHDonors(mol),  # type: ignore[attr-defined]
            'HBA': Lipinski.NumHAcceptors(mol),  # type: ignore[attr-defined]
            'TPSA': Descriptors.TPSA(mol),  # type: ignore[attr-defined]
            'RotatableBonds': Descriptors.NumRotatableBonds(mol),  # type: ignore[attr-defined]
            'AromaticRings': Descriptors.NumAromaticRings(mol)  # type: ignore[attr-defined]
        }
        
        # Fragment descriptors (fr_*)
        for name in dir(Fragments):
            if name.startswith('fr_'):
                func = getattr(Fragments, name)
                if callable(func):
                    try:
                        descriptors[name] = func(mol)
                    except:
                        descriptors[name] = np.nan
        
        return descriptors
        
    except Exception as e:
        return {}

def calculate_erg_fingerprint_simple(smiles):
    """Simplified ErG-like fingerprint (without full 3D conformer generation)"""
    if not RDKIT_AVAILABLE:
        return {}

    if Chem is None or Descriptors is None:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Simple pharmacophore counts (approximation without 3D)
        erg_features = {
            'count_HA': 0,  # H-bond acceptors
            'count_HD': 0,  # H-bond donors
            'count_AR': 0,  # Aromatic rings
            'count_HY': 0,  # Hydrophobic centers
            'count_PLUS': 0,  # Positive charges
            'count_MINUS': 0  # Negative charges
        }
        
        # Count features
        for atom in mol.GetAtoms():
            # Hydrogen bond acceptors
            if atom.GetSymbol() in ['O', 'N']:
                erg_features['count_HA'] += 1
            
            # Hydrogen bond donors (atoms with H)
            if atom.GetSymbol() in ['O', 'N'] and atom.GetTotalNumHs() > 0:
                erg_features['count_HD'] += 1
            
            # Charged atoms
            if atom.GetFormalCharge() > 0:
                erg_features['count_PLUS'] += 1
            elif atom.GetFormalCharge() < 0:
                erg_features['count_MINUS'] += 1
            
            # Hydrophobic (non-aromatic carbons)
            if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
                erg_features['count_HY'] += 1
        
        # Aromatic rings
        erg_features['count_AR'] = Descriptors.NumAromaticRings(mol)  # type: ignore[attr-defined]
        
        return erg_features
        
    except Exception as e:
        return {}

# ============================================================================
# MACHINE LEARNING FUNCTIONS
# ============================================================================

def prepare_ml_dataset(df):
    """Prepare dataset for ML by selecting descriptor columns"""
    # Identify descriptor columns
    descriptor_cols = []
    
    # Basic descriptors
    basic_desc = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotatableBonds', 'AromaticRings']
    descriptor_cols.extend([col for col in basic_desc if col in df.columns])
    
    # Fragment descriptors
    fr_cols = [col for col in df.columns if col.startswith('fr_')]
    descriptor_cols.extend(fr_cols)
    
    # ErG counts (from simple or analyzer)
    erg_cols = [col for col in df.columns if col.startswith('count_')]
    descriptor_cols.extend(erg_cols)

    # ErG distance-bin features (HA_HD_d1 ... etc.)
    dist_cols = [c for c in df.columns if any(c.endswith(f"_d{i}") for i in range(1, 16))]
    descriptor_cols.extend(dist_cols)
    
    return descriptor_cols

def train_model_with_optuna(X_train, y_train, X_test, y_test, model_type='xgboost', n_trials=50):
    """Train model with Optuna hyperparameter optimization"""

    best_model = None
    best_params: Dict[str, Any] = {}
    
    if model_type == 'xgboost' and XGBOOST_AVAILABLE and OPTUNA_AVAILABLE and optuna is not None and TPESampler is not None:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'n_jobs': -1
            }
            
            if XGBRegressor is None:
                raise RuntimeError('XGBoost is not available')
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            return r2
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = dict(study.best_params)
        if XGBRegressor is None:
            raise RuntimeError('XGBoost is not available')
        best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        
    elif model_type == 'random_forest' or not XGBOOST_AVAILABLE:
        if OPTUNA_AVAILABLE and optuna is not None and TPESampler is not None:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                model = RandomForestRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                return r2
            
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = dict(study.best_params)
            best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        else:
            # Default parameters if Optuna not available
            best_params = {'n_estimators': 200, 'max_depth': 10}
            best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    
    if best_model is None:
        raise RuntimeError('No model could be constructed (missing optional dependency).')

    # Train final model
    best_model.fit(X_train, y_train)
    
    return best_model, best_params

def evaluate_model(model, X_test, y_test, descriptor_names):
    """Evaluate model and extract top features"""
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': descriptor_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_10_features = feature_importance.head(10)
    else:
        top_10_features = pd.DataFrame()
    
    results = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'y_test': y_test,
        'y_pred': y_pred,
        'top_10_features': top_10_features
    }
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_predictions(y_test, y_pred):
    """Create scatter plot of actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_test, y_pred, alpha=0.5, s=50)
    
    # Add diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax.set_xlabel('Actual pChEMBL', fontsize=12)
    ax.set_ylabel('Predicted pChEMBL', fontsize=12)
    ax.set_title('Test Set Predictions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_feature_importance(top_features):
    """Create horizontal bar chart for top 10 features"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = top_features['feature'].values
    importances = top_features['importance'].values
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 10 Most Influential Descriptors', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig

def plot_roc_curve_simple(y_test, y_proba):
    """Simple ROC curve plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    if y_proba is not None and len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return fig

def plot_confusion_matrix_simple(cm):
    """Simple confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.title("🧬 ChEMBL ML Predictor")
    st.markdown("### Train predictive models on ChEMBL bioactivity data")
    
    st.sidebar.header("⚙️ Configuration")
    # Allow selecting activity value types from the sidebar
    st.sidebar.subheader("Activity value types")
    default_value_types = ALL_VALUE_TYPES
    selected_value_types = [
        vt for vt in default_value_types
        if st.sidebar.checkbox(vt, value=True, key=f"vt_{vt}")
    ]
    if not selected_value_types:
        st.sidebar.warning("No value types selected. Defaulting to IC50.")
        selected_value_types = ['IC50']

    # Step 1: UniProt ID Input
    st.header("1️⃣ Input UniProt ID")
    uniprot_id = st.text_input("Enter UniProt ID:", placeholder="e.g., P00533")
    
    if st.button("🔍 Fetch Data from ChEMBL"):
        if not uniprot_id:
            st.error("Please enter a UniProt ID")
            return
        
        with st.spinner("Fetching data from ChEMBL..."):
            # Get HGNC name
            hgnc_symbol = get_hgnc_from_uniprot(uniprot_id)
            if hgnc_symbol:
                st.success(f"✅ HGNC Symbol: **{hgnc_symbol}**")
            else:
                st.warning("⚠️ Could not retrieve HGNC symbol")
            
            # Get ChEMBL target
            target_id = get_chembl_target_by_uniprot(uniprot_id)
            if not target_id:
                st.error("❌ No ChEMBL target found for this UniProt ID")
                return
            
            st.info(f"📌 ChEMBL Target ID: **{target_id}**")
            
            # Get assays
            st.info("Fetching biochemical assays (Type B/F, Confidence 8-9)...")
            assay_ids = get_chembl_assays(target_id)
            
            if not assay_ids:
                st.error("❌ No qualifying assays found")
                return
            
            st.success(f"✅ Found {len(assay_ids)} qualifying assays")
            
            # Get activities (fetch all supported types once)
            st.info("Fetching bioactivity data...")
            activities = get_activities_for_assays(assay_ids, value_types=ALL_VALUE_TYPES)
            
            if not activities:
                st.error("❌ No activities found")
                return
            
            st.success(f"✅ Found {len(activities)} activity records")
            
            # Build dataframe
            st.info("Processing molecules and calculating descriptors...")
            data_records = []
            
            progress_bar = st.progress(0)
            for idx, act in enumerate(activities):
                progress_bar.progress((idx + 1) / len(activities))
                
                mol_id = act.get('molecule_chembl_id')
                pchembl = act.get('pchembl_value')
                
                if not mol_id or not pchembl:
                    continue
                
                # Get molecule data
                mol_data = get_molecule_data(mol_id)
                if not mol_data or not mol_data.get('smiles'):
                    continue
                
                smiles = mol_data['smiles']
                mw = mol_data.get('mw')
                
                # Filter by MW < 2000
                if mw is not None and mw >= 2000:
                    continue
                
                # Calculate descriptors
                rdkit_desc = calculate_rdkit_descriptors(smiles)
                erg_desc = calculate_erg_fingerprint_simple(smiles)

                # Generate full ErG fingerprint (3D-free, topological distances) and fr_* using analyzer
                erg_fp = None
                if ERG_ANALYZER_AVAILABLE and st.session_state.get('erg_analyzer') is not None:
                    try:
                        if Chem is None:
                            raise RuntimeError('RDKit not available')
                        mol_tmp = Chem.MolFromSmiles(smiles)
                        if mol_tmp is not None:
                            erg_fp = st.session_state.erg_analyzer.generate_erg_fingerprint(mol_tmp, smiles)
                    except Exception:
                        erg_fp = None

                record = {
                    'molecule_chembl_id': mol_id,
                    'smiles': smiles,
                    'pchembl_value': float(pchembl),
                    'activity_type': act.get('standard_type'),
                    'assay_chembl_id': act.get('assay_chembl_id')
                }
                # Merge descriptors: basic RDKit + simple ErG counts
                record.update(rdkit_desc)
                record.update(erg_desc)

                # Merge full ErG fingerprint (only count_* and *_d* keys)
                if erg_fp:
                    for k, v in erg_fp.items():
                        if k.startswith('count_') or k.endswith(tuple(f"_d{i}" for i in range(1, 16))):
                            record[k] = v

                data_records.append(record)
            
            progress_bar.empty()
            
            if not data_records:
                st.error("❌ No valid molecules found after filtering")
                return
            
            df = pd.DataFrame(data_records)
            df = df.dropna(subset=['pchembl_value'])
            
            # Store full dataset and filtered view
            st.session_state.full_data = df
            st.session_state.data = df[df['activity_type'].isin(selected_value_types)]
            st.session_state.hgnc_symbol = hgnc_symbol
            st.session_state.target_id = target_id
            
            st.success(f"✅ Dataset ready: {len(st.session_state.data)} rows (filtered) / {len(df)} total, {len(df.columns)} features")
    
    # Step 2: Display Dataset
    if st.session_state.full_data is not None:
        # Re-apply filter based on current sidebar selection without re-fetching
        st.session_state.data = st.session_state.full_data[
            st.session_state.full_data['activity_type'].isin(selected_value_types)
        ]
        st.header("2️⃣ Dataset Overview")
        df = st.session_state.data
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Compounds", len(df))
        with col2:
            st.metric("Descriptors", len(df.columns) - 5)
        with col3:
            st.metric("pChEMBL Range", f"{df['pchembl_value'].min():.1f} - {df['pchembl_value'].max():.1f}")
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Step 3: Train Predictive Model
        st.header("3️⃣ Train Predictive Model")
        
        col1, col2 = st.columns(2)
        with col1:
            options = ['Random Forest'] + (['XGBoost'] if XGBOOST_AVAILABLE else [])
            model_type = st.selectbox("Model Type", options)
        with col2:
            n_trials = st.slider("Optuna Trials", 10, 100, 50)
            remove_duplicates = st.checkbox(
                "Remove duplicate molecules (keep highest pChEMBL per molecule)",
                value=False
            )
        # Classification fallback option
        try_classification = st.checkbox("Try classification if R² < 0.70 (pChEMBL ≥ 6.5 => Active)", value=False)
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training model with Optuna optimization..."):
                # Prepare data (optionally deduplicate)
                df_train = df
                if remove_duplicates:
                    before = len(df_train)
                    df_train = (
                        df_train.sort_values('pchembl_value', ascending=False)
                                .drop_duplicates(subset=['molecule_chembl_id'], keep='first')
                    )
                    st.info(f"Deduplicated training set: {before} ➜ {len(df_train)} rows")
                
                descriptor_cols = prepare_ml_dataset(df_train)
                
                if len(descriptor_cols) == 0:
                    st.error("No valid descriptors found")
                    return
                
                X = df_train[descriptor_cols].values
                y = df_train['pchembl_value'].values
                
                # Train/test split (no leakage from preprocessing)
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Impute NaNs using training set statistics, then remove zero-variance based on training set
                imputer = SimpleImputer(strategy='median')
                X_train_imp = np.asarray(imputer.fit_transform(X_train_raw))
                X_test_imp = np.asarray(imputer.transform(X_test_raw))
                
                # Remove zero-variance descriptors using training set variance
                train_vars = np.var(X_train_imp, axis=0)
                non_zero_mask = train_vars > 0.0
                removed = int((~non_zero_mask).sum())
                if removed > 0:
                    st.info(f"Removed {removed} zero-variance descriptor(s)")
                X_train_imp = X_train_imp[:, non_zero_mask]
                X_test_imp = X_test_imp[:, non_zero_mask]
                descriptor_cols = [col for i, col in enumerate(descriptor_cols) if non_zero_mask[i]]
                
                # Standardize using training set only
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_imp)
                X_test = scaler.transform(X_test_imp)
                
                # Train regression model
                model_type_key = 'xgboost' if model_type == 'XGBoost' else 'random_forest'
                model, best_params = train_model_with_optuna(
                    X_train, y_train, X_test, y_test, 
                    model_type=model_type_key, n_trials=n_trials
                )
                
                # Evaluate regression
                results = evaluate_model(model, X_test, y_test, descriptor_cols)

                # Optional classification fallback (Optuna optimizing MCC)
                mode = 'regression'
                if try_classification and results['r2'] < 0.70:
                    st.warning("R² < 0.70. Running classification fallback (threshold pChEMBL ≥ 6.5)...")
                    # Build binary labels
                    labels = (df_train['pchembl_value'] >= 6.5).astype(int).values
                    # Same split indices
                    Xtr_raw_cls, Xte_raw_cls, ytr_bin, yte_bin = train_test_split(
                        X, labels, test_size=0.2, random_state=42
                    )
                    # Same preprocessing
                    Xtr_imp_cls = np.asarray(imputer.transform(Xtr_raw_cls))[:, non_zero_mask]
                    Xte_imp_cls = np.asarray(imputer.transform(Xte_raw_cls))[:, non_zero_mask]
                    Xtr_cls = scaler.transform(Xtr_imp_cls)
                    Xte_cls = scaler.transform(Xte_imp_cls)
                    
                    # SMOTE once if minority class < 35%
                    Xtr_bal, ytr_bal = Xtr_cls, ytr_bin
                    minority_frac = min(np.mean(ytr_bin == 0), np.mean(ytr_bin == 1))
                    if minority_frac < 0.35 and IMBLEARN_AVAILABLE and SMOTE is not None:
                        try:
                            minority_count = min(np.sum(ytr_bin == 0), np.sum(ytr_bin == 1))
                            k_neighbors = max(1, min(5, minority_count - 1))
                            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                            resampled = sm.fit_resample(Xtr_cls, ytr_bin)
                            Xtr_bal, ytr_bal = resampled[0], resampled[1]
                            st.info(f"Applied SMOTE: training samples {Xtr_cls.shape[0]} ➜ {Xtr_bal.shape[0]}")
                        except Exception as e:
                            st.warning(f"SMOTE failed: {e}")
                    elif minority_frac < 0.35 and not IMBLEARN_AVAILABLE:
                        st.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")
                    
                    # Optuna objective maximizing MCC on validation (test split)
                    best_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}
                    if OPTUNA_AVAILABLE and optuna is not None and TPESampler is not None:
                        def objective(trial):
                            params = {
                                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                                'max_depth': trial.suggest_int('max_depth', 5, 20),
                                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                            }
                            clf_tmp = RandomForestClassifier(
                                **params, random_state=42, n_jobs=-1
                            )
                            clf_tmp.fit(Xtr_bal, ytr_bal)
                            y_pred_val = clf_tmp.predict(Xte_cls)
                            try:
                                mcc = matthews_corrcoef(yte_bin, y_pred_val)
                            except Exception:
                                mcc = 0.0
                            return mcc
                        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
                        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
                        best_params = dict(study.best_params)
                    
                    # Train final classifier with best params
                    clf = RandomForestClassifier(
                        n_estimators=int(best_params['n_estimators']),
                        max_depth=int(best_params['max_depth']),
                        min_samples_split=int(best_params['min_samples_split']),
                        min_samples_leaf=int(best_params['min_samples_leaf']),
                        random_state=42,
                        n_jobs=-1,
                    )
                    clf.fit(Xtr_bal, ytr_bal)
                    y_proba = clf.predict_proba(Xte_cls)[:, 1]
                    y_pred_cls = (y_proba >= 0.5).astype(int)
                    acc = float((y_pred_cls == yte_bin).mean())
                    try:
                        mcc = float(matthews_corrcoef(yte_bin, y_pred_cls))
                    except Exception:
                        mcc = 0.0
                    cm = confusion_matrix(yte_bin, y_pred_cls)
                    
                    # Top-10 features
                    if hasattr(clf, 'feature_importances_'):
                        importances = clf.feature_importances_
                        top_idx = np.argsort(importances)[::-1][:10]
                        top_10 = pd.DataFrame({
                            'feature': [descriptor_cols[i] for i in top_idx],
                            'importance': importances[top_idx]
                        })
                    else:
                        top_10 = pd.DataFrame()
                    
                    # Override results for classification mode
                    model = clf
                    results = {
                        'accuracy': acc,
                        'mcc': mcc,
                        'y_test': yte_bin,
                        'y_pred': y_pred_cls,
                        'y_proba': y_proba,
                        'confusion_matrix': cm,
                        'top_10_features': top_10
                    }
                    mode = 'classification'
                
                # Store results
                st.session_state.model_results = {
                    'mode': mode,
                    'model': model,
                    'imputer': imputer,
                    'scaler': scaler,
                    'descriptor_cols': descriptor_cols,
                    'best_params': best_params,
                    'results': results,
                    'model_type': model_type
                }
                st.session_state.training_complete = True
                
                st.success("✅ Model training complete!")
    
    # Step 4: Display Results
    if st.session_state.training_complete:
        st.header("4️⃣ Model Results")
        
        results = st.session_state.model_results['results']
        mode = st.session_state.model_results.get('mode', 'regression')
        
        if mode == 'classification':
            # Metrics (no ROC shown)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.3f}")
            with col2:
                st.metric("MCC", f"{results['mcc']:.3f}")
            
            # Only Confusion Matrix and Top Features
            st.subheader("Confusion Matrix")
            fig_cm = plot_confusion_matrix_simple(results['confusion_matrix'])
            st.pyplot(fig_cm)
            
            st.subheader("Top 10 Influential Descriptors")
            if isinstance(results.get('top_10_features'), pd.DataFrame) and not results['top_10_features'].empty:
                fig2 = plot_feature_importance(results['top_10_features'])
                st.pyplot(fig2)
        else:
            # Regression (default)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{results['r2']:.3f}")
            with col2:
                st.metric("RMSE", f"{results['rmse']:.3f}")
            with col3:
                st.metric("MAE", f"{results['mae']:.3f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Test Set Predictions")
                fig1 = plot_predictions(results['y_test'], results['y_pred'])
                st.pyplot(fig1)
            with col2:
                st.subheader("Top 10 Influential Descriptors")
                if not results['top_10_features'].empty:
                    fig2 = plot_feature_importance(results['top_10_features'])
                    st.pyplot(fig2)

        # Best parameters
        with st.expander("🔧 Best Hyperparameters / Settings"):
            st.json(st.session_state.model_results['best_params'])
        
        # Step 5: Download Results
        st.header("5️⃣ Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Download model (pickle)
        with col1:
            model_data = {
                'mode': st.session_state.model_results.get('mode', 'regression'),
                'model': st.session_state.model_results['model'],
                'imputer': st.session_state.model_results['imputer'],
                'scaler': st.session_state.model_results['scaler'],
                'descriptor_cols': st.session_state.model_results['descriptor_cols'],
                'best_params': st.session_state.model_results['best_params'],
                'metrics': (
                    {
                        'r2': results.get('r2'),
                        'rmse': results.get('rmse'),
                        'mae': results.get('mae')
                    } if st.session_state.model_results.get('mode', 'regression') == 'regression' else
                    {
                        'accuracy': results.get('accuracy'),
                        'mcc': results.get('mcc')
                    }
                )
            }
            
            buffer = BytesIO()
            pickle.dump(model_data, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="📦 Download Model (PKL)",
                data=buffer,
                file_name=f"chembl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream"
            )
        
        # Download predictions (CSV)
        with col2:
            if st.session_state.model_results.get('mode', 'regression') == 'classification':
                pred_df = pd.DataFrame({
                    'label_actual': results['y_test'],
                    'label_pred': results['y_pred'],
                    'prob_active': results['y_proba']
                })
            else:
                pred_df = pd.DataFrame({
                    'actual': results['y_test'],
                    'predicted': results['y_pred'],
                    'residual': results['y_test'] - results['y_pred']
                })
            
            csv_buffer = BytesIO()
            pred_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="📊 Download Predictions (CSV)",
                data=csv_buffer,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Download top features (JSON)
        with col3:
            top_feats = results.get('top_10_features')
            if isinstance(top_feats, pd.DataFrame):
                top_records = top_feats.to_dict('records')
            else:
                top_records = []
            features_data = {
                'mode': st.session_state.model_results.get('mode', 'regression'),
                'top_10_features': top_records,
                'metrics': (
                    {
                        'r2': float(results['r2']),
                        'rmse': float(results['rmse']),
                        'mae': float(results['mae'])
                    } if st.session_state.model_results.get('mode', 'regression') == 'regression' else
                    {
                        'accuracy': float(results['accuracy']),
                        'mcc': float(results['mcc'])
                    }
                )
            }
            
            json_buffer = BytesIO()
            json_buffer.write(json.dumps(features_data, indent=2).encode())
            json_buffer.seek(0)
            
            st.download_button(
                label="📋 Download Features (JSON)",
                data=json_buffer,
                file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Download full dataset
        st.markdown("---")
        if st.session_state.data is not None:
            csv_full = BytesIO()
            st.session_state.data.to_csv(csv_full, index=False)
            csv_full.seek(0)
            
            st.download_button(
                label="💾 Download Full Dataset (CSV)",
                data=csv_full,
                file_name=f"chembl_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    if not RDKIT_AVAILABLE:
        st.error("⚠️ RDKit is required. Install with: `pip install rdkit`")
        st.stop()
    
    main()
