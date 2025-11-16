#!/usr/bin/env python3
"""
lga_analyzer_with_settlements_and_labels.py

Merged Streamlit app with:
- Builtin LGA centroids (Gabasawa, Kiru, Ungogo)
- Settlement coordinates (from user's paste)
- LGA & child predictions (loads artifacts from ./artifacts/)
- Settlement-level aggregation, distance-to-centroid, prioritization, maps
- In-app LGA retrain flow (optional)
- LGA labels on map and settlement cluster visualization with different colours and visual styles

NOTE: Modified so settlement points are grouped & coloured by LGA (3 colours).
Also fixed pydeck tooltip formatting by precomputing text fields.
"""

from pathlib import Path
import zipfile
import base64
import tempfile
import logging
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import json
import math
import time
import warnings

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Paths & artifacts
# ----------------------
BASE = Path.cwd()
ARTIFACTS = BASE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

LGA_MODEL_FILE = ARTIFACTS / "lga_dropoff_model.joblib"
LGA_FEATURES_FILE = ARTIFACTS / "lga_features.joblib"
LGA_REPORT_FILE = ARTIFACTS / "lga_report.csv"
# improved outputs
LGA_MODEL_IMP = ARTIFACTS / "lga_dropoff_model_improved.joblib"
LGA_FEATURES_IMP = ARTIFACTS / "lga_features_improved.joblib"
LGA_REPORT_IMP = ARTIFACTS / "lga_report_improved.csv"
LGA_PERM_IMPORTANCE = ARTIFACTS / "lga_perm_importance.csv"

CHILD_MODEL_FILE = ARTIFACTS / "child_dropoff_model.joblib"
CHILD_FEATURES_FILE = ARTIFACTS / "child_features.joblib"

# ----------------------
# Built-in coordinates & settlements
# ----------------------
BUILTIN_LGA_COORDS = {
    "GABASAWA LGA": {"latitude": 12.18007, "longitude": 8.91123},
    "KIRU LGA": {"latitude": 11.70158, "longitude": 8.13637},
    "UNGOGO LGA": {"latitude": 12.09209, "longitude": 8.49552}
}

SETTLEMENT_COORDS = {
  # Gabasawa (partial)
  "malamawa": (12.26146, 8.25433),
  "santsi chikin gari": (11.99068, 8.54526),
  "wasarde": (12.03696, 8.86530),
  "hunbunare": (10.26729, 12.47183),
  "gagarawa": (12.40853, 9.52885),
  "mekiya gabas": (12.23550, 8.87346),
  "magama": (12.20037, 8.92292),
  "wadugur": (12.11098, 9.02575),
  "garin malam": (11.68375, 8.37204),
  "joda": (11.99896, 8.87447),
  "kurukuru": (13.55160, 6.01388),
  "kadage": (11.02315, 7.45228),
  "daurawa": (11.65416, 8.16492),
  "sabon gari": (12.01787, 8.53577),
  "gunduwa": (12.02285, 8.63746),
  "shargalle": (12.96121, 8.10400),
  "birgima": (11.95955, 7.37355),
  "yautar arewa": (12.26954, 8.75665),
  "cikin garin yauta": (10.72794, 7.93235),
  "badawa": (12.01379, 8.56845),
  "mazan gudu": (12.19529, 8.90722),
  "takalmawa": (12.18953, 8.85239),
  "doga": (12.21226, 8.79709),
  "badage": (12.27130, 9.14211),
  "tumbau": (11.99447, 8.81054),
  "unguwar gara": (11.94637, 8.55530),
  "daneji": (11.56536, 7.85565),
  "garin danga": (12.21088, 8.84309),
  "odoji": (7.11088, 5.07387),
  # Kiru
  "gidan danfadama": (13.05554, 5.17737),
  "unguwar liman": (10.01275, 9.78147),
  "tashar yanharawa": (11.96639, 7.74656),
  "makera": (10.46186, 7.39710),
  "tsohon gari": (11.25139, 8.39961),
  "maraku cikin gari": (10.64736, 8.68824),
  "unguwar maishuni": (11.55724, 8.99631),
  "gidan makama": (11.98870, 8.52104),
  "makera kofar gabas": (12.98455, 7.58892),
  "sarkakiya": (12.12741, 8.33788),
  "kwangwaro dutse": (11.49153, 9.58014),
  # Ungogo
  "jangaru": (11.49168, 8.58721),
  "rimi gata": (10.52769, 6.93541),
  "yar aduwa": (9.06752, 7.48674),
  "kududdufawa": (12.04614, 8.44683),
  "rijiyar zaki": (12.02836, 8.45240),
  "tsamiyar tazarce": (12.23450, 8.51193),
  "dausayi/rijiyar dinya": (12.05110, 8.47506),
  "rimin zakara": (12.00037, 8.44629),
  "muntsira": (11.97560, 8.43996),
  "kadawa?gabas": (11.64658, 8.44733),
  "gidan gona": (10.49585, 4.90003),
  "zangon marikita": (12.23481, 8.41515),
}

# ----------------------
# Utilities
# ----------------------
def read_csv_safe(upload):
    if upload is None:
        return None
    try:
        return pd.read_csv(upload, dtype=str, low_memory=False)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def save_zip_of(paths: List[Path], zip_name: str = "artifacts_export.zip"):
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    with zipfile.ZipFile(tf.name, 'w') as zf:
        for p in paths:
            if p.exists():
                zf.write(p, arcname=p.name)
    with open(tf.name, 'rb') as fh:
        b64 = base64.b64encode(fh.read()).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="{zip_name}">Download {zip_name}</a>'
        st.markdown(href, unsafe_allow_html=True)

def download_link(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def clean_distance_series(s: pd.Series) -> pd.Series:
    def clean(x):
        if pd.isna(x):
            return np.nan
        try:
            t = str(x).strip()
            if t.lower().endswith("km"):
                t = t[:-2]
            t = t.replace(",", "").strip()
            return float(t) if t else np.nan
        except Exception:
            return np.nan
    return s.apply(clean)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def rgba_to_hex(color):
    r,g,b = color[0], color[1], color[2]
    return f"#{r:02x}{g:02x}{b:02x}"

# ----------------------
# Data preparation & feature engineering for LGA retrain
# ----------------------
def prepare_zerodose(zd: pd.DataFrame) -> pd.DataFrame:
    zd = zd.copy()
    zd.columns = [c.strip() for c in zd.columns]
    if 'Distance to HF' in zd.columns:
        zd['Distance to HF'] = clean_distance_series(zd['Distance to HF'])
    if 'estimated_age_months' in zd.columns:
        zd['estimated_age_months'] = pd.to_numeric(zd['estimated_age_months'], errors='coerce')
    if 'Status' not in zd.columns:
        raise ValueError("zerodose.csv must contain a 'Status' column.")
    zd['dropoff'] = (zd['Status'].astype(str).str.strip().str.lower() == 'active').astype(int)
    zd['LGA_norm'] = zd['LGA'].astype(str).str.strip()
    zd['Settlement_norm'] = zd.get('Settlement','').astype(str).str.strip().str.lower()
    return zd

def build_lga_features(zd: pd.DataFrame, fv: Optional[pd.DataFrame]=None) -> pd.DataFrame:
    if 'LGA_norm' not in zd.columns:
        zd = prepare_zerodose(zd)
    g = zd.groupby('LGA_norm', dropna=False).agg(
        total_children=('dropoff','count'),
        dropoffs=('dropoff','sum'),
        mean_distance=('Distance to HF','mean'),
        mean_age_months=('estimated_age_months','mean')
    ).reset_index().rename(columns={'LGA_norm':'LGA'})
    g['dropoff_rate'] = g['dropoffs'] / g['total_children']
    # reason flags heuristic
    for c in zd.columns:
        if c in ['LGA','LGA_norm','Settlement','Settlement_norm','Status','dropoff','Distance to HF','estimated_age_months']:
            continue
        try:
            tmp = pd.to_numeric(zd[c], errors='coerce')
            nunique = tmp.dropna().nunique()
            if nunique <= 5:
                col_name = f"rate_{c}"
                rate = zd.groupby('LGA_norm')[c].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).mean()).reset_index(name=col_name)
                rate.rename(columns={'LGA_norm':'LGA'}, inplace=True)
                g = g.merge(rate, on='LGA', how='left')
        except Exception:
            pass
    # facility vaccine aggregates -> per-child rates
    if fv is not None:
        fv = fv.copy()
        fv.columns = [c.strip() for c in fv.columns]
        lga_col = next((c for c in ['lga_name','LGA','lga','lga_id'] if c in fv.columns), None)
        if lga_col is not None:
            fv = fv.rename(columns={lga_col:'lga_name'})
            candidate = [c for c in fv.columns if c not in ['lga_name','id','facility','date','Facility']]
            numeric = []
            for c in candidate:
                try:
                    fv[c] = pd.to_numeric(fv[c], errors='coerce').fillna(0)
                    numeric.append(c)
                except Exception:
                    continue
            if numeric:
                agg = fv.groupby('lga_name')[numeric].sum().reset_index().rename(columns={'lga_name':'LGA'})
                agg = agg.rename(columns={c: f"lga_vacc_{c}" for c in numeric})
                g = g.merge(agg, on='LGA', how='left')
                for c in numeric:
                    col = f"lga_vacc_{c}"
                    perchild = f"{col}_per_child"
                    g[perchild] = g[col] / g['total_children'].replace(0, np.nan)
    g.replace([np.inf, -np.inf], np.nan, inplace=True)
    g.fillna(0, inplace=True)
    return g

# ----------------------
# Simple retrain routine (keeps RandomForest & optional LightGBM attempt)
# ----------------------
def try_lightgbm_import():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

def weighted_kfold_eval(model, X, y, sample_weight, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        wtr = sample_weight.iloc[train_idx]
        # some models accept sample_weight at fit; try passing when available
        try:
            model.fit(Xtr, ytr, sample_weight=wtr)
        except TypeError:
            model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        maes.append(mean_absolute_error(yte, preds))
        rmses.append(mean_squared_error(yte, preds, squared=False))
        r2s.append(r2_score(yte, preds))
    return np.mean(maes), np.mean(rmses), np.mean(r2s)

def retrain_lga_model(lga_df: pd.DataFrame, cv: int=5, n_iter:int=30, min_children:int=1, drop_small=False):
    start = time.time()
    df = lga_df.copy()
    if drop_small:
        df = df[df['total_children'] >= min_children].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in {'LGA','dropoffs','total_children','dropoff_rate'}]
    X = df[feature_cols].astype(float).fillna(0)
    y = df['dropoff_rate'].astype(float)
    sample_weight = df['total_children'].astype(float).fillna(0).clip(lower=1.0)
    baseline_mae = mean_absolute_error(y, np.repeat(y.mean(), len(y)))
    # RandomForest search
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [100,200,400],
        'max_depth': [5,10,15,None],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4],
        'max_features': ['auto','sqrt',0.5]
    }
    rsearch = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=min(n_iter,30), cv=3,
                                 scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=0)
    rsearch.fit(X, y)
    best_rf = rsearch.best_estimator_
    rf_mae, rf_rmse, rf_r2 = weighted_kfold_eval(best_rf, X, y, sample_weight, cv=cv)
    best_model = best_rf
    best_score = rf_mae
    used_algo = "RandomForest"
    lgb = try_lightgbm_import()
    if lgb is not None:
        try:
            lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
            # quick param tries
            param_grid = [
                {'n_estimators':100, 'learning_rate':0.1, 'num_leaves':31},
                {'n_estimators':200, 'learning_rate':0.05, 'num_leaves':50},
                {'n_estimators':400, 'learning_rate':0.05, 'num_leaves':80},
            ]
            for p in param_grid[:min(n_iter, len(param_grid))]:
                model_try = lgb.LGBMRegressor(random_state=42, **p)
                model_try.fit(X, y, sample_weight=sample_weight)
                mae_cv, rmse_cv, r2_cv = weighted_kfold_eval(model_try, X, y, sample_weight, cv=cv)
                if mae_cv < best_score:
                    best_score = mae_cv
                    best_model = model_try
                    used_algo = "LightGBM"
        except Exception as e:
            logger.warning(f"LGB search failed: {e}")
    # final fit
    try:
        best_model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        best_model.fit(X, y)
    preds = best_model.predict(X)
    preds = np.clip(preds, 0.0, 1.0)
    final_mae = mean_absolute_error(y, preds)
    final_rmse = mean_squared_error(y, preds, squared=False)
    final_r2 = r2_score(y, preds)
    # permutation importance
    try:
        perm = permutation_importance(best_model, X, y, n_repeats=20, random_state=42, n_jobs=-1)
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std
        }).sort_values('importance_mean', ascending=False)
        imp_df.to_csv(LGA_PERM_IMPORTANCE, index=False)
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        imp_df = pd.DataFrame({'feature': X.columns, 'importance_mean': 0.0})
    # save artifacts
    joblib.dump(best_model, LGA_MODEL_IMP)
    joblib.dump(list(X.columns), LGA_FEATURES_IMP)
    out = df.copy()
    out['predicted_dropoff_rate'] = preds
    out.to_csv(LGA_REPORT_IMP, index=False)
    metrics = {
        'baseline_mae': baseline_mae,
        'rf_cv_mae': rf_mae,
        'rf_cv_rmse': rf_rmse,
        'rf_cv_r2': rf_r2,
        'final_mae': final_mae,
        'final_rmse': final_rmse,
        'final_r2': final_r2,
        'algo': used_algo
    }
    return best_model, list(X.columns), out, imp_df, metrics

# ----------------------
# Load existing artifacts
# ----------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    for p, key in [(LGA_MODEL_FILE, 'lga_model'), (LGA_FEATURES_FILE, 'lga_features'),
                   (LGA_REPORT_FILE, 'lga_report'), (LGA_MODEL_IMP, 'lga_model_imp'),
                   (LGA_FEATURES_IMP, 'lga_features_imp'), (LGA_REPORT_IMP, 'lga_report_imp'),
                   (CHILD_MODEL_FILE, 'child_model'), (CHILD_FEATURES_FILE, 'child_features')]:
        try:
            if p.exists():
                if p.suffix in ['.joblib', '.pkl']:
                    artifacts[key] = joblib.load(p)
                elif p.suffix == '.csv' or p.name.endswith('.csv'):
                    artifacts[key] = pd.read_csv(p)
        except Exception as e:
            st.warning(f"Could not load {p.name}: {e}")
    return artifacts

art = load_artifacts()
lga_model = art.get('lga_model')
lga_features = art.get('lga_features')
lga_model_imp = art.get('lga_model_imp')
lga_features_imp = art.get('lga_features_imp')
lga_report_imp = art.get('lga_report_imp')
precomputed_lga_report = art.get('lga_report')
child_model = art.get('child_model')
child_features = art.get('child_features')

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="LGA & Settlement Analyzer + Labels & Clusters", layout="wide")
st.title("🏘️ LGA & Settlement Dropoff Analyzer — Labels & Clusters")
st.markdown("""
Upload `zerodose.csv` (required). This app:
- Aggregates to LGA & settlement
- Uses builtin coordinates for Gabasawa, Kiru, Ungogo and settlement list
- Loads existing artifacts from `./artifacts/`
- Shows LGA labels and settlement clusters on maps
- Optionally retrain an improved LGA model (sidebar)
""")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    uploaded_zd = st.file_uploader("Upload zerodose.csv (required)", type=["csv"])
with col2:
    uploaded_fv = st.file_uploader("Upload facility_visit.csv (optional)", type=["csv"])
with col3:
    uploaded_geojson = st.file_uploader("Upload LGAs GeoJSON (optional)", type=["geojson","json"])

use_local = False
if uploaded_zd is None:
    if (BASE / "zerodose.csv").exists():
        st.info("No upload detected — using local zerodose.csv")
        use_local = True
    else:
        st.warning("Please upload zerodose.csv or place it in the working folder.")
        st.stop()

# Sidebar controls for retrain + map options
st.sidebar.header("Retrain & Map options")
do_retrain = st.sidebar.checkbox("Enable retrain", value=False)
min_children = st.sidebar.number_input("min_children for training", min_value=1, value=5, step=1)
drop_small = st.sidebar.checkbox("Drop LGAs with < min_children", value=False)
cv_folds = st.sidebar.number_input("CV folds", min_value=2, max_value=10, value=5, step=1)
n_iter = st.sidebar.number_input("Search iterations", min_value=5, max_value=200, value=30, step=5)
retrain_button = st.sidebar.button("Run retrain now")

# Map label toggles
show_lga_labels = st.sidebar.checkbox("Show LGA labels on map", value=True)
show_settlement_labels = st.sidebar.checkbox("Show settlement labels on map", value=False)
n_clusters = st.sidebar.slider("Number of settlement clusters (unused for grouping-by-LGA)", 2, 8, 3)

# Color palette for clusters (RGB lists)
COLOR_PALETTE = [
    [255, 99, 71, 180],    # tomato
    [60, 179, 113, 180],   # mediumseagreen
    [65, 105, 225, 180],   # royalblue
    [238, 130, 238, 180],  # violet
    [255, 215, 0, 180],    # gold
    [70, 130, 180, 180],   # steelblue
    [220,20,60,180],       # crimson
    [34,139,34,180]        # forestgreen
]

# Use first 3 for settlement-LGA grouping
SETTLEMENT_COLORS = COLOR_PALETTE[:3]

# Main action
if st.button("Run analysis (aggregate + model predict)"):
    try:
        # load data
        zd = read_csv_safe(uploaded_zd) if uploaded_zd is not None else pd.read_csv(BASE / "zerodose.csv", dtype=str, low_memory=False)
        fv = read_csv_safe(uploaded_fv) if uploaded_fv is not None else (pd.read_csv(BASE / "facility_visit.csv", dtype=str, low_memory=False) if (BASE / "facility_visit.csv").exists() else None)
        zd = prepare_zerodose(zd)

        # build lga features
        lga_feats_df = build_lga_features(zd, fv)

        # choose model (prefer improved if present)
        chosen_model = None
        chosen_features = None
        chosen_report = None
        if lga_model_imp is not None and lga_features_imp is not None and lga_report_imp is not None:
            chosen_model = lga_model_imp
            chosen_features = lga_features_imp
            chosen_report = lga_report_imp
            st.info("Using previously improved LGA model artifacts from ./artifacts/")
        elif LGA_MODEL_IMP.exists() and LGA_FEATURES_IMP.exists() and LGA_REPORT_IMP.exists():
            chosen_model = joblib.load(LGA_MODEL_IMP)
            chosen_features = joblib.load(LGA_FEATURES_IMP)
            chosen_report = pd.read_csv(LGA_REPORT_IMP)
            st.info("Loaded improved artifacts from disk.")
        elif lga_model is not None and lga_features is not None:
            chosen_model = lga_model
            chosen_features = lga_features
            chosen_report = precomputed_lga_report if precomputed_lga_report is not None else lga_feats_df[['LGA','total_children','dropoff_rate']].copy()
            st.info("Using existing LGA model artifact (original).")
        else:
            chosen_model = None
            chosen_features = None
            chosen_report = lga_feats_df[['LGA','total_children','dropoff_rate']].copy()
            st.info("No saved LGA model found — will show observed aggregation and you may retrain.")

        # Retrain if user asked
        if do_retrain and retrain_button:
            st.info("Starting retrain — this may take a while.")
            with st.spinner("Retraining LGA model..."):
                model_imp, feat_imp, report_imp, imp_df, metrics = retrain_lga_model(
                    lga_df=lga_feats_df,
                    cv=cv_folds,
                    n_iter=n_iter,
                    min_children=min_children,
                    drop_small=drop_small
                )
            chosen_model = model_imp
            chosen_features = feat_imp
            chosen_report = report_imp
            st.success("Retrain complete. Improved model and report saved to ./artifacts/")
            st.write("Retrain metrics (on training data):")
            st.json(metrics)
            try:
                if imp_df is not None and not imp_df.empty:
                    st.subheader("Permutation importances (top 20)")
                    st.dataframe(imp_df.head(20))
                    download_link(imp_df, LGA_PERM_IMPORTANCE.name, "Download permutation importances (CSV)")
            except Exception:
                pass

        # If improved model file exists but not loaded earlier, try load
        if chosen_model is None and LGA_MODEL_IMP.exists() and LGA_FEATURES_IMP.exists():
            chosen_model = joblib.load(LGA_MODEL_IMP)
            chosen_features = joblib.load(LGA_FEATURES_IMP)
            chosen_report = pd.read_csv(LGA_REPORT_IMP)
            st.info("Loaded improved model from disk.")

        # Predict on current LGA features using chosen_model (if available)
        lga_report = lga_feats_df.copy()
        if chosen_features is not None and chosen_model is not None:
            for f in chosen_features:
                if f not in lga_report.columns:
                    lga_report[f] = 0
            Xp = lga_report[chosen_features].astype(float).fillna(0)
            try:
                preds = chosen_model.predict(Xp)
            except Exception as e:
                preds = chosen_model.predict(Xp.values)
            preds = np.clip(preds, 0.0, 1.0)
            lga_report['predicted_dropoff_rate'] = preds
            def recommend(row):
                if row['predicted_dropoff_rate'] >= 0.30 and row.get('total_children', 0) >= 50:
                    return 'Immediate outreach'
                if row['predicted_dropoff_rate'] >= 0.15:
                    return 'Targeted awareness'
                return 'Monitor'
            lga_report['recommended_action'] = lga_report.apply(recommend, axis=1)
            try:
                mae = mean_absolute_error(lga_report['dropoff_rate'], lga_report['predicted_dropoff_rate'])
                rmse = float(np.sqrt(mean_squared_error(lga_report['dropoff_rate'], lga_report['predicted_dropoff_rate'])))
                r2 = r2_score(lga_report['dropoff_rate'], lga_report['predicted_dropoff_rate'])
                st.subheader("LGA model metrics (on current data)")
                st.metric("MAE", f"{mae:.4f}")
                st.metric("RMSE", f"{rmse:.4f}")
                st.metric("R²", f"{r2:.4f}")
            except Exception:
                st.info("Could not compute LGA metrics (maybe missing dropoff_rate).")
        else:
            lga_report['predicted_dropoff_rate'] = lga_report['dropoff_rate']
            lga_report['recommended_action'] = 'Monitor'
            st.info("No LGA model available; using observed dropoff_rate as predicted.")

        # Precompute nicely formatted fields for tooltips (fixes pydeck formatting issue)
        lga_report['pred'] = lga_report['predicted_dropoff_rate'].astype(float)
        lga_report['observed'] = lga_report.get('dropoff_rate', np.nan).astype(float)
        lga_report['pred_text'] = lga_report['pred'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "n/a")
        lga_report['obs_text'] = lga_report['observed'].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "n/a")
        lga_report['children_text'] = lga_report['total_children'].apply(lambda x: f"{int(x)}" if not pd.isna(x) else "0")
        lga_report['action_text'] = lga_report['recommended_action'].astype(str)
        try:
            lga_report['avg_pred'] = float(lga_report['pred'].mean())
        except Exception:
            lga_report['avg_pred'] = np.nan

        st.subheader("Top LGAs by predicted dropoff_rate")
        st.dataframe(lga_report.sort_values('predicted_dropoff_rate', ascending=False).head(50))
        download_link(lga_report, "lga_report_from_app.csv", "Download LGA report (CSV)")

        # Child-level predictions
        if child_model is not None and child_features is not None:
            st.subheader("Child-level predictions & metrics")
            X_child = pd.DataFrame()
            for f in child_features:
                if f in zd.columns:
                    X_child[f] = zd[f]
                else:
                    if f in ['Distance to HF', 'estimated_age_months'] or f.startswith('rate_') or f.startswith('lga_vacc_'):
                        X_child[f] = 0
                    else:
                        X_child[f] = 'missing'
            for c in X_child.columns:
                if c in ['Distance to HF','estimated_age_months'] or c.startswith('rate_') or c.startswith('lga_vacc_'):
                    X_child[c] = pd.to_numeric(X_child[c], errors='coerce').fillna(0).astype(float)
                else:
                    X_child[c] = X_child[c].fillna('missing').astype(str)
            try:
                if hasattr(child_model, 'predict_proba'):
                    probs = child_model.predict_proba(X_child)[:,1]
                else:
                    probs = child_model.predict(X_child)
                preds = child_model.predict(X_child)
                zd['pred_prob'] = probs
                zd['pred_class'] = preds
                try:
                    roc = roc_auc_score(zd['dropoff'].astype(int), probs)
                except Exception:
                    roc = None
                acc = accuracy_score(zd['dropoff'].astype(int), preds)
                prec = precision_score(zd['dropoff'].astype(int), preds, zero_division=0)
                rec = recall_score(zd['dropoff'].astype(int), preds, zero_division=0)
                f1 = f1_score(zd['dropoff'].astype(int), preds, zero_division=0)
                brier = brier_score_loss(zd['dropoff'].astype(int), probs)
                st.write({
                    "ROC AUC": roc,
                    "Accuracy": float(acc),
                    "Precision": float(prec),
                    "Recall": float(rec),
                    "F1": float(f1),
                    "Brier": float(brier)
                })
            except Exception as e:
                st.warning(f"Child model prediction failed: {e}")
        else:
            st.info("Child model not available. Child-level predictions not shown.")
            zd['pred_prob'] = zd.get('pred_prob', 0.0)
            zd['pred_class'] = zd.get('pred_class', 0)

        # -----------------------
        # Settlement matching & clusters (now grouped by LGA colours)
        # -----------------------
        st.subheader("Settlement matching & top-settlement priority")
        def lookup_settlement_coord(name: str) -> Tuple[Optional[float], Optional[float]]:
            if not isinstance(name, str) or name.strip() == "":
                return (np.nan, np.nan)
            k = name.strip().lower()
            if k in SETTLEMENT_COORDS:
                return SETTLEMENT_COORDS[k]
            k2 = ''.join(ch for ch in k if ch.isalnum() or ch.isspace()).strip()
            if k2 in SETTLEMENT_COORDS:
                return SETTLEMENT_COORDS[k2]
            for sk, coords in SETTLEMENT_COORDS.items():
                if sk.startswith(k2) or k2.startswith(sk):
                    return coords
            return (np.nan, np.nan)

        if 'Settlement' in zd.columns:
            zd['Settlement_norm'] = zd['Settlement'].astype(str).str.strip().str.lower()
            zd['settlement_lat'], zd['settlement_lon'] = zip(*zd['Settlement_norm'].apply(lambda s: lookup_settlement_coord(s)))
            settlement_agg = zd.groupby('Settlement').agg(
                LGA=('LGA','first'),
                total_children=('pred_class','count'),
                high_risk_children=('pred_prob', lambda s: (s >= 0.6).sum() if s.notna().any() else 0),
                avg_prob=('pred_prob','mean')
            ).reset_index().sort_values(['high_risk_children','avg_prob'], ascending=[False,False])
            # attach coords
            def attach_coords(r):
                lat, lon = lookup_settlement_coord(str(r['Settlement']))
                return pd.Series({'latitude': lat, 'longitude': lon})
            coords_df = settlement_agg.apply(attach_coords, axis=1)
            settlement_agg = pd.concat([settlement_agg, coords_df], axis=1)
            # compute distance to LGA centroid (if builtin centroid exists)
            lga_centroid_map = {k.strip().upper(): v for k, v in BUILTIN_LGA_COORDS.items()}
            def dist_to_centroid(row):
                lat = row['latitude']; lon = row['longitude']; lga = row['LGA']
                if pd.isna(lat) or pd.isna(lon):
                    return np.nan
                cent = lga_centroid_map.get(str(lga).strip().upper())
                if not cent:
                    return np.nan
                return haversine_km(lat, lon, cent['latitude'], cent['longitude'])
            settlement_agg['dist_to_lga_km'] = settlement_agg.apply(dist_to_centroid, axis=1)
            st.dataframe(settlement_agg.head(200))
            download_link(settlement_agg, "settlement_priority.csv", "Download settlement priority CSV")
            # clustering settlements by location + avg_prob to help visualization (kept for continuity)
            pts_for_cluster = settlement_agg.dropna(subset=['latitude','longitude']).copy()
            if not pts_for_cluster.empty:
                try:
                    cluster_features = pts_for_cluster[['latitude','longitude','avg_prob']].fillna(0).astype(float)
                    # keep original KMeans attempt if needed elsewhere (we won't rely on its clusters for colour)
                    if len(pts_for_cluster) >= n_clusters:
                        km = KMeans(n_clusters=n_clusters, random_state=42)
                        pts_for_cluster['kmeans_cluster'] = km.fit_predict(cluster_features)
                    else:
                        pts_for_cluster['kmeans_cluster'] = 0
                except Exception:
                    pts_for_cluster['kmeans_cluster'] = 0
            else:
                pts_for_cluster = pd.DataFrame()
        else:
            st.info("No 'Settlement' column in zerodose.csv — cannot compute per-settlement aggregates.")
            settlement_agg = pd.DataFrame()
            pts_for_cluster = pd.DataFrame()

        # -----------------------
        # LGA centroid map with labels (pydeck)
        # -----------------------
        st.subheader("LGA centroid priority map (builtin coords with labels)")
        lga_report['LGA_match'] = lga_report['LGA'].astype(str).str.strip().str.upper()
        builtin_rows = [{"LGA_match": k.strip().upper(), "latitude": v["latitude"], "longitude": v["longitude"], "LGA_builtin": k} for k, v in BUILTIN_LGA_COORDS.items()]
        lga_coords_df = pd.DataFrame(builtin_rows)
        merged = lga_report.merge(lga_coords_df[['LGA_match','latitude','longitude']], on='LGA_match', how='left')
        if 'predicted_dropoff_rate' not in merged.columns:
            merged['predicted_dropoff_rate'] = merged['dropoff_rate']
        def color_from_rate(rate: float):
            r = int(min(max(rate, 0.0), 1.0) * 255)
            g = int(min(max(1.0 - rate, 0.0), 1.0) * 200)
            b = 40
            return [r, g, b, 180]
        merged['map_color'] = merged['predicted_dropoff_rate'].apply(lambda x: color_from_rate(float(x)))
        max_children = merged['total_children'].replace(0, np.nan).max() or 1
        merged['map_radius'] = merged['total_children'].apply(lambda x: (float(x) / max_children) * 400 + 60)
        pts = merged.dropna(subset=['latitude','longitude']).copy()
        if not pts.empty:
            centroid_layer = pdk.Layer(
                "ScatterplotLayer",
                data=pts,
                get_position=["longitude","latitude"],
                get_radius="map_radius",
                radius_min_pixels=8,
                radius_max_pixels=400,
                get_fill_color="map_color",
                get_line_color=[0,0,0],
                pickable=True
            )
            deck_layers = [centroid_layer]
            if show_lga_labels:
                # TextLayer for LGA labels
                text_layer = pdk.Layer(
                    "TextLayer",
                    data=pts,
                    pickable=False,
                    get_position=["longitude","latitude"],
                    get_text="LGA",
                    get_color=[255, 255, 255],
                    get_size=16,
                    get_angle=0,
                    get_text_anchor="'middle'",
                    get_alignment_baseline="'bottom'",
                    get_offset=[0, -30]
                )
                deck_layers.append(text_layer)
            view = pdk.ViewState(latitude=float(pts['latitude'].median()), longitude=float(pts['longitude'].median()), zoom=7, pitch=0)
            tooltip = {"html": "<b>{LGA}</b><br/>Pred: {pred_text}<br/>Observed: {obs_text}<br/>Children: {children_text}<br/>Action: {action_text}", "style": {"color": "white"}}
            st.pydeck_chart(pdk.Deck(layers=deck_layers, initial_view_state=view, tooltip=tooltip))
            st.success("LGA centroid map rendered.")
            st.dataframe(pts[['LGA','predicted_dropoff_rate','dropoff_rate','total_children','latitude','longitude']].sort_values('predicted_dropoff_rate', ascending=False).head(20))
        else:
            st.info("No builtin centroid matches found — ensure your LGA names match the builtins or provide coordinates.")

        # -----------------------
        # Settlement cluster map (pydeck) with distinct colours per LGA (3 colours)
        # -----------------------
        st.subheader("Settlement cluster map (grouped by LGA colours)")
        if not pts_for_cluster.empty:
            sp = pts_for_cluster.copy()
            # group by LGA and map to 3 colours
            sp['LGA_norm'] = sp['LGA'].astype(str).str.strip()
            unique_lgas = list(sp['LGA_norm'].unique())
            # assign one of 3 colours cyclically
            lga_to_color = {lga: SETTLEMENT_COLORS[i % len(SETTLEMENT_COLORS)] for i, lga in enumerate(unique_lgas)}
            sp['map_color'] = sp['LGA_norm'].map(lga_to_color)
            # radius scaled by high-risk children
            max_hr = sp['high_risk_children'].replace(0, np.nan).max() or 1
            sp['map_radius'] = sp['high_risk_children'].apply(lambda x: (float(x) / max_hr) * 300 + 20)
            # prepare layers: one ScatterplotLayer per LGA to guarantee colour separation
            cluster_layers = []
            text_layers = []
            center_lat = float(sp['latitude'].median())
            center_lon = float(sp['longitude'].median())
            for lga in sorted(sp['LGA_norm'].unique()):
                part = sp[sp['LGA_norm'] == lga].copy()
                if part.empty:
                    continue
                color = lga_to_color.get(lga, SETTLEMENT_COLORS[0])
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=part,
                    get_position=["longitude","latitude"],
                    get_radius="map_radius",
                    radius_min_pixels=6,
                    radius_max_pixels=200,
                    get_fill_color=color,
                    get_line_color=[0,0,0],
                    pickable=True,
                    auto_highlight=True
                )
                cluster_layers.append(layer)
                if show_settlement_labels:
                    tlay = pdk.Layer(
                        "TextLayer",
                        data=part,
                        pickable=False,
                        get_position=["longitude","latitude"],
                        get_text="Settlement",
                        get_color=[255,255,255],
                        get_size=12,
                        get_angle=0,
                        get_text_anchor="'middle'",
                        get_alignment_baseline="'bottom'",
                        get_offset=[0, -12]
                    )
                    text_layers.append(tlay)
            all_layers = cluster_layers + text_layers
            view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8, pitch=0)
            tooltip = {"html":"<b>{Settlement}</b><br/>LGA: {LGA}<br/>High-risk children: {high_risk_children}<br/>Avg prob: {avg_prob:.2f}", "style":{"color":"white"}}
            # Fix tooltip: use avg_prob_text instead of formatting inside the string
            # so create avg_prob_text field
            sp['avg_prob_text'] = sp['avg_prob'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "n/a")
            tooltip = {"html":"<b>{Settlement}</b><br/>LGA: {LGA}<br/>High-risk children: {high_risk_children}<br/>Avg prob: {avg_prob_text}", "style":{"color":"white"}}
            st.pydeck_chart(pdk.Deck(layers=all_layers, initial_view_state=view, tooltip=tooltip))
            st.success("Rendered settlement cluster map grouped by LGA colours.")
            # add legend
            legend_html = "<div style='display:flex;flex-wrap:wrap;padding:6px;'>"
            for lga, col in lga_to_color.items():
                hexc = rgba_to_hex(col)
                legend_html += f"<div style='margin-right:10px;margin-bottom:6px;display:flex;align-items:center'><div style='width:18px;height:18px;background:{hexc};border-radius:4px;margin-right:6px;'></div><div style='font-size:13px;color:#fff'>{lga}</div></div>"
            legend_html += "</div>"
            st.markdown("**Legend: Settlement colour → LGA**")
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.info("No settlement coordinates matched from the internal list or not enough data for clustering.")

        # -----------------------
        # Prioritization & downloads
        # -----------------------
        st.subheader("Prioritization & downloads")
        risk_thresh = st.slider("LGA predicted dropoff rate threshold", 0.0, 1.0, 0.20, 0.01)
        settlement_prob_thresh = st.slider("Settlement avg_prob threshold", 0.0, 1.0, 0.6, 0.01)
        high_lgas = lga_report[lga_report['predicted_dropoff_rate'] >= risk_thresh].sort_values('predicted_dropoff_rate', ascending=False)
        st.markdown(f"**LGAs above threshold ({len(high_lgas)}):**")
        st.dataframe(high_lgas[['LGA','total_children','dropoff_rate','predicted_dropoff_rate','recommended_action']].head(200))
        download_link(high_lgas, "high_risk_lgas.csv", "Download high-risk LGAs CSV")
        if 'settlement_agg' in locals() and not settlement_agg.empty:
            ssum = settlement_agg.copy()
            ssum['high_risk_flag'] = ssum['avg_prob'] >= settlement_prob_thresh
            st.markdown(f"**Settlements with high-risk children (avg_prob >= {settlement_prob_thresh:.2f})**")
            st.dataframe(ssum.sort_values(['high_risk_children','avg_prob'], ascending=[False,False]).head(200))
            download_link(ssum, "high_risk_settlements.csv", "Download high-risk settlements CSV")

        # Save outputs to artifacts if user wants
        if st.button("Save current LGA report & settlement outputs to ./artifacts/"):
            lga_report.to_csv(ARTIFACTS / "lga_report_from_app.csv", index=False)
            if 'settlement_agg' in locals():
                settlement_agg.to_csv(ARTIFACTS / "settlement_priority_from_app.csv", index=False)
            if 'pred_prob' in zd.columns:
                zd.to_csv(ARTIFACTS / "child_predictions_from_app.csv", index=False)
            st.success("Saved outputs to ./artifacts/")

        if st.button("Download available artifact files (zip)"):
            to_zip = [p for p in [LGA_MODEL_FILE, LGA_FEATURES_FILE, LGA_MODEL_IMP, LGA_FEATURES_IMP, LGA_REPORT_IMP, CHILD_MODEL_FILE, CHILD_FEATURES_FILE] if p.exists()]
            if to_zip:
                save_zip_of(to_zip, "available_artifacts.zip")
            else:
                st.warning("No artifacts found to zip.")
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        logger.exception(e)

st.markdown("---")
st.markdown("""
**Notes & tips**
- Labels can clutter maps; use the sidebar toggles (`Show LGA labels`, `Show settlement labels`) to enable/disable.
- Settlement points are now coloured by their LGA (first 3 palette colours). If you want every LGA to have a unique colour, tell me and I'll expand the palette.
- To further improve distinct shapes, provide a small set of icon URLs and use `IconLayer` for cluster-specific icons.
- For model accuracy improvements: clean and enrich features (temporal vaccine trends, facility accessibility), and consider hierarchical shrinkage for small LGAs.
""")
