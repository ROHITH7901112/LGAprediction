#!/usr/bin/env python3
"""
streamlit_child_predictor.py

Child risk predictor Streamlit app — robust dtype handling with binary flags enforced.

Run:
    streamlit run streamlit_child_predictor.py
"""
from pathlib import Path
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ----------------------
# Config & constants
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE = Path.cwd()
ARTIFACTS = BASE / "artifacts"
CHILD_MODEL_FILE = ARTIFACTS / "child_dropoff_model.joblib"
CHILD_FEATURES_FILE = ARTIFACTS / "child_features.joblib"

# Reason-for-ZD flags used in your training
REASON_FLAGS = [
    'busy_caregiver', 'family_problems', 'family_problems_busy_caregiver',
    'family_problems_missed_appointment', 'family_problems_transportation',
    'fear_of_AE', 'financial_resources', 'financial_resources_missed_appointment_busy_caregiver',
    'hp_poor_attitude', 'low_trust', 'missed_appointment_hp_poor_attitude',
    'missed_appointment_uncooperative_husband', 'no_need_felt', 'no_need_felt_busy_caregiver',
    'no_need_felt_financial_resources', 'no_need_felt_low_trust', 'no_permissions',
    'reported_AE', 'sick_child', 'transportation', 'uncooperative_husband',
    'uncooperative_husband_low_trust'
]

st.set_page_config(page_title="Child Risk Predictor", layout="centered")

# ----------------------
# Helpers
# ----------------------
def download_link_df(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def safe_numeric(val, default=0.0):
    try:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            return default
        return float(val)
    except Exception:
        return default

def safe_categorical(val):
    if val is None:
        return "missing"
    if isinstance(val, float) and np.isnan(val):
        return "missing"
    return str(val)

def inspect_pipeline_feature_groups(pipeline) -> Tuple[List[str], List[str], List[str]]:
    """
    Inspect a sklearn Pipeline that has a 'pre' ColumnTransformer step.
    Returns (numeric_cols, categorical_cols, passthrough_cols).
    """
    numeric_cols, categorical_cols, passthrough_cols = [], [], []
    try:
        pre = pipeline.named_steps.get('pre', None)
        if pre is None and hasattr(pipeline, 'steps'):
            first = pipeline.steps[0][1]
            pre = first if hasattr(first, 'transformers_') else None
        if pre is None:
            return [], [], []
        for name, transformer, cols in pre.transformers_:
            if transformer is None or transformer == 'passthrough':
                if isinstance(cols, (list, tuple)):
                    passthrough_cols.extend(list(cols))
            else:
                # try to infer by inner pipeline steps
                t = transformer
                if hasattr(t, 'named_steps'):
                    names = " ".join(t.named_steps.keys()).lower()
                    if 'ohe' in names or 'onehot' in names:
                        if isinstance(cols, (list, tuple)):
                            categorical_cols.extend(list(cols))
                    else:
                        if isinstance(cols, (list, tuple)):
                            numeric_cols.extend(list(cols))
                else:
                    # fallback
                    if isinstance(cols, (list, tuple)):
                        numeric_cols.extend(list(cols))
        # dedupe
        return sorted(set(numeric_cols)), sorted(set(categorical_cols)), sorted(set(passthrough_cols))
    except Exception as e:
        logger.warning(f"Pipeline inspection failed: {e}")
        return [], [], []

@st.cache_resource
def load_artifacts():
    res = {}
    if CHILD_MODEL_FILE.exists() and CHILD_FEATURES_FILE.exists():
        res['model'] = joblib.load(CHILD_MODEL_FILE)
        res['features'] = joblib.load(CHILD_FEATURES_FILE)
    else:
        missing = []
        if not CHILD_MODEL_FILE.exists(): missing.append(str(CHILD_MODEL_FILE))
        if not CHILD_FEATURES_FILE.exists(): missing.append(str(CHILD_FEATURES_FILE))
        st.warning(f"Missing artifact files: {', '.join(missing)}. Place them in ./artifacts/")
    return res

@st.cache_data
def load_local_uniques(max_unique=200) -> Dict[str, List[str]]:
    path = BASE / "zerodose.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        uniques = {}
        for c in df.columns:
            vals = df[c].dropna().unique().tolist()
            if 1 <= len(vals) <= max_unique:
                uniques[c] = sorted([str(v) for v in vals])
        return uniques
    except Exception:
        return {}

# ----------------------
# Load model + introspect
# ----------------------
local_uniques = load_local_uniques()
art = load_artifacts()
model = art.get('model')
child_features = art.get('features')

if model is None or child_features is None:
    st.title("🔎 Individual Child Risk Predictor")
    st.error("Child model or feature list not found in ./artifacts/. Place `child_dropoff_model.joblib` and `child_features.joblib` there.")
    st.stop()

pipe_numeric, pipe_categorical, pipe_passthrough = inspect_pipeline_feature_groups(model)

# Build sets for numeric/categorical
numeric_features = set(pipe_numeric)
# passthrough that should be numeric
for c in pipe_passthrough:
    if c.startswith('lga_vacc_') or c in REASON_FLAGS or c.startswith('rate_') or c in ['Distance to HF', 'estimated_age_months']:
        numeric_features.add(c)
# enforce known numeric columns even if pipeline inspection missed them
for f in child_features:
    if f in ['Distance to HF', 'estimated_age_months'] or f.startswith('lga_vacc_') or f.startswith('rate_') or f in REASON_FLAGS:
        numeric_features.add(f)

numeric_features = sorted(numeric_features)

st.title("🔎 Individual Child Risk Predictor (binary flags enforced)")
st.markdown("""
- All **reason flags** and **lga_vacc_*** are treated as binary (0/1).
- All numeric fields are coerced to float before prediction.
- Categorical fields use dropdowns.
""")

# Optional CSV to populate choices
st.info("Optional: upload a CSV to populate dropdown choices for categorical fields.")
uploaded_choices = st.file_uploader("Optional: upload CSV for dropdown choices", type=["csv"], key="choices_uploader")
batch_choices = {}
if uploaded_choices is not None:
    try:
        tmp_df = pd.read_csv(uploaded_choices, dtype=str, low_memory=False)
        for c in tmp_df.columns:
            vals = tmp_df[c].dropna().unique().tolist()
            if 1 <= len(vals) <= 500:
                batch_choices[c] = sorted([str(v) for v in vals])
        st.success("Loaded choice values from uploaded CSV.")
    except Exception as e:
        st.warning(f"Could not read uploaded CSV for choices: {e}")

mode = st.radio("Mode", ["Single prediction (form)", "Batch prediction (CSV)"])

def prepare_single_from_inputs(single_inputs: Dict[str, Any]) -> pd.DataFrame:
    row = {}
    for f in child_features:
        if f in single_inputs:
            row[f] = single_inputs[f]
        else:
            row[f] = 0.0 if f in numeric_features else "missing"
    X = pd.DataFrame([row])
    # numeric coercion
    for c in numeric_features:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0).astype(float)
    # categorical to string
    for c in X.columns:
        if c not in numeric_features:
            X[c] = X[c].fillna('missing').astype(str)
    return X

def prepare_batch_from_df(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for f in child_features:
        if f in df.columns:
            X[f] = df[f]
        else:
            X[f] = 0.0 if f in numeric_features else "missing"
    for c in numeric_features:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0).astype(float)
    for c in X.columns:
        if c not in numeric_features:
            X[c] = X[c].fillna('missing').astype(str)
    return X

# ----------------------
# UI
# ----------------------
if mode == "Single prediction (form)":
    st.subheader("Single child prediction")
    with st.form("single_form"):
        single_inputs: Dict[str, Any] = {}
        for f in child_features:
            # Handle binary flags first
            if f in REASON_FLAGS or f.startswith('lga_vacc_'):
                # show checkbox (True/False) mapped to 1/0
                val = st.checkbox(f"{f}", value=False)
                single_inputs[f] = 1 if val else 0
            elif f in numeric_features:
                # other numeric fields
                single_inputs[f] = st.number_input(f"{f} (numeric)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
            else:
                # categorical: dropdown options priority - batch choices, local uniques, defaults
                choices: Optional[List[str]] = None
                if f in batch_choices:
                    choices = ["missing"] + batch_choices[f]
                elif f in local_uniques:
                    choices = ["missing"] + local_uniques[f]
                else:
                    if f.lower() == 'gender':
                        choices = ["missing", "Male", "Female", "Other"]
                    elif f.lower() in ['woman or child', 'woman_or_child', 'womanorchild']:
                        choices = ["missing", "Woman", "Child"]
                    else:
                        choices = ["missing"]
                single_inputs[f] = st.selectbox(f"{f} (categorical)", options=choices, index=0)
        submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            X_single = prepare_single_from_inputs(single_inputs)
            # debug view
            st.caption("Prepared one-row input (dtypes):")
            st.write({k: str(v) for k, v in X_single.dtypes.to_dict().items()})
            # Predict
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_single)[:,1][0])
            else:
                prob = None
            pred = int(model.predict(X_single)[0])
            st.markdown("### Prediction")
            if prob is not None:
                st.metric("Dropoff probability", f"{prob:.3f}")
            st.write("Predicted class:", "DROPOFF (likely)" if pred == 1 else "LOW DROPOFF RISK")
            if prob is not None:
                if prob >= 0.99:
                    st.warning("High risk — recommend immediate follow-up/outreach.")
                elif prob >= 0.6:
                    st.info("Moderate risk — consider targeted outreach or reminder.")
                else:
                    st.success("Low risk — routine monitoring.")
            out = X_single.copy()
            out['pred_prob'] = prob
            out['pred_class'] = pred
            download_link_df(out, "single_child_prediction.csv", "Download prediction (CSV)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            logger.exception(e)

else:
    st.subheader("Batch prediction (CSV)")
    st.markdown("Upload a CSV with child records. All binary flags (reason & lga_vacc_*) should be 0/1.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="batch_uploader")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, dtype=str, low_memory=False)
            # Convert known binary columns to 0/1 if strings like 'missing' or blanks appear
            for col in df.columns:
                if col in REASON_FLAGS or col.startswith('lga_vacc_'):
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            st.write("Sample uploaded data:")
            st.dataframe(df.head(5))
            X_batch = prepare_batch_from_df(df)
            st.write("Prepared features (sample):")
            st.dataframe(X_batch.head(5))
            if st.button("Run batch prediction"):
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch)[:,1]
                else:
                    probs = [None]*len(X_batch)
                preds = model.predict(X_batch)
                results = df.copy()
                results['pred_prob'] = probs
                results['pred_class'] = preds
                st.subheader("Batch predictions (sample):")
                st.dataframe(results.head(10))
                download_link_df(results, "batch_child_predictions.csv", "Download batch predictions CSV")
        except Exception as e:
            st.error(f"Failed to run batch prediction: {e}")
            logger.exception(e)

st.markdown("---")
st.markdown("""
**Important:**  
- All *reason flags* and `lga_vacc_*` fields are binary (0 or 1).  
- If you paste rows like the one you showed earlier, replace `'missing'` with `0` for those fields.
""")
