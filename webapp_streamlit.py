import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GA Feature Selection", page_icon="🧬", layout="wide")
st.title("GA Feature Selection")
st.caption("Select fewer features. Prove it with cross-validation.")

# ---- GA params ----
with st.sidebar:
    st.header("GA Parameters")
    task = st.selectbox("Task", ["Regression"], index=0)
    regressor = st.selectbox("Regressor", ["Ridge"], index=0)
    cv_folds = st.number_input("CV folds", min_value=3, max_value=10, value=5, step=1)
    population = st.number_input("Population", min_value=10, max_value=200, value=60, step=10)
    generations = st.number_input("Generations", min_value=5, max_value=100, value=14, step=1)
    crossover = st.slider("Crossover rate", 0.0, 1.0, 0.8, 0.05)
    mutation = st.slider("Mutation rate", 0.0, 1.0, 0.08, 0.01)
    lam = st.slider("λ penalty (features)", 0.0, 0.5, 0.10, 0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    st.caption("Run button will be enabled in the next PR.")

# ---- robust reader (CSV/XLSX) ----
def safe_read_uploaded(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    # Try Excel first
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception:
            pass
    for enc in ("utf-8","utf-8-sig","cp1256","latin1"):
        for sep in (None,",",";","\t","|"):
            try:
                df_try = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                if df_try.shape[1] > 1:
                    return df_try
            except Exception:
                continue
    return pd.read_excel(io.BytesIO(raw))

# ---- Upload & Preview ----
st.subheader("Upload & Preview")
file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
if file is None:
    st.info("Drop a CSV/XLSX file here to begin.")
    st.stop()

try:
    df = safe_read_uploaded(file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

rows, cols = df.shape
st.success(f"Loaded: {file.name} · Shape: ({rows}, {cols})")
st.dataframe(df.head(), use_container_width=True)

num_cols = list(df.select_dtypes(include=["int64","float64","int32","float32"]).columns)
if not num_cols:
    st.error("No numeric columns found. Please upload a dataset with a numeric target.")
    st.stop()

default_idx = num_cols.index("Strength") if "Strength" in num_cols else 0
target_col = st.selectbox("Target column (y)", options=num_cols, index=default_idx)

st.divider()
st.write("Run GA button will be added in the next PR. Baselines tab comes later.")