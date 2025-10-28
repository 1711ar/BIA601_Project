# webapp_streamlit.py — UI Skeleton (Upload + Preview)
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GA Feature Selection", page_icon="🧬", layout="wide")
st.title("GA Feature Selection")
st.caption("Select fewer features. Prove it with cross-validation.")

with st.sidebar:
    st.header("Steps")
    st.markdown("1) Upload data\n2) Choose target (y)\n3) Run GA (next)\n4) Compare baselines (next)")

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
    # CSV heuristics
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin1"):
        for sep in (None, ",", ";", "\t", "|"):
            try:
                df_try = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                if df_try.shape[1] > 1:
                    return df_try
            except Exception:
                continue
    # fallback: try excel one more time
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
st.success(f"Loaded: **{file.name}** · Shape: **({rows}, {cols})**")
st.dataframe(df.head(), use_container_width=True)

# Pick numeric target
num_cols = list(df.select_dtypes(include=["int64","float64","int32","float32"]).columns)
if not num_cols:
    st.error("No numeric columns found. Please upload a dataset with a numeric target.")
    st.stop()

default_idx = num_cols.index("Strength") if "Strength" in num_cols else 0
target_col = st.selectbox("Target column (y)", options=num_cols, index=default_idx)
st.caption(f"Target is **{target_col}**. GA & baselines come in next PRs.")
