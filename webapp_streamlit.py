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

# Upload & Preview
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

# ===== Run Genetic Algorithm =====
st.divider()
run = st.button("Run GA", type="primary")

if run:
    import io, json
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from ga_feature_select import GeneticFeatureSelector

    # X = numeric features (excluding target), y = target
    X_full = df.drop(columns=[target_col]).select_dtypes(include=["number"])
    y = df[target_col].values

    if X_full.shape[1] < 2:
        st.error("Need at least 2 numeric features besides the target.")
        st.stop()

    selector = GeneticFeatureSelector(
        estimator=Ridge(),
        population=int(population),
        generations=int(generations),
        crossover_rate=float(crossover),
        mutation_rate=float(mutation),
        lam=float(lam),
        cv=int(cv_folds),
        random_state=int(seed),
    )
    selector.fit(X_full.values, y)

    mask = selector.get_support()
    selected_cols = list(X_full.columns[mask])

    kf = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))
    est = Ridge()
    X_sel = X_full[selected_cols].values if selected_cols else X_full.values
    y_pred = cross_val_predict(est, X_sel, y, cv=kf)

    tab_results, tab_selected, tab_baselines, tab_logs = st.tabs(
        ["Results", "Selected", "Baselines", "Logs"]
    )

    with tab_results:
        c1, c2, c3 = st.columns(3)
        c1.metric("GA (raw CV)", f"{selector.best_score_raw_:.4f}")
        c2.metric("GA (penalized)", f"{selector.best_score_pen_:.4f}")
        c3.metric("Selected", f"{mask.sum()}/{X_full.shape[1]}")

        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.6)
        mn, mx = float(np.min(y)), float(np.max(y))
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs True (GA subset)")
        st.pyplot(fig)

        # Download: predictions.csv
        import pandas as pd
        pred_df = pd.DataFrame({"y_true": y, "y_pred": y_pred})
        buf = io.StringIO(); pred_df.to_csv(buf, index=False)
        st.download_button("Download predictions.csv", buf.getvalue().encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

    with tab_selected:
        st.subheader("Selected features")
        if selected_cols:
            st.write(", ".join(selected_cols))
            # Download: selected_features.csv
            sel_df = pd.DataFrame({"feature": selected_cols})
            buf2 = io.StringIO(); sel_df.to_csv(buf2, index=False)
            st.download_button("Download selected_features.csv", buf2.getvalue().encode("utf-8"),
                               file_name="selected_features.csv", mime="text/csv")
        else:
            st.info("No features selected")

        # Download: boolean mask (JSON)
        mask_json = json.dumps({col: bool(m) for col, m in zip(X_full.columns, mask.tolist())}, ensure_ascii=False)
        st.download_button("Download feature_mask.json", mask_json.encode("utf-8"),
                           file_name="feature_mask.json", mime="application/json")

    with tab_baselines:
        st.info("Baselines (BASE / PCA / SelectKBest) will be added in a later PR.")

    with tab_logs:
        hist = getattr(selector, "history_", None)
        if hist:
            log_df = pd.DataFrame([{"gen": h.gen, "best_raw": h.best_raw,
                                    "best_pen": h.best_pen, "k": h.k} for h in hist])
            st.dataframe(log_df, use_container_width=True)
            buf3 = io.StringIO(); log_df.to_csv(buf3, index=False)
            st.download_button("Download ga_history.csv", buf3.getvalue().encode("utf-8"),
                               file_name="ga_history.csv", mime="text/csv")
        else:
            st.info("No GA history available.")
else:
    st.info("Set GA parameters (left), choose target, then click Run GA.")
