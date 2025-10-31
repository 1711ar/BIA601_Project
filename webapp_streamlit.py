from __future__ import annotations
import io, platform
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from ga_feature_select import GeneticFeatureSelector
try:
    from backend_db import get_logs, log_run, save_uploaded_file
except Exception:
    get_logs = log_run = save_uploaded_file = None

#  Page setup & Brand 
st.set_page_config(page_title="GA Feature Selection", page_icon="🧬", layout="wide")


st.markdown("""
<style>
header[data-testid="stHeader"]{display:none !important;}
div[data-testid="stToolbar"]{display:none !important;}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
.viewerBadge_container__1QSob{display:none !important;}
.block-container{padding-top:.4rem;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
:root{
  --g1:#0DD4C6;   /* teal */
  --g2:#10B981;   /* emerald */
  --g3:#2563EB;   /* sapphire */
  --text:#0F172A; --muted:#64748B;
  --surface:#FFFFFF; --surface-2:#F8FAFC; --border:#E2E8F0;
  --radius:14px;
}

/* Background with subtle algorithmic network */
[data-testid="stAppViewContainer"]{
  background: linear-gradient(180deg, rgba(13,212,198,0.03), rgba(37,99,235,0.03)),
              radial-gradient(circle at 25px 25px, rgba(16,185,129,0.10) 2px, transparent 2.2px),
              radial-gradient(circle at 75px 75px, rgba(13,212,198,0.08) 1.8px, transparent 2px),
              var(--surface-2);
  background-size: 100% 100%, 100px 100px, 140px 140px, auto;
  color: var(--text);
}

/* Hero */
.hero{
  background: linear-gradient(120deg, var(--g1), var(--g2), var(--g3));
  color:#fff; padding:18px 20px; border-radius:var(--radius);
  box-shadow:0 10px 32px rgba(37,99,235,.20); margin:6px 0 16px;
}
.hero h1{margin:0; font-size:1.35rem; line-height:1.2}
.hero p{margin:.3rem 0 0; opacity:.95}

/* Layout helpers */
.section{
  background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:16px; margin:.6rem 0;
}
.left-sticky{ position: sticky; top: 12px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:.35rem }
.stTabs [data-baseweb="tab"]{
  background:#F0FDFA; border:1px solid #CCFBF1; color:#064E3B; border-radius:999px;
  padding:.45rem .9rem; font-weight:700;
}
.stTabs [aria-selected="true"]{
  background:#fff; box-shadow:0 1px 10px rgba(6,95,70,.12);
}

/* Buttons */
.stButton>button{
  background: linear-gradient(90deg, var(--g1), var(--g2), var(--g3));
  color:#fff; border:0; border-radius:999px; padding:.55rem 1.1rem; font-weight:800;
  box-shadow:0 6px 16px rgba(16,185,129,.25);
}
.stButton>button:hover{ filter:brightness(1.05) }
.btn-ghost>button{
  background:#FFFFFF !important; color:#0F172A !important; border:1px solid var(--border) !important;
  box-shadow:none !important; font-weight:700 !important;
}

/* Sticky action bar (bottom of left panel) */
.action-dock{
  position: sticky; bottom:0; z-index:5;
  background:rgba(255,255,255,.96);
  border-top:1px solid var(--border);
  padding:10px; margin-top:8px; backdrop-filter:saturate(120%) blur(6px);
  border-bottom-left-radius:var(--radius); border-bottom-right-radius:var(--radius);
}

/* Metric cards with gradient top border */
.metric-card{
  background:#FFFFFF; border:1px solid var(--border); border-radius:var(--radius);
  padding:14px 16px; box-shadow:0 1px 8px rgba(0,0,0,.03);
  border-top:4px solid; border-image: linear-gradient(90deg, var(--g1), var(--g2), var(--g3)) 1;
}
.metric-card .small{ color:var(--muted); font-size:.9rem }

/* Inputs */
[data-baseweb="input"]>div, .stNumberInput, .stSelectbox, .stSlider{ --border-color: var(--border); }

/* Dataframe tweaks */
.block-container{padding-top:.6rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>GA-Based Feature Selection</h1>
  <p>    Simple, transparent feature selection using a genetic algorithm</p>
</div>
""",
    unsafe_allow_html=True,
)

# Keep 1 on Windows to avoid joblib deadlocks inside Streamlit
N_JOBS = 1 if platform.system() == "Windows" else -1

#  Helpers
def safe_read_uploaded(uploaded_file) -> pd.DataFrame:
    """Robust reader: Excel first, then CSV with multiple encodings and separators."""
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(raw))
        except Exception:
            pass
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin1"):
        for sep in (None, ",", ";", "\t", "|"):
            try:
                df_try = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                if df_try.shape[1] > 1:
                    return df_try
            except Exception:
                continue
    # fallback to Excel parser if CSV heuristics failed
    return pd.read_excel(io.BytesIO(raw))

def infer_task(y_series: pd.Series, override: str) -> str:
    if override == "Classification":
        return "clf"
    if override == "Regression":
        return "reg"
    is_num = pd.api.types.is_numeric_dtype(y_series)
    if not is_num:
        return "clf"
    uniq = np.unique(y_series.values)
    return "clf" if len(uniq) <= max(20, int(0.05 * len(y_series))) else "reg"

def reset_app():
    st.session_state.clear()
    st.rerun()

#  Layout
left, right = st.columns([0.36, 0.64], gap="large")

#  Left panel (sticky controls)
with left:
    st.markdown('<div class="section left-sticky">', unsafe_allow_html=True)
    st.subheader("Controls", help="Configure dataset, model, and genetic algorithm parameters.")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    df = None
    stored_path = None
    filename = None
    if uploaded is not None:
        df = safe_read_uploaded(uploaded)
        filename = uploaded.name
        if save_uploaded_file is not None:
            try:
                stored_path = save_uploaded_file(uploaded)
            except Exception as e:
                st.caption(f"(Could not save uploaded file to DB store: {e})")
        st.caption(f"Loaded: **{filename}** · Shape: {None if df is None else df.shape}")

    target_col = st.selectbox("Target column (y)", options=(list(df.columns) if df is not None else []))
    st.markdown("---")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        task_pick = st.selectbox("Task", ["Auto", "Classification", "Regression"])
        model_name_clf = st.selectbox("Classifier", ["LogisticRegression", "RandomForest"])
        model_name_reg = st.selectbox("Regressor", ["Ridge", "RandomForestRegressor"])
    with col_b:
        generations    = st.number_input("Generations", 5, 200, 14, 1)
        pop_size       = st.number_input("Population size", 10, 500, 60, 5)
        cv_splits      = st.number_input("CV folds", 2, 10, 3, 1)
        fast_rows      = st.number_input("Fast mode: first N rows (0=all)", 0, 2_000_000, 3000, 500)

    with st.expander("Advanced GA settings", expanded=False):
        lambda_penalty = st.slider("Lambda penalty", 0.0, 2.0, 0.20, 0.05,
                                   help="Penalize larger subsets to prefer parsimony.")
        crossover_prob = st.slider("Crossover probability", 0.0, 1.0, 0.80, 0.05)
        mutation_prob  = st.slider("Mutation probability", 0.0, 1.0, 0.08, 0.01)
        random_state   = st.number_input("Random state", 0, 10000, 42, 1)
        high_card_cutoff = st.number_input("High-cardinality cutoff (drop if unique >)", 10, 5000, 50, 10)
        run_baselines = st.toggle("Run baselines (BASE / PCA / SKB)", value=False)

    st.markdown('<div class="action-dock">', unsafe_allow_html=True)
    c_run, c_reset = st.columns([1, 1])
    run_click = c_run.button("▶️ Run GA", use_container_width=True)
    reset_click = c_reset.container().button("Reset", use_container_width=True, key="reset_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Right panel 
with right:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Preview")
    if df is None:
        st.info("Upload a CSV/XLSX to continue.", icon="📥")
    else:
        st.dataframe(df.head(12), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Run if requested
    if reset_click:
        reset_app()

    if run_click:
        if df is None or not target_col:
            st.error("Please upload a dataset and select the target column to run GA.", icon="⚠️")
        else:
            # Prepare X, y
            y = df[target_col].copy()
            X = df.drop(columns=[target_col]).copy()

            if fast_rows and len(X) > fast_rows:
                X = X.head(fast_rows)
                y = y.iloc[:fast_rows]
                st.info(f"Fast mode ON → using first {len(X)} rows out of {len(df)}", icon="⏩")

            # drop high-cardinality categoricals
            high_card_cols = [c for c in X.columns if X[c].dtype == "object"
                              and X[c].nunique(dropna=True) > int(high_card_cutoff)]
            if high_card_cols:
                st.caption("Removed high-cardinality columns:")
                st.write(", ".join(high_card_cols))
                X = X.drop(columns=high_card_cols)

            # one-hot + fill missing
            X = pd.get_dummies(X, drop_first=True)
            X = X.fillna(X.median(numeric_only=True))

            task = infer_task(y, task_pick)
            label_names = None
            if task == "clf":
                if not pd.api.types.is_numeric_dtype(y):
                    _le = LabelEncoder()
                    y = _le.fit_transform(y.astype(str))
                    label_names = _le.classes_
                else:
                    label_names = np.unique(y)
                y = np.asarray(y).astype(int)
                base_est = (
                    LogisticRegression(max_iter=2000, solver="liblinear")
                    if model_name_clf == "LogisticRegression"
                    else RandomForestClassifier(n_estimators=200, random_state=random_state)
                )
                score_name = "accuracy"
                # safe n_splits
                _, counts = np.unique(y, return_counts=True)
                n_splits = int(min(cv_splits, counts.min() if counts.size else cv_splits))
                n_splits = max(n_splits, 2)
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                score_func = mutual_info_classif
            else:
                y = np.asarray(y).astype(float)
                base_est = Ridge(alpha=1.0) if model_name_reg == "Ridge" else RandomForestRegressor(n_estimators=300, random_state=random_state)
                score_name = "r2"
                cv = KFold(n_splits=int(cv_splits), shuffle=True, random_state=random_state)
                score_func = f_regression

            scaler = StandardScaler(with_mean=False)

            # GA
            ga_pipeline = Pipeline([("scaler", scaler), ("est", base_est)])
            ga = GeneticFeatureSelector(
                estimator=ga_pipeline,
                generations=int(generations),
                pop_size=int(pop_size),
                crossover_prob=float(crossover_prob),
                mutation_prob=float(mutation_prob),
                lambda_penalty=float(lambda_penalty),
                cv_splits=int(cv_splits),
                early_stopping_rounds=8, 
                random_state=int(random_state),
                task=task,
                n_jobs=(1 if platform.system() == "Windows" else -1),
            )

            with st.spinner("Running Genetic Algorithm…"):
                ga.fit(X.values, y)
                mask = ga.best_mask_
                if mask is None or mask.sum() == 0:
                    st.warning("GA selected 0 features — falling back to all features.", icon="🧩")
                    mask = np.ones(X.shape[1], dtype=bool)
                ga_raw_pipe = Pipeline([("scaler", scaler), ("est", base_est)])
                ga_raw_score = cross_val_score(
                    ga_raw_pipe, X.values[:, mask], y, cv=cv, scoring=score_name, n_jobs=N_JOBS
                ).mean()

            selected = list(X.columns[mask])
            df_sel = pd.concat([X.loc[:, selected], pd.Series(y, name=target_col)], axis=1)

            # Baselines
            acc_base = acc_pca = acc_skb = None
            best_pca_k = best_skb_k = None
            if run_baselines:
                scaler_b = StandardScaler(with_mean=False)
                pipe_base = Pipeline([("scaler", scaler_b), ("est", base_est)])
                acc_base = cross_val_score(
                    pipe_base, X.values, y, cv=cv, scoring=score_name, n_jobs=N_JOBS
                ).mean()

                # Safe k grid for PCA/SKB
                n_feat = X.shape[1]
                n_splits_cv = getattr(cv, "n_splits", cv.get_n_splits())
                min_train_size = max(2, int(np.floor(len(X) * (n_splits_cv - 1) / n_splits_cv)))
                max_k_by_samples = max(1, min(n_feat, min_train_size - 1))

                k_raw = [max(1, n_feat // 10), 5, 10, min(20, n_feat)]
                k_grid = sorted({k for k in k_raw if 1 <= k <= max_k_by_samples}) or [min(n_feat, max_k_by_samples)]

                # PCA
                best_pca = (-1e9, None)
                for k in k_grid:
                    try:
                        pipe_pca = Pipeline([
                            ("scaler", scaler_b),
                            ("pca", PCA(n_components=k, svd_solver="auto", random_state=random_state)),
                            ("est", base_est),
                        ])
                        sc = cross_val_score(
                            pipe_pca, X.values, y, cv=cv, scoring=score_name, n_jobs=N_JOBS
                        ).mean()
                        if sc > best_pca[0]: best_pca = (sc, k)
                    except Exception:
                        continue
                acc_pca, best_pca_k = best_pca

                # SKB
                score_func_eff = mutual_info_classif if task == "clf" else f_regression
                best_skb = (-1e9, None)
                for k in k_grid:
                    try:
                        pipe_skb = Pipeline([
                            ("scaler", scaler_b),
                            ("skb", SelectKBest(score_func=score_func_eff, k=k)),
                            ("est", base_est),
                        ])
                        sc = cross_val_score(
                            pipe_skb, X.values, y, cv=cv, scoring=score_name, n_jobs=N_JOBS
                        ).mean()
                        if sc > best_skb[0]: best_skb = (sc, k)
                    except Exception:
                        continue
                acc_skb, best_skb_k = best_skb

            #  Logging to DB
            if log_run is not None:
                try:
                    sel_indices = np.where(mask)[0].tolist()
                    params = {
                        "task": task, "generations": int(generations), "pop_size": int(pop_size),
                        "cv_splits": int(cv_splits), "lambda_penalty": float(lambda_penalty),
                        "crossover_prob": float(crossover_prob), "mutation_prob": float(mutation_prob),
                        "random_state": int(random_state),
                        "model_clf": model_name_clf, "model_reg": model_name_reg,
                        "fast_rows": int(fast_rows), "high_card_cutoff": int(high_card_cutoff),
                        "run_baselines": bool(run_baselines),
                        "best_pca_k": int(best_pca_k) if best_pca_k is not None else None,
                        "best_skb_k": int(best_skb_k) if best_skb_k is not None else None,
                    }
                    filename_f = filename or "inline"
                    stored_path_f = stored_path or ""  # NOT NULL in schema

                    log_run(
                        filename=filename_f,
                        stored_path=stored_path_f,
                        target_column=target_col,
                        n_features_total=int(X.shape[1]),
                        n_features_selected=int(mask.sum()),
                        acc_ga=float(ga_raw_score),
                        acc_pca=(float(acc_pca) if acc_pca is not None else None),
                        acc_skb=(float(acc_skb) if acc_skb is not None else None),
                        selected_indices=sel_indices,  
                        params=params,                  
                    )
                    st.toast("Logged to DB", icon="🗂️")
                except Exception as e:
                    st.caption(f"(Logging skipped: {e})")
            else:
                st.caption("(DB logging module not available.)")

            st.success("GA finished", icon="✅")

            # Tabs
            tab_res, tab_feats, tab_base, tab_logs = st.tabs(["📈 Results", "🧩 Selected features", "🧪 Baselines", "📝 Logs"])

            with tab_res:
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric-card"><div class="small">GA (raw CV)</div><h2>{ga_raw_score:.4f}</h2></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="small">GA (penalized)</div><h2>{ga.best_fitness_:.4f}</h2></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="small">Selected</div><h2>{mask.sum()} / {len(mask)}</h2></div>', unsafe_allow_html=True)

                # plots
                try:
                    ga_raw_pipe = Pipeline([("scaler", scaler), ("est", base_est)])
                    if task == "clf":
                        y_pred = cross_val_predict(ga_raw_pipe, X.values[:, mask], y, cv=cv, n_jobs=1)
                        cm = confusion_matrix(y, y_pred)
                        fig, ax = plt.subplots(figsize=(6,6))
                        ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax)
                        ax.set_title("Confusion Matrix (GA subset)")
                        st.pyplot(fig, use_container_width=True)
                    else:
                        y_pred = cross_val_predict(ga_raw_pipe, X.values[:, mask], y, cv=cv, n_jobs=1)
                        fig, ax = plt.subplots()
                        ax.scatter(y, y_pred, s=12, alpha=.6)
                        lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
                        ax.plot(lims, lims, "k--", lw=1)
                        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
                        ax.set_title("Predicted vs True (GA subset)")
                        st.pyplot(fig, use_container_width=True)
                except Exception:
                    st.caption("Plot skipped (small sample or incompatible types).")

            with tab_feats:
                st.markdown("#### Selected features")
                if selected:
                    st.dataframe(pd.DataFrame({"feature": selected}), use_container_width=True, hide_index=True)
                else:
                    st.write("(no features)")
                st.download_button(
                    "Download selected subset (CSV)",
                    df_sel.to_csv(index=False).encode("utf-8"),
                    "selected_subset.csv",
                    "text/csv",
                )
                st.download_button(
                    "Download selected mask (CSV)",
                    pd.Series(mask.astype(int), name="selected").to_csv(index=False).encode("utf-8"),
                    "selected_mask.csv",
                    "text/csv",
                )

            with tab_base:
                if run_baselines:
                    if acc_base is not None:
                        st.write(f"- **BASE (all features)**: {score_name} = **{acc_base:.4f}**")
                    if acc_pca is not None:
                        st.write(f"- **PCA** best k={best_pca_k}: **{acc_pca:.4f}**")
                    if acc_skb is not None:
                        st.write(f"- **SelectKBest** best k={best_skb_k}: **{acc_skb:.4f}**")
                else:
                    st.caption("Baselines skipped (enable in Controls if needed).")

            with tab_logs:
                if get_logs is not None:
                    try:
                        logs = get_logs()
                        st.dataframe(logs, use_container_width=True)
                    except Exception as e:
                        st.caption(f"(Couldn't read DB logs: {e})")
                else:
                    st.caption("No DB logging module detected.")
