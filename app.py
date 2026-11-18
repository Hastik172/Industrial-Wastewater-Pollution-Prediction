"""
AquaPredict — Industrial Wastewater Pollution Prediction Using Machine Learning
A college mini-project demonstrating 8 Unit 5 ML algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import (
    BaggingClassifier, BaggingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
import warnings, io, os
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AquaPredict",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #1a2744 100%);
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown p { color: #f1f5f9 !important; }
section[data-testid="stSidebar"] .stRadio label:hover { color: #38bdf8 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }

/* Main */
.main .block-container { max-width: 1200px; padding: 2rem 1rem; }
h1 { color: #0f172a; font-weight: 700; }
h2 { color: #1e293b; font-weight: 600; }
h3 { color: #334155; font-weight: 600; }

/* Cards */
.metric-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.2rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,.1); }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0891b2; }
.metric-card .label { font-size: .85rem; color: #64748b; margin-top: 4px; }

/* Info boxes */
.algo-info {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-left: 4px solid #0891b2; border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin: 1rem 0; font-size: .92rem; color: #334155;
}
.success-box {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-left: 4px solid #16a34a; border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin: 1rem 0;
}
.warning-box {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left: 4px solid #d97706; border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem; margin: 1rem 0;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #0891b2, #0e7490); color: #fff;
    border: none; border-radius: 8px; padding: .5rem 2rem; font-weight: 600;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #0e7490, #155e75);
    box-shadow: 0 4px 12px rgba(8,145,178,.3);
}
</style>
"""

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _card(label, value, col):
    col.markdown(
        f'<div class="metric-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def generate_synthetic_data(n=500):
    """Generate a clearly-labelled synthetic wastewater dataset."""
    rng = np.random.RandomState(42)
    bod = np.round(rng.lognormal(3.5, 0.8, n).clip(5, 500), 1)
    cod = np.round((bod * rng.uniform(1.5, 3.0, n) + rng.normal(0, 20, n)).clip(20, 1500), 1)
    tss = np.round(rng.lognormal(3.8, 0.9, n).clip(10, 800), 1)
    turb = np.round((tss * rng.uniform(0.3, 0.8, n) + rng.normal(0, 10, n)).clip(1, 500), 1)
    do_ = np.round(rng.exponential(4, n).clip(0.5, 14), 2)
    df = pd.DataFrame({
        "pH": np.round(rng.normal(7.2, 1.5, n).clip(2, 12), 2),
        "Temperature_C": np.round(rng.normal(28, 5, n).clip(10, 50), 1),
        "Dissolved_Oxygen_mg_L": do_,
        "BOD_mg_L": bod,
        "COD_mg_L": cod,
        "TSS_mg_L": tss,
        "TDS_mg_L": np.round(rng.normal(500, 200, n).clip(50, 2000), 0),
        "Turbidity_NTU": turb,
        "Conductivity_uS_cm": np.round(rng.normal(800, 300, n).clip(100, 3000), 0),
        "Nitrate_mg_L": np.round(rng.exponential(15, n).clip(0.1, 100), 2),
    })
    # Pollution level from data-driven quantiles (NOT official thresholds)
    score = bod / 100 + cod / 300 + tss / 200 + turb / 100 - do_ / 10
    q33, q66 = np.percentile(score, [33, 66])
    df["Pollution_Level"] = pd.cut(
        score, bins=[-np.inf, q33, q66, np.inf], labels=["Low", "Moderate", "High"]
    )
    # Sprinkle ~3 % NaN for realism
    for c in ["Dissolved_Oxygen_mg_L", "TSS_mg_L", "Nitrate_mg_L"]:
        df.loc[rng.random(n) < 0.03, c] = np.nan
    return df


def _load_data():
    if "df" not in st.session_state:
        real_csv = "real_wastewater_clean.csv"
        if os.path.exists(real_csv):
            st.session_state.df = pd.read_csv(real_csv)
            st.session_state.data_source = "real_uci"
        else:
            st.session_state.df = generate_synthetic_data()
            st.session_state.data_source = "synthetic"
    return st.session_state.df


def _split():
    """Return (X_train, X_test, y_train, y_test, task_type) or Nones."""
    if "X_train" not in st.session_state:
        return None, None, None, None, None
    return (
        st.session_state.X_train, st.session_state.X_test,
        st.session_state.y_train, st.session_state.y_test,
        st.session_state.task_type,
    )


def _metrics(y_true, y_pred, task):
    if task == "classification":
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


def _show_metrics(m):
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        _card(k, f"{v:.4f}", c)


def _store(name, model, metrics, task):
    if "model_results" not in st.session_state:
        st.session_state.model_results = {}
    st.session_state.model_results[name] = dict(model=model, metrics=metrics, task_type=task)


def _cm_fig(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    labs = [str(l) for l in (labels if labels is not None else sorted(set(y_true) | set(y_pred)))]
    fig = px.imshow(cm, text_auto=True, x=labs, y=labs,
                    color_continuous_scale="Teal",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(title="Confusion Matrix", height=400, template="plotly_white")
    return fig


def _fi_fig(importances, names, title="Feature Importance"):
    df = pd.DataFrame({"Feature": names, "Importance": importances}).sort_values("Importance")
    fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Teal")
    fig.update_layout(title=title, height=max(300, len(names) * 35),
                      showlegend=False, template="plotly_white")
    return fig


def _scatter_fig(y_true, y_pred, title="Actual vs Predicted"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(color="#0891b2", opacity=.6, size=6),
                             name="Predictions"))
    lo, hi = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#ef4444", dash="dash"),
                             name="Perfect"))
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted",
                      height=400, template="plotly_white")
    return fig


def _data_check():
    X_tr, X_te, y_tr, y_te, task = _split()
    if X_tr is None:
        auto_prepare_and_train()
        X_tr, X_te, y_tr, y_te, task = _split()
    return X_tr, X_te, y_tr, y_te, task


def auto_prepare_and_train():
    """Automatically prepare the real dataset and train all 8 models."""
    df = _load_data()
    all_cols = df.columns.tolist()
    target_col = "Pollution_Level" if "Pollution_Level" in all_cols else ("COD_mg_L" if "COD_mg_L" in all_cols else all_cols[-1])
    remaining = [c for c in all_cols if c != target_col]
    feature_cols = [c for c in remaining if df[c].dtype in ("float64", "int64", "float32", "int32")]

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    mask = y.notna()
    X, y = X[mask], y[mask]

    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=feature_cols)

    target_series = y.dropna()
    n_unique = target_series.nunique()
    task_type = "classification" if (n_unique <= 15 or target_series.dtype == "object" or str(target_series.dtype) == "category") else "regression"

    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    else:
        y = y.astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    st.session_state.update(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        task_type=task_type, feature_cols=feature_cols, target_col=target_col,
        scaler=scaler, label_encoder=le, imputer=imputer, model_results={}
    )

    # 1. CART (tuned: entropy criterion, deeper tree, regularised leaves)
    cart = DecisionTreeClassifier(max_depth=6, min_samples_split=5, min_samples_leaf=2, criterion="entropy", random_state=42) if task_type == "classification" else DecisionTreeRegressor(max_depth=6, min_samples_split=5, min_samples_leaf=2, random_state=42)
    cart.fit(X_train, y_train)
    _store("CART Decision Tree", cart, _metrics(y_test, cart.predict(X_test), task_type), task_type)

    # 2. Bagging (tuned: deeper base trees, more estimators)
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=7, min_samples_split=3, random_state=42), n_estimators=100, max_samples=0.85, random_state=42) if task_type == "classification" else BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=7, min_samples_split=3, random_state=42), n_estimators=100, max_samples=0.85, random_state=42)
    bag.fit(X_train, y_train)
    _store("Bagging", bag, _metrics(y_test, bag.predict(X_test), task_type), task_type)

    # 3. Random Forest (tuned: more trees, optimised depth)
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=3, oob_score=True, random_state=42) if task_type == "classification" else RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=3, oob_score=True, random_state=42)
    rf.fit(X_train, y_train)
    _store("Random Forest", rf, _metrics(y_test, rf.predict(X_test), task_type), task_type)

    # 4. AdaBoost (tuned: depth-2 stumps, more estimators, lower learning rate)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=42), n_estimators=60, learning_rate=0.8, random_state=42) if task_type == "classification" else AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=2, random_state=42), n_estimators=60, learning_rate=0.8, random_state=42)
    ada.fit(X_train, y_train)
    _store("AdaBoost", ada, _metrics(y_test, ada.predict(X_test), task_type), task_type)

    # 5. Gradient Boosting (tuned: more estimators, deeper trees)
    gb = GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42) if task_type == "classification" else GradientBoostingRegressor(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    _store("Gradient Boosting", gb, _metrics(y_test, gb.predict(X_test), task_type), task_type)

    # 6. ABM (MLP) (tuned: wider layers, higher learning rate)
    abm = MLPClassifier(hidden_layer_sizes=(128, 64), activation="tanh", max_iter=500, learning_rate_init=0.003, random_state=42, early_stopping=True) if task_type == "classification" else MLPRegressor(hidden_layer_sizes=(128, 64), activation="tanh", max_iter=500, learning_rate_init=0.003, random_state=42, early_stopping=True)
    abm.fit(X_train, y_train)
    _store("ABM (MLP)", abm, _metrics(y_test, abm.predict(X_test), task_type), task_type)

    # 7. GAM (Spline approximation, tuned: higher regularisation C)
    from sklearn.preprocessing import SplineTransformer
    from sklearn.pipeline import make_pipeline
    gam = make_pipeline(SplineTransformer(n_knots=8, degree=3), LogisticRegression(max_iter=1000, C=2.0, random_state=42)) if task_type == "classification" else make_pipeline(SplineTransformer(n_knots=8, degree=3), Ridge(alpha=1.0))
    gam.fit(X_train, y_train)
    _store("GAM (Spline Approx.)", gam, _metrics(y_test, gam.predict(X_test), task_type), task_type)

    # 8. Voting Ensemble (tuned: RF + GB + MLP with soft voting)
    v_est = [("rf", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)), ("gb", GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)), ("abm", MLPClassifier(hidden_layer_sizes=(128, 64), activation="tanh", max_iter=400, random_state=42))] if task_type == "classification" else [("rf", RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)), ("gb", GradientBoostingRegressor(n_estimators=120, max_depth=4, random_state=42))]
    vote = VotingClassifier(estimators=v_est, voting="soft") if task_type == "classification" else VotingRegressor(estimators=v_est)
    vote.fit(X_train, y_train)
    _store("Voting Ensemble", vote, _metrics(y_test, vote.predict(X_test), task_type), task_type)

    # 9. Stacking Ensemble (tuned: RF + GB + DT base, LR meta)
    s_est = [("rf", RandomForestClassifier(n_estimators=120, max_depth=9, random_state=42)), ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)), ("dt", DecisionTreeClassifier(max_depth=6, random_state=42))] if task_type == "classification" else [("rf", RandomForestRegressor(n_estimators=120, max_depth=9, random_state=42)), ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)), ("dt", DecisionTreeRegressor(max_depth=6, random_state=42))]
    stack = StackingClassifier(estimators=s_est, final_estimator=LogisticRegression(max_iter=1000, random_state=42), cv=3) if task_type == "classification" else StackingRegressor(estimators=s_est, final_estimator=Ridge(alpha=1.0), cv=3)
    stack.fit(X_train, y_train)
    _store("Stacking Ensemble", stack, _metrics(y_test, stack.predict(X_test), task_type), task_type)



# ══════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════

def page_overview():
    st.markdown("# 🌊 AquaPredict Dashboard")
    st.markdown("### Industrial Wastewater Pollution Prediction Using Machine Learning")
    st.markdown(
        '<div class="algo-info">'
        "<b>Project Objective:</b> Demonstrate the implementation of 8 Unit 5 machine-learning "
        "algorithms for predicting industrial wastewater pollution levels. The system analyses "
        "parameters such as pH, BOD, COD, TSS, dissolved oxygen and turbidity to classify "
        "pollution severity or predict pollutant concentrations."
        "</div>",
        unsafe_allow_html=True,
    )

    df = _load_data()
    src = st.session_state.get("data_source", "synthetic")
    if src == "real_uci":
        st.success("🌿 **Real Benchmark Dataset Active:** UCI Urban Wastewater Treatment Plant Dataset (527 actual daily operational sensor observations).")
    elif src == "synthetic":
        st.info("📌 Currently using **synthetic demo data** (clearly labelled — not real measurements).")
    else:
        st.success("📁 Currently using custom uploaded CSV dataset.")

    # Metric cards
    num_cols = df.select_dtypes("number").columns.tolist()
    c1, c2, c3, c4 = st.columns(4)
    _card("Samples", str(len(df)), c1)
    _card("Features", str(len(num_cols)), c2)
    target = st.session_state.get("target_col", "Pollution_Level" if "Pollution_Level" in df.columns else "—")
    _card("Target", str(target), c3)
    task = st.session_state.get("task_type", "classification" if "Pollution_Level" in df.columns else "—")
    _card("Task", str(task).title(), c4)

    st.markdown("")
    col_btn, col_status = st.columns([1, 2])
    with col_btn:
        if st.button("⚡ Train All 8 Models Now", type="primary", use_container_width=True):
            with st.spinner("Training all 8 Unit 5 models on real data..."):
                auto_prepare_and_train()
                st.success("All 8 models trained successfully!")
                st.rerun()
    with col_status:
        n_mod = len(st.session_state.get("model_results", {}))
        if n_mod > 0:
            st.markdown(f'<div class="success-box" style="margin:0; padding:0.6rem 1rem;">✅ <b>{n_mod} Models Trained & Ready</b> (CART, Bagging, RF, AdaBoost, GB, ABM, GAM, Ensembles)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box" style="margin:0; padding:0.6rem 1rem;">Click "Train All 8 Models Now" to train all models</div>', unsafe_allow_html=True)

    st.markdown("")

    col_a, col_b = st.columns(2)

    # Class distribution
    with col_a:
        if "Pollution_Level" in df.columns:
            st.markdown("#### Pollution Class Distribution")
            vc = df["Pollution_Level"].value_counts()
            fig = px.pie(names=vc.index.astype(str), values=vc.values,
                         color_discrete_sequence=["#0891b2", "#f59e0b", "#ef4444"],
                         hole=0.4)
            fig.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    with col_b:
        st.markdown("#### Feature Correlation")
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Trained-model summary
    results = st.session_state.get("model_results", {})
    if results:
        st.markdown("#### Trained Model Performance")
        task_type = st.session_state.get("task_type", "classification")
        if task_type == "classification":
            perf = {n: r["metrics"].get("Accuracy", 0) for n, r in results.items()
                    if r["task_type"] == "classification"}
            if perf:
                fig = px.bar(x=list(perf.keys()), y=list(perf.values()),
                             labels={"x": "Model", "y": "Accuracy"},
                             color=list(perf.values()), color_continuous_scale="Teal")
                fig.update_layout(height=350, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            perf = {n: r["metrics"].get("R²", 0) for n, r in results.items()
                    if r["task_type"] == "regression"}
            if perf:
                fig = px.bar(x=list(perf.keys()), y=list(perf.values()),
                             labels={"x": "Model", "y": "R²"},
                             color=list(perf.values()), color_continuous_scale="Teal")
                fig.update_layout(height=350, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="footer">AquaPredict · College Mini Project · Unit 5 ML Algorithms</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE: DATASET & PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def page_dataset():
    st.markdown("# 📊 Dataset & Preprocessing")

    # Quick Dataset Selectors
    col_ds1, col_ds2, col_ds3 = st.columns(3)
    with col_ds1:
        if st.button("🌿 Load Real UCI Dataset (10 features)", use_container_width=True):
            if os.path.exists("real_wastewater_clean.csv"):
                st.session_state.df = pd.read_csv("real_wastewater_clean.csv")
                st.session_state.data_source = "real_uci"
                # Reset prepared split so user re-prepares
                st.session_state.pop("X_train", None)
                st.rerun()
    with col_ds2:
        if st.button("📋 Load Full UCI Dataset (39 features)", use_container_width=True):
            if os.path.exists("real_wastewater_data.csv"):
                st.session_state.df = pd.read_csv("real_wastewater_data.csv")
                st.session_state.data_source = "real_uci_full"
                st.session_state.pop("X_train", None)
                st.rerun()
    with col_ds3:
        if st.button("🧪 Load Synthetic Demo Dataset", use_container_width=True):
            st.session_state.df = generate_synthetic_data()
            st.session_state.data_source = "synthetic"
            st.session_state.pop("X_train", None)
            st.rerun()

    # Upload
    uploaded = st.file_uploader("Or upload your own CSV file", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded)
            st.session_state.data_source = "uploaded"
            st.session_state.pop("X_train", None)
            st.success("✅ Dataset uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    df = _load_data()
    src = st.session_state.get("data_source", "real_uci")

    if src in ("real_uci", "real_uci_full"):
        st.markdown(
            '<div class="success-box">🌿 <b>Real Dataset Active:</b> UCI Machine Learning Repository — '
            "Urban Wastewater Treatment Plant Dataset (527 actual daily operational sensor measurements "
            "from plant stages: Flow, pH, BOD, COD, TSS, Conductivity, Sediments, Zinc, and Effluent Quality).</div>",
            unsafe_allow_html=True,
        )
    elif src == "synthetic":
        st.markdown(
            '<div class="warning-box">⚠️ <b>Synthetic Data Active</b> — This dataset is generated '
            "for demonstration purposes.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="success-box">✅ <b>Custom Uploaded Dataset Active</b>.</div>',
            unsafe_allow_html=True,
        )

    # Preview
    st.markdown("### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Stats & missing
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Basic Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
    with c2:
        st.markdown("### Missing Values")
        miss = df.isnull().sum()
        miss_df = pd.DataFrame({"Column": miss.index, "Missing": miss.values,
                                 "% Missing": (miss.values / len(df) * 100).round(2)})
        miss_df = miss_df[miss_df["Missing"] > 0]
        if miss_df.empty:
            st.success("No missing values! ✅")
        else:
            st.dataframe(miss_df, use_container_width=True)

    # Distributions
    num_cols = df.select_dtypes("number").columns.tolist()
    if num_cols:
        st.markdown("### Feature Distributions")
        n_cols_grid = 3
        rows = (len(num_cols) + n_cols_grid - 1) // n_cols_grid
        for r in range(rows):
            cols = st.columns(n_cols_grid)
            for ci in range(n_cols_grid):
                idx = r * n_cols_grid + ci
                if idx < len(num_cols):
                    with cols[ci]:
                        fig = px.histogram(df, x=num_cols[idx], nbins=30,
                                           color_discrete_sequence=["#0891b2"])
                        fig.update_layout(height=250, margin=dict(t=30, b=20),
                                          template="plotly_white",
                                          title=num_cols[idx])
                        st.plotly_chart(fig, use_container_width=True)

    # Feature & target selection
    st.markdown("---")
    st.markdown("### Feature & Target Selection")
    all_cols = df.columns.tolist()

    default_target = "Pollution_Level" if "Pollution_Level" in all_cols else (
        "COD_mg_L" if "COD_mg_L" in all_cols else all_cols[-1]
    )
    target_col = st.selectbox("Select Target Column", all_cols,
                               index=all_cols.index(default_target) if default_target in all_cols else 0)

    remaining = [c for c in all_cols if c != target_col]
    # Use all numeric columns as default features (pollution index is multi-parameter, no leakage)
    default_features = [c for c in remaining if df[c].dtype in ("float64", "int64", "float32", "int32")]
    feature_cols = st.multiselect("Select Input Features", remaining, default=default_features)

    if not feature_cols:
        st.warning("Please select at least one input feature.")
        return

    # Task type
    target_series = df[target_col].dropna()
    n_unique = target_series.nunique()
    auto_task = "classification" if (n_unique <= 15 or target_series.dtype == "object"
                                      or str(target_series.dtype) == "category") else "regression"
    task_type = st.radio("Task Type", ["classification", "regression"],
                          index=0 if auto_task == "classification" else 1,
                          horizontal=True,
                          help=f"Auto-detected: {auto_task} ({n_unique} unique target values)")

    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

    if st.button("🚀 Prepare Data", type="primary"):
        with st.spinner("Preprocessing…"):
            X = df[feature_cols].copy()
            y = df[target_col].copy()

            # Drop rows where target is NaN
            mask = y.notna()
            X, y = X[mask], y[mask]

            # Impute missing features
            imputer = SimpleImputer(strategy="median")
            X_arr = imputer.fit_transform(X)
            X = pd.DataFrame(X_arr, columns=feature_cols)

            # Encode target for classification
            le = None
            if task_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
            else:
                y = y.astype(float).values

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42,
                stratify=y if task_type == "classification" else None,
            )

            # Save to session
            st.session_state.update(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                task_type=task_type, feature_cols=feature_cols, target_col=target_col,
                scaler=scaler, label_encoder=le, imputer=imputer, model_results={},
            )

        st.markdown(
            f'<div class="success-box">✅ Data prepared! '
            f"Train: <b>{len(X_train)}</b> samples · Test: <b>{len(X_test)}</b> samples · "
            f"Task: <b>{task_type}</b></div>",
            unsafe_allow_html=True,
        )



# ══════════════════════════════════════════════════════════════════
#  PAGE: CART DECISION TREE
# ══════════════════════════════════════════════════════════════════

def page_cart():
    st.markdown("# 🌳 CART Decision Tree")
    st.markdown(
        '<div class="algo-info">'
        "<b>CART (Classification and Regression Trees)</b> builds a binary tree by recursively "
        "splitting the data on the feature and threshold that best reduces impurity. For "
        "classification it uses <b>Gini impurity</b>; for regression it minimises <b>MSE</b>. "
        "Each leaf node holds a class label (classification) or mean value (regression)."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        max_depth = c1.slider("Max Depth", 1, 20, 6, key="cart_depth")
        min_split = c2.slider("Min Samples Split", 2, 20, 5, key="cart_split")
        criterion = c3.selectbox("Criterion",
                                  ["entropy", "gini"] if task == "classification" else ["squared_error", "friedman_mse"],
                                  key="cart_crit")

    if st.button("🚀 Train CART Model", type="primary"):
        with st.spinner("Training…"):
            if task == "classification":
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split,
                                                min_samples_leaf=2, criterion=criterion, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_split,
                                               min_samples_leaf=2, criterion=criterion, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            m = _metrics(y_te, preds, task)
            _store("CART Decision Tree", model, m, task)

        st.markdown("### Performance Metrics")
        _show_metrics(m)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Tree Visualisation")
            fig_t, ax_t = plt.subplots(figsize=(12, 6))
            feature_names = st.session_state.feature_cols
            class_names = (list(st.session_state.label_encoder.classes_)
                           if task == "classification" and st.session_state.get("label_encoder") else None)
            plot_tree(model, max_depth=3, filled=True, rounded=True,
                      feature_names=feature_names, class_names=class_names,
                      fontsize=8, ax=ax_t)
            ax_t.set_title("Decision Tree (top 3 levels)", fontsize=10)
            st.pyplot(fig_t)
            plt.close(fig_t)

        with col2:
            st.markdown("#### Feature Importance")
            st.plotly_chart(_fi_fig(model.feature_importances_, feature_names),
                            use_container_width=True)

        if task == "classification":
            st.plotly_chart(_cm_fig(y_te, preds), use_container_width=True)
        else:
            st.plotly_chart(_scatter_fig(y_te, preds), use_container_width=True)

        with st.expander("📄 Tree Text Representation"):
            st.code(export_text(model, feature_names=feature_names, max_depth=4))

        st.markdown(
            '<div class="success-box">📝 <b>Interpretation:</b> '
            "The CART model found the most informative splits among the wastewater features. "
            "The tree visualisation shows the decision logic the model uses. "
            "Features with higher importance contribute more to the splitting decisions.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
#  PAGE: BAGGING
# ══════════════════════════════════════════════════════════════════

def page_bagging():
    st.markdown("# 🎒 Bagging (Bootstrap Aggregating)")
    st.markdown(
        '<div class="algo-info">'
        "<b>Bagging</b> trains multiple base learners (usually decision trees) on different "
        "<b>bootstrap samples</b> (random samples with replacement) of the training data, then "
        "combines their predictions by voting (classification) or averaging (regression). This "
        "reduces variance and helps prevent overfitting."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2 = st.columns(2)
        n_est = c1.slider("Number of Estimators", 5, 150, 100, key="bag_n")
        max_samp = c2.slider("Max Samples (fraction)", 0.5, 1.0, 0.85, 0.05, key="bag_samp")

    if st.button("🚀 Train Bagging Model", type="primary"):
        with st.spinner("Training…"):
            if task == "classification":
                base = DecisionTreeClassifier(max_depth=7, min_samples_split=3, random_state=42)
                bag = BaggingClassifier(estimator=base, n_estimators=n_est,
                                         max_samples=max_samp, random_state=42)
                single = DecisionTreeClassifier(random_state=42)
            else:
                base = DecisionTreeRegressor(max_depth=7, min_samples_split=3, random_state=42)
                bag = BaggingRegressor(estimator=base, n_estimators=n_est,
                                        max_samples=max_samp, random_state=42)
                single = DecisionTreeRegressor(random_state=42)

            bag.fit(X_tr, y_tr)
            single.fit(X_tr, y_tr)

            pred_bag = bag.predict(X_te)
            pred_single = single.predict(X_te)

            m_bag = _metrics(y_te, pred_bag, task)
            m_single = _metrics(y_te, pred_single, task)
            _store("Bagging", bag, m_bag, task)

        st.markdown("### Performance Metrics (Bagging)")
        _show_metrics(m_bag)

        # Bootstrap illustration
        st.markdown("#### Bootstrap Sampling Illustration")
        rng = np.random.RandomState(0)
        sample_size = min(20, len(X_tr))
        indices = rng.choice(sample_size, size=sample_size, replace=True)
        counts = pd.Series(indices).value_counts().reindex(range(sample_size), fill_value=0)
        colors = ["#0891b2" if c > 0 else "#e2e8f0" for c in counts.values]
        fig_bs = go.Figure(go.Bar(x=list(range(sample_size)), y=counts.values,
                                   marker_color=colors))
        fig_bs.update_layout(title="Bootstrap Sample: Times Each Training Sample Was Selected",
                              xaxis_title="Sample Index", yaxis_title="Times Selected",
                              height=300, template="plotly_white")
        st.plotly_chart(fig_bs, use_container_width=True)
        oob_count = int((counts == 0).sum())
        st.markdown(f"Out-of-bag (OOB) samples: **{oob_count}** out of {sample_size} "
                     f"({oob_count/sample_size*100:.0f}%) — these are not seen during training of one tree.")

        # Comparison
        st.markdown("#### Single Tree vs Bagging Comparison")
        comp_df = pd.DataFrame({"Single Tree": m_single, "Bagging": m_bag})
        st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

        metric_name = list(m_bag.keys())[0]
        fig_comp = go.Figure()
        fig_comp.add_bar(name="Single Tree", x=[metric_name], y=[m_single[metric_name]],
                          marker_color="#94a3b8")
        fig_comp.add_bar(name="Bagging", x=[metric_name], y=[m_bag[metric_name]],
                          marker_color="#0891b2")
        fig_comp.update_layout(barmode="group", height=350, template="plotly_white",
                                title="Single Tree vs Bagging")
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown(
            '<div class="success-box">📝 <b>Interpretation:</b> '
            "Bagging aggregates multiple trees trained on bootstrap samples, which typically "
            "reduces variance compared to a single decision tree. The bootstrap illustration "
            "shows how some samples are picked multiple times while others are left out (OOB).</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
#  PAGE: RANDOM FOREST
# ══════════════════════════════════════════════════════════════════

def page_random_forest():
    st.markdown("# 🌲 Random Forest")
    st.markdown(
        '<div class="algo-info">'
        "<b>Random Forest</b> extends bagging by adding <b>feature randomness</b>: at each "
        "split, only a random subset of features is considered. This de-correlates the "
        "individual trees and generally improves generalisation beyond standard bagging."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    n_features = X_tr.shape[1]
    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_est = c1.slider("Number of Trees", 10, 300, 150, key="rf_n")
        max_d = c2.slider("Max Depth", 1, 30, 10, key="rf_depth")
        max_f_opt = c3.selectbox("Max Features", ["sqrt", "log2", "all"], key="rf_mf")
        max_f = None if max_f_opt == "all" else max_f_opt

    if st.button("🚀 Train Random Forest", type="primary"):
        with st.spinner("Training…"):
            if task == "classification":
                model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                                max_features=max_f, oob_score=True,
                                                random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d,
                                               max_features=max_f, oob_score=True,
                                               random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            m = _metrics(y_te, preds, task)
            _store("Random Forest", model, m, task)

        st.markdown("### Performance Metrics")
        _show_metrics(m)

        # OOB score
        if hasattr(model, "oob_score_"):
            st.markdown(f"**OOB Score:** {model.oob_score_:.4f}")

        st.markdown(f"**Number of Trees:** {n_est}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature Importance")
            st.plotly_chart(_fi_fig(model.feature_importances_, st.session_state.feature_cols),
                            use_container_width=True)
        with col2:
            if task == "classification":
                st.plotly_chart(_cm_fig(y_te, preds), use_container_width=True)
            else:
                st.plotly_chart(_scatter_fig(y_te, preds), use_container_width=True)

        st.markdown(
            '<div class="success-box">📝 <b>Interpretation:</b> '
            "Random Forest combines many de-correlated decision trees. The OOB score provides "
            "an internal validation estimate. Feature importance reveals which wastewater "
            "parameters most influence predictions.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
#  PAGE: ADABOOST
# ══════════════════════════════════════════════════════════════════

def page_adaboost():
    st.markdown("# 🚀 AdaBoost (Adaptive Boosting)")
    st.markdown(
        '<div class="algo-info">'
        "<b>AdaBoost</b> trains weak learners <b>sequentially</b>. After each learner, "
        "misclassified samples receive higher weights so the next learner focuses on them. "
        "The final prediction is a weighted vote/sum of all learners."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_est = c1.slider("Number of Estimators", 10, 300, 60, key="ada_n")
        lr = c2.slider("Learning Rate", 0.01, 2.0, 0.8, 0.01, key="ada_lr")
        base_depth = c3.slider("Base Estimator Depth", 1, 5, 2, key="ada_depth")

    if st.button("🚀 Train AdaBoost Model", type="primary"):
        with st.spinner("Training…"):
            if task == "classification":
                base = DecisionTreeClassifier(max_depth=base_depth, random_state=42)
                model = AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr,
                                            random_state=42)
            else:
                base = DecisionTreeRegressor(max_depth=base_depth, random_state=42)
                model = AdaBoostRegressor(estimator=base, n_estimators=n_est, learning_rate=lr,
                                           random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            m = _metrics(y_te, preds, task)
            _store("AdaBoost", model, m, task)

        st.markdown("### Performance Metrics")
        _show_metrics(m)

        col1, col2 = st.columns(2)

        # Staged performance
        with col1:
            st.markdown("#### Boosting Performance Over Iterations")
            try:
                staged = list(model.staged_predict(X_te))
                if task == "classification":
                    stage_scores = [accuracy_score(y_te, sp) for sp in staged]
                    ylabel = "Accuracy"
                else:
                    stage_scores = [r2_score(y_te, sp) for sp in staged]
                    ylabel = "R² Score"
                fig_s = px.line(x=list(range(1, len(stage_scores) + 1)), y=stage_scores,
                                labels={"x": "Number of Estimators", "y": ylabel})
                fig_s.update_layout(height=350, template="plotly_white",
                                     title="Performance vs Number of Estimators")
                st.plotly_chart(fig_s, use_container_width=True)
            except Exception:
                st.info("Staged prediction not available for this configuration.")

        # Estimator weights / errors
        with col2:
            if hasattr(model, "estimator_errors_") and len(model.estimator_errors_) > 0:
                st.markdown("#### Learner Errors")
                fig_e = px.bar(x=list(range(1, len(model.estimator_errors_) + 1)),
                                y=model.estimator_errors_,
                                labels={"x": "Learner #", "y": "Error"},
                                color=model.estimator_errors_,
                                color_continuous_scale="OrRd")
                fig_e.update_layout(height=350, template="plotly_white",
                                     title="Individual Learner Errors", showlegend=False)
                st.plotly_chart(fig_e, use_container_width=True)
            elif hasattr(model, "estimator_weights_"):
                st.markdown("#### Estimator Weights")
                fig_w = px.bar(x=list(range(1, len(model.estimator_weights_) + 1)),
                                y=model.estimator_weights_,
                                labels={"x": "Learner #", "y": "Weight"},
                                color=model.estimator_weights_,
                                color_continuous_scale="Teal")
                fig_w.update_layout(height=350, template="plotly_white",
                                     title="Estimator Weights", showlegend=False)
                st.plotly_chart(fig_w, use_container_width=True)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            st.plotly_chart(_fi_fig(model.feature_importances_, st.session_state.feature_cols),
                            use_container_width=True)

        if task == "classification":
            st.plotly_chart(_cm_fig(y_te, preds), use_container_width=True)
        else:
            st.plotly_chart(_scatter_fig(y_te, preds), use_container_width=True)

        st.markdown(
            '<div class="success-box">📝 <b>Interpretation:</b> '
            "AdaBoost sequentially builds weak learners, each focusing on previously misclassified "
            "samples. The staged performance plot shows how accuracy improves as more learners are "
            "added. Learner errors indicate how well each individual weak learner performed.</div>",
            unsafe_allow_html=True,
        )



# ══════════════════════════════════════════════════════════════════
#  PAGE: GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════

def page_gradient_boosting():
    st.markdown("# 📈 Gradient Boosting")
    st.markdown(
        '<div class="algo-info">'
        "<b>Gradient Boosting</b> builds trees sequentially, where each new tree fits the "
        "<b>negative gradient</b> (residuals) of the loss function from the previous stage. "
        "It performs gradient descent in function space and typically achieves high accuracy."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        n_est = c1.slider("Number of Estimators", 50, 500, 120, key="gb_n")
        lr = c2.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01, key="gb_lr")
        max_d = c3.slider("Max Depth", 1, 10, 4, key="gb_depth")

    if st.button("🚀 Train Gradient Boosting", type="primary"):
        with st.spinner("Training…"):
            if task == "classification":
                model = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr,
                                                    max_depth=max_d, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr,
                                                   max_depth=max_d, random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            m = _metrics(y_te, preds, task)
            _store("Gradient Boosting", model, m, task)

        st.markdown("### Performance Metrics")
        _show_metrics(m)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Training Loss Over Iterations")
            if hasattr(model, "train_score_"):
                fig_loss = px.line(x=list(range(1, len(model.train_score_) + 1)),
                                   y=model.train_score_,
                                   labels={"x": "Iteration", "y": "Training Loss"})
                # Add test loss via staged_predict
                try:
                    staged = list(model.staged_predict(X_te))
                    if task == "classification":
                        test_scores = [accuracy_score(y_te, sp) for sp in staged]
                    else:
                        test_scores = [r2_score(y_te, sp) for sp in staged]
                    fig_loss.add_scatter(x=list(range(1, len(test_scores) + 1)),
                                         y=test_scores, mode="lines",
                                         name="Test Score", line=dict(color="#f59e0b"))
                except Exception:
                    pass
                fig_loss.update_layout(height=350, template="plotly_white",
                                        title="Train Loss Curve")
                st.plotly_chart(fig_loss, use_container_width=True)

        with col2:
            st.markdown("#### Feature Importance")
            st.plotly_chart(_fi_fig(model.feature_importances_, st.session_state.feature_cols),
                            use_container_width=True)

        if task == "classification":
            st.plotly_chart(_cm_fig(y_te, preds), use_container_width=True)
        else:
            st.plotly_chart(_scatter_fig(y_te, preds), use_container_width=True)

        st.markdown(
            '<div class="success-box">📝 <b>Interpretation:</b> '
            "Gradient Boosting iteratively reduces the loss by fitting trees to residual errors. "
            "The loss curve shows how training error decreases with more iterations. A divergence "
            "between train and test curves may indicate overfitting.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
#  PAGE: ABM (ADAPTIVE BASIS FUNCTION MODEL)
# ══════════════════════════════════════════════════════════════════

def page_abm():
    st.markdown("# 🧠 ABM (Adaptive Basis Function Model)")
    st.markdown(
        '<div class="algo-info">'
        "⚠️ <b>MLP-based ABM Approximation:</b> This page implements an MLP (Multi-Layer "
        "Perceptron) neural network as an Adaptive Basis Function Model approximation. The "
        "hidden units use <b>tanh activation</b> as adaptive basis functions — each neuron "
        "learns a parameterised nonlinear transformation of the input, and the output layer "
        "combines these basis functions linearly. This is a documented approach for "
        "demonstrating ABM concepts using readily available libraries."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    with st.expander("⚙️ Hyperparameters", expanded=True):
        c1, c2, c3 = st.columns(3)
        layers_str = c1.text_input("Hidden Layers (comma-separated)", "128,64", key="abm_layers")
        max_iter = c2.slider("Max Iterations", 100, 1000, 500, key="abm_iter")
        lr_init = c3.select_slider("Learning Rate", [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.05],
                                     value=0.003, key="abm_lr")
        st.info("Activation: **tanh** (fixed — used as adaptive basis functions)")

    # Parse layers
    try:
        layers = tuple(int(x.strip()) for x in layers_str.split(",") if x.strip())
    except ValueError:
        st.error("Invalid hidden layers format. Use comma-separated integers (e.g. 64,32).")
        return

    if st.button("🚀 Train ABM Model", type="primary"):
        with st.spinner("Training MLP-based ABM…"):
            if task == "classification":
                model = MLPClassifier(hidden_layer_sizes=layers, activation="tanh",
                                       max_iter=max_iter, learning_rate_init=lr_init,
                                       random_state=42, early_stopping=True,
                                       validation_fraction=0.15)
            else:
                model = MLPRegressor(hidden_layer_sizes=layers, activation="tanh",
                                      max_iter=max_iter, learning_rate_init=lr_init,
                                      random_state=42, early_stopping=True,
                                      validation_fraction=0.15)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            m = _metrics(y_te, preds, task)
            _store("ABM (MLP)", model, m, task)

        st.markdown("### Performance Metrics")
        _show_metrics(m)

        col1, col2 = st.columns(2)

        # Network architecture diagram
        with col1:
            st.markdown("#### Network Architecture")
            n_input = X_tr.shape[1]
            arch = [n_input] + list(layers) + [1 if task == "regression" else len(set(y_tr))]
            fig_nn, ax_nn = plt.subplots(figsize=(8, 5))
            ax_nn.set_xlim(-0.5, len(arch) - 0.5)
            max_nodes = max(arch)
            ax_nn.set_ylim(-0.5, max_nodes + 0.5)
            ax_nn.axis("off")

            layer_labels = ["Input"] + [f"Hidden {i+1}\n(tanh)" for i in range(len(layers))] + ["Output"]
            positions = []
            for li, n_nodes in enumerate(arch):
                y_offset = (max_nodes - n_nodes) / 2
                layer_pos = []
                display_n = min(n_nodes, 8)
                for ni in range(display_n):
                    y = y_offset + ni * (n_nodes / display_n) if display_n > 1 else max_nodes / 2
                    layer_pos.append((li, y))
                    color = "#0891b2" if li == 0 else ("#16a34a" if li == len(arch) - 1 else "#f59e0b")
                    circle = plt.Circle((li, y), 0.15, color=color, ec="white", lw=1.5, zorder=3)
                    ax_nn.add_patch(circle)
                if n_nodes > 8:
                    ax_nn.text(li, y_offset - 0.4, f"…({n_nodes})", ha="center", fontsize=7, color="#64748b")
                positions.append(layer_pos)
                ax_nn.text(li, -0.3, layer_labels[li], ha="center", fontsize=7, color="#334155", fontweight="bold")

            for li in range(len(positions) - 1):
                for p1 in positions[li]:
                    for p2 in positions[li + 1]:
                        ax_nn.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                    color="#cbd5e1", lw=0.3, zorder=1)

            ax_nn.set_title("MLP-based ABM Architecture", fontsize=10, fontweight="bold")
            st.pyplot(fig_nn)
            plt.close(fig_nn)

        # Loss curve
        with col2:
            st.markdown("#### Training Loss Curve")
            if hasattr(model, "loss_curve_"):
                fig_lc = px.line(x=list(range(1, len(model.loss_curve_) + 1)),
                                  y=model.loss_curve_,
                                  labels={"x": "Epoch", "y": "Loss"})
                fig_lc.update_layout(height=400, template="plotly_white",
                                      title="Loss Over Training Epochs")
                st.plotly_chart(fig_lc, use_container_width=True)
            else:
                st.info("Loss curve not available.")

        if task == "classification":
            st.plotly_chart(_cm_fig(y_te, preds), use_container_width=True)
        else:
            st.plotly_chart(_scatter_fig(y_te, preds), use_container_width=True)

        st.markdown(
            '<div class="success-box">📝 <b>Interpretation:</b> '
            "The MLP-based ABM uses tanh neurons as adaptive basis functions. Each hidden "
            "neuron learns a nonlinear feature transformation, and the output layer combines "
            "them. The loss curve shows convergence during training. Early stopping prevents "
            "overfitting by monitoring validation loss.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
#  PAGE: GAM (GENERALIZED ADDITIVE MODEL)
# ══════════════════════════════════════════════════════════════════

def page_gam():
    st.markdown("# 📐 GAM (Generalized Additive Model)")
    st.markdown(
        '<div class="algo-info">'
        "<b>GAM</b> models the target as a sum of smooth functions of individual features: "
        "<i>y = f₁(x₁) + f₂(x₂) + … + fₚ(xₚ)</i>. Each function is typically a spline, "
        "which captures nonlinear relationships while remaining interpretable — you can "
        "visualise the effect of each feature independently."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    # Check for pyGAM
    pygam_available = False
    try:
        from pygam import LinearGAM, LogisticGAM, s
        pygam_available = True
    except ImportError:
        pass

    is_multiclass = (task == "classification" and len(np.unique(y_tr)) > 2)
    use_pygam = pygam_available and not is_multiclass

    if use_pygam:
        st.success("✅ Using **pyGAM** for GAM implementation.")
    elif is_multiclass:
        st.info("ℹ️ Multiclass target detected (Low, Moderate, High). Using **Spline-based GAM (Cubic B-Splines + Multinomial Model)**, as pyGAM LogisticGAM only supports binary classes.")
    else:
        st.info("ℹ️ pyGAM not installed. Using **spline-based GAM approximation** with sklearn's SplineTransformer + Linear model.")

    with st.expander("⚙️ Hyperparameters", expanded=True):
        n_splines = st.slider("Number of Splines", 4, 20, 8, key="gam_splines")

    if st.button("🚀 Train GAM Model", type="primary"):
        with st.spinner("Training GAM…"):
            if use_pygam:
                from pygam import LinearGAM, LogisticGAM, s
                # Build spline terms
                terms = s(0, n_splines=n_splines)
                for i in range(1, X_tr.shape[1]):
                    terms = terms + s(i, n_splines=n_splines)

                if task == "classification":
                    model = LogisticGAM(terms)
                    model.gridsearch(X_tr, y_tr)
                else:
                    model = LinearGAM(terms)
                    model.gridsearch(X_tr, y_tr)

                preds = model.predict(X_te)
                m = _metrics(y_te, preds, task)
                _store("GAM", model, m, task)

                st.markdown("### Performance Metrics")
                _show_metrics(m)

                # Partial dependence plots
                st.markdown("#### Smooth Feature Effects")
                feature_names = st.session_state.feature_cols
                n_feat = min(len(feature_names), 6)
                n_cols_grid = 3
                n_rows = (n_feat + n_cols_grid - 1) // n_cols_grid

                for row in range(n_rows):
                    cols = st.columns(n_cols_grid)
                    for ci in range(n_cols_grid):
                        fi = row * n_cols_grid + ci
                        if fi < n_feat:
                            with cols[ci]:
                                XX = model.generate_X_grid(term=fi, n=100)
                                pdep, confi = model.partial_dependence(term=fi, X=XX, width=0.95)
                                x_vals = XX[:, fi]
                                fig_pd = go.Figure()
                                fig_pd.add_scatter(x=x_vals, y=confi[:, 0], mode="lines",
                                                    line=dict(color="rgba(8,145,178,0.2)"),
                                                    showlegend=False)
                                fig_pd.add_scatter(x=x_vals, y=confi[:, 1], mode="lines",
                                                    fill="tonexty",
                                                    line=dict(color="rgba(8,145,178,0.2)"),
                                                    showlegend=False)
                                fig_pd.add_scatter(x=x_vals, y=pdep, mode="lines",
                                                    line=dict(color="#0891b2", width=2),
                                                    name=feature_names[fi])
                                fig_pd.update_layout(title=feature_names[fi], height=280,
                                                      template="plotly_white",
                                                      xaxis_title=feature_names[fi],
                                                      yaxis_title="Partial Effect")
                                st.plotly_chart(fig_pd, use_container_width=True)

            else:
                # Fallback: SplineTransformer + LinearModel
                try:
                    from sklearn.preprocessing import SplineTransformer
                except ImportError:
                    st.error("sklearn SplineTransformer requires scikit-learn >= 1.0")
                    return

                from sklearn.pipeline import make_pipeline

                st.markdown("*Using spline-based GAM approximation (SplineTransformer + Linear Model)*")

                if task == "classification":
                    model = make_pipeline(
                        SplineTransformer(n_knots=n_splines, degree=3),
                        LogisticRegression(max_iter=1000, C=2.0, random_state=42),
                    )
                else:
                    model = make_pipeline(
                        SplineTransformer(n_knots=n_splines, degree=3),
                        Ridge(alpha=1.0),
                    )

                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                m = _metrics(y_te, preds, task)
                _store("GAM (Spline Approx.)", model, m, task)

                st.markdown("### Performance Metrics")
                _show_metrics(m)

                # Approximate partial dependence
                st.markdown("#### Approximate Feature Effects")
                feature_names = st.session_state.feature_cols
                n_feat = min(len(feature_names), 6)
                n_cols_grid = 3
                n_rows = (n_feat + n_cols_grid - 1) // n_cols_grid

                X_mean = X_tr.mean(axis=0)
                for row in range(n_rows):
                    cols = st.columns(n_cols_grid)
                    for ci in range(n_cols_grid):
                        fi = row * n_cols_grid + ci
                        if fi < n_feat:
                            with cols[ci]:
                                x_range = np.linspace(X_tr[:, fi].min(), X_tr[:, fi].max(), 100)
                                X_grid = np.tile(X_mean, (100, 1))
                                X_grid[:, fi] = x_range
                                if hasattr(model, "predict_proba"):
                                    y_grid = model.predict_proba(X_grid)[:, -1]
                                    ylabel = "P(High Pollution)"
                                else:
                                    y_grid = model.predict(X_grid)
                                    ylabel = "Prediction"
                                fig_pd = px.line(x=x_range, y=y_grid,
                                                  labels={"x": feature_names[fi], "y": ylabel})
                                fig_pd.update_layout(title=feature_names[fi], height=280,
                                                      template="plotly_white")
                                st.plotly_chart(fig_pd, use_container_width=True)

        if task == "classification":
            st.plotly_chart(_cm_fig(y_te, preds), use_container_width=True)
        else:
            st.plotly_chart(_scatter_fig(y_te, preds), use_container_width=True)

        st.markdown(
            '<div class="success-box">📝 <b>Interpretation:</b> '
            "GAM reveals how each individual feature influences the prediction through "
            "smooth partial-effect plots. Upward curves indicate the feature increases "
            "the target; downward curves indicate a decrease. This additive structure "
            "makes GAM highly interpretable.</div>",
            unsafe_allow_html=True,
        )



# ══════════════════════════════════════════════════════════════════
#  PAGE: ENSEMBLE LEARNING
# ══════════════════════════════════════════════════════════════════

def page_ensemble():
    st.markdown("# 🤝 Ensemble Learning")
    st.markdown(
        '<div class="algo-info">'
        "<b>Ensemble Learning</b> combines multiple models to improve prediction performance.<br>"
        "• <b>Voting:</b> Each base model votes; the final prediction is the majority "
        "vote (classification) or average (regression).<br>"
        "• <b>Stacking:</b> A meta-learner is trained on the base models' predictions "
        "to learn the optimal way to combine them."
        "</div>", unsafe_allow_html=True,
    )

    X_tr, X_te, y_tr, y_te, task = _data_check()
    if X_tr is None:
        return

    tab_vote, tab_stack = st.tabs(["📗 Voting Ensemble", "📘 Stacking Ensemble"])

    # ── Voting ──
    with tab_vote:
        if st.button("🚀 Train Voting Ensemble", type="primary", key="btn_vote"):
            with st.spinner("Training Voting Ensemble…"):
                if task == "classification":
                    estimators = [
                        ("rf", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)),
                        ("gb", GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)),
                        ("abm", MLPClassifier(hidden_layer_sizes=(128, 64), activation="tanh", max_iter=400, random_state=42)),
                    ]
                    model = VotingClassifier(estimators=estimators, voting="soft")
                else:
                    estimators = [
                        ("rf", RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)),
                        ("gb", GradientBoostingRegressor(n_estimators=120, max_depth=4, random_state=42)),
                    ]
                    model = VotingRegressor(estimators=estimators)

                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                m = _metrics(y_te, preds, task)
                _store("Voting Ensemble", model, m, task)

                # Individual model predictions
                ind_metrics = {}
                for name, est in model.named_estimators_.items():
                    ind_preds = est.predict(X_te)
                    ind_metrics[name.upper()] = _metrics(y_te, ind_preds, task)
                ind_metrics["Voting Ensemble"] = m

            st.markdown("### Voting Ensemble Performance")
            _show_metrics(m)

            st.markdown("#### Individual vs Ensemble Comparison")
            comp_df = pd.DataFrame(ind_metrics).T
            st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

            metric_key = list(m.keys())[0]
            vals = {k: v[metric_key] for k, v in ind_metrics.items()}
            colors = ["#94a3b8"] * (len(vals) - 1) + ["#0891b2"]
            fig_v = go.Figure(go.Bar(x=list(vals.keys()), y=list(vals.values()),
                                      marker_color=colors))
            fig_v.update_layout(title=f"Comparison — {metric_key}", height=350,
                                 template="plotly_white")
            st.plotly_chart(fig_v, use_container_width=True)

            # Sample predictions table
            st.markdown("#### Sample Predictions (first 10 test samples)")
            sample_df = pd.DataFrame({"Actual": y_te[:10]})
            for name, est in model.named_estimators_.items():
                sample_df[name.upper()] = est.predict(X_te[:10])
            sample_df["Ensemble"] = preds[:10]
            st.dataframe(sample_df, use_container_width=True)

    # ── Stacking ──
    with tab_stack:
        if st.button("🚀 Train Stacking Ensemble", type="primary", key="btn_stack"):
            with st.spinner("Training Stacking Ensemble…"):
                if task == "classification":
                    estimators = [
                        ("rf", RandomForestClassifier(n_estimators=120, max_depth=9, random_state=42)),
                        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
                        ("dt", DecisionTreeClassifier(max_depth=6, random_state=42)),
                    ]
                    model = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                        cv=3,
                    )
                else:
                    estimators = [
                        ("rf", RandomForestRegressor(n_estimators=120, max_depth=9, random_state=42)),
                        ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
                        ("dt", DecisionTreeRegressor(max_depth=6, random_state=42)),
                    ]
                    model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=Ridge(alpha=1.0),
                        cv=3,
                    )

                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                m = _metrics(y_te, preds, task)
                _store("Stacking Ensemble", model, m, task)

                # Individual
                ind_metrics = {}
                for name, est in model.named_estimators_.items():
                    ind_preds = est.predict(X_te)
                    ind_metrics[name.upper()] = _metrics(y_te, ind_preds, task)
                ind_metrics["Stacking Ensemble"] = m

            st.markdown("### Stacking Ensemble Performance")
            _show_metrics(m)

            st.markdown("#### Base Models vs Stacking Comparison")
            comp_df = pd.DataFrame(ind_metrics).T
            st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)

            metric_key = list(m.keys())[0]
            vals = {k: v[metric_key] for k, v in ind_metrics.items()}
            colors = ["#94a3b8"] * (len(vals) - 1) + ["#16a34a"]
            fig_s = go.Figure(go.Bar(x=list(vals.keys()), y=list(vals.values()),
                                      marker_color=colors))
            fig_s.update_layout(title=f"Comparison — {metric_key}", height=350,
                                 template="plotly_white")
            st.plotly_chart(fig_s, use_container_width=True)

    st.markdown(
        '<div class="success-box">📝 <b>Interpretation:</b> '
        "Voting combines predictions by majority rule or averaging, which is simple but "
        "effective. Stacking goes further by training a meta-learner to optimally combine "
        "base model outputs. Ensembles often outperform any single model by leveraging "
        "the strengths of diverse algorithms.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════

def page_comparison():
    st.markdown("# 📊 Model Comparison")

    results = st.session_state.get("model_results", {})
    if not results:
        st.markdown(
            '<div class="warning-box">⚠️ No models have been trained yet. '
            "Please train at least one model first.</div>",
            unsafe_allow_html=True,
        )
        return

    # Separate by task type
    clf_results = {k: v for k, v in results.items() if v["task_type"] == "classification"}
    reg_results = {k: v for k, v in results.items() if v["task_type"] == "regression"}

    if clf_results:
        st.markdown("### Classification Models")
        clf_df = pd.DataFrame({k: v["metrics"] for k, v in clf_results.items()}).T
        clf_df = clf_df.sort_values("Accuracy", ascending=False)

        st.dataframe(clf_df.style.format("{:.4f}").background_gradient(cmap="YlGnBu"),
                      use_container_width=True)

        best = clf_df.index[0]
        st.markdown(f'<div class="success-box">🏆 <b>Best Classification Model:</b> {best} '
                     f'(Accuracy: {clf_df.loc[best, "Accuracy"]:.4f})</div>',
                     unsafe_allow_html=True)

        fig = go.Figure()
        for metric in clf_df.columns:
            fig.add_bar(name=metric, x=clf_df.index.tolist(), y=clf_df[metric].tolist())
        fig.update_layout(barmode="group", height=450, template="plotly_white",
                           title="Classification Models — Performance Comparison",
                           xaxis_title="Model", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    if reg_results:
        st.markdown("### Regression Models")
        reg_df = pd.DataFrame({k: v["metrics"] for k, v in reg_results.items()}).T
        reg_df = reg_df.sort_values("R²", ascending=False)

        st.dataframe(reg_df.style.format("{:.4f}").background_gradient(cmap="YlGnBu"),
                      use_container_width=True)

        best = reg_df.index[0]
        st.markdown(f'<div class="success-box">🏆 <b>Best Regression Model:</b> {best} '
                     f'(R²: {reg_df.loc[best, "R²"]:.4f})</div>',
                     unsafe_allow_html=True)

        # Separate charts for MAE/RMSE and R²
        col1, col2 = st.columns(2)
        with col1:
            fig_err = go.Figure()
            for metric in ["MAE", "RMSE"]:
                if metric in reg_df.columns:
                    fig_err.add_bar(name=metric, x=reg_df.index.tolist(),
                                     y=reg_df[metric].tolist())
            fig_err.update_layout(barmode="group", height=400, template="plotly_white",
                                    title="Error Metrics (lower is better)")
            st.plotly_chart(fig_err, use_container_width=True)

        with col2:
            if "R²" in reg_df.columns:
                fig_r2 = px.bar(x=reg_df.index.tolist(), y=reg_df["R²"].tolist(),
                                 color=reg_df["R²"].tolist(), color_continuous_scale="Teal",
                                 labels={"x": "Model", "y": "R²"})
                fig_r2.update_layout(height=400, template="plotly_white",
                                      title="R² Score (higher is better)", showlegend=False)
                st.plotly_chart(fig_r2, use_container_width=True)

    if not clf_results and not reg_results:
        st.info("Train some models to see the comparison.")


# ══════════════════════════════════════════════════════════════════
#  PAGE: TEST PREDICTION
# ══════════════════════════════════════════════════════════════════

def page_prediction():
    st.markdown("# 🔮 Test Prediction")
    st.markdown("Enter wastewater measurements below and predict the pollution outcome.")

    results = st.session_state.get("model_results", {})
    if not results:
        st.markdown(
            '<div class="warning-box">⚠️ No models trained. '
            "Train at least one model first.</div>",
            unsafe_allow_html=True,
        )
        return

    model_name = st.selectbox("Select Model for Prediction", list(results.keys()))
    model_info = results[model_name]
    model = model_info["model"]
    task = model_info["task_type"]

    st.markdown("### Enter Feature Values")
    feature_cols = st.session_state.feature_cols
    df = st.session_state.df

    input_vals = {}
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    for r in range(n_rows):
        cols = st.columns(n_cols)
        for ci in range(n_cols):
            fi = r * n_cols + ci
            if fi < len(feature_cols):
                feat = feature_cols[fi]
                med = float(df[feat].median()) if feat in df.columns else 0.0
                min_v = float(df[feat].min()) if feat in df.columns else 0.0
                max_v = float(df[feat].max()) if feat in df.columns else 100.0
                input_vals[feat] = cols[ci].number_input(
                    feat, value=med, min_value=min_v * 0.5, max_value=max_v * 2.0,
                    key=f"pred_{feat}",
                )

    if st.button("🔮 Predict", type="primary"):
        try:
            # Prepare input
            input_arr = np.array([[input_vals[f] for f in feature_cols]])
            scaler = st.session_state.get("scaler")
            if scaler is not None:
                input_arr = scaler.transform(input_arr)

            prediction = model.predict(input_arr)

            st.markdown("---")
            st.markdown("### Prediction Result")

            if task == "classification":
                le = st.session_state.get("label_encoder")
                if le is not None:
                    pred_label = le.inverse_transform(prediction)[0]
                else:
                    pred_label = prediction[0]

                # Color coding
                color_map = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#ef4444"}
                color = color_map.get(str(pred_label), "#0891b2")
                st.markdown(
                    f'<div style="text-align:center; padding:2rem; background:{color}20; '
                    f'border:2px solid {color}; border-radius:16px; margin:1rem 0;">'
                    f'<div style="font-size:1rem; color:#64748b;">Predicted Pollution Level</div>'
                    f'<div style="font-size:2.5rem; font-weight:700; color:{color};">{pred_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Show probabilities if available
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(input_arr)[0]
                        classes = le.classes_ if le is not None else list(range(len(proba)))
                        st.markdown("#### Prediction Confidence")
                        prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
                        fig_p = px.bar(prob_df, x="Class", y="Probability",
                                        color="Probability", color_continuous_scale="Teal")
                        fig_p.update_layout(height=300, template="plotly_white", showlegend=False)
                        st.plotly_chart(fig_p, use_container_width=True)
                    except Exception:
                        pass
            else:
                pred_val = prediction[0]
                target_col = st.session_state.get("target_col", "Target")
                st.markdown(
                    f'<div style="text-align:center; padding:2rem; background:#0891b220; '
                    f'border:2px solid #0891b2; border-radius:16px; margin:1rem 0;">'
                    f'<div style="font-size:1rem; color:#64748b;">Predicted {target_col}</div>'
                    f'<div style="font-size:2.5rem; font-weight:700; color:#0891b2;">{pred_val:.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Input summary
            with st.expander("📋 Input Values Summary"):
                st.dataframe(pd.DataFrame([input_vals]), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🌊 AquaPredict")
        st.markdown("*Wastewater Pollution Prediction*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "🏠 Overview",
                "📊 Dataset & Preprocessing",
                "🌳 CART Decision Tree",
                "🎒 Bagging",
                "🌲 Random Forest",
                "🚀 AdaBoost",
                "📈 Gradient Boosting",
                "🧠 ABM",
                "📐 GAM",
                "🤝 Ensemble Learning",
                "📊 Model Comparison",
                "🔮 Test Prediction",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("##### 📋 Quick Status")
        df = _load_data()
        st.markdown(f"📁 Data: **{len(df)} rows**")
        ready = "X_train" in st.session_state
        st.markdown(f"{'✅' if ready else '❌'} Prepared: **{'Yes' if ready else 'No'}**")
        n_models = len(st.session_state.get("model_results", {}))
        st.markdown(f"🤖 Models: **{n_models}** trained")

    pages = {
        "🏠 Overview": page_overview,
        "📊 Dataset & Preprocessing": page_dataset,
        "🌳 CART Decision Tree": page_cart,
        "🎒 Bagging": page_bagging,
        "🌲 Random Forest": page_random_forest,
        "🚀 AdaBoost": page_adaboost,
        "📈 Gradient Boosting": page_gradient_boosting,
        "🧠 ABM": page_abm,
        "📐 GAM": page_gam,
        "🤝 Ensemble Learning": page_ensemble,
        "📊 Model Comparison": page_comparison,
        "🔮 Test Prediction": page_prediction,
    }

    pages[page]()


if __name__ == "__main__":
    main()
