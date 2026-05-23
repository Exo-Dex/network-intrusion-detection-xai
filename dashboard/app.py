"""
app.py — NIDS-XAI Attack Interpretation Dashboard
===================================================
Network Intrusion Detection using ML with Explainable AI
IEEE CS Bangalore Chapter — SIMP 2026
Team: Piyush M. Borkar, Varun Gada

Run from project root:
    streamlit run dashboard/app.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'results', 'models')
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'processed')

# ─────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NIDS · XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  Theme State
# ─────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'instance_idx' not in st.session_state:
    st.session_state.instance_idx = 0

T = st.session_state.theme

# ─────────────────────────────────────────
#  Design Tokens
# ─────────────────────────────────────────
DARK = {
    'bg'         : '#0d1117',
    'sidebar_bg' : '#161b22',
    'card_bg'    : '#1c2333',
    'border'     : '#30363d',
    'text'       : '#e6edf3',
    'subtext'    : '#8b949e',
    'accent'     : '#58a6ff',
    'accent2'    : '#1f6feb',
    'attack'     : '#f85149',
    'normal'     : '#3fb950',
    'warning'    : '#d29922',
    'mpl_bg'     : '#1c2333',
    'mpl_ax'     : '#0d1117',
    'mpl_text'   : '#e6edf3',
    'mpl_grid'   : '#30363d',
}

LIGHT = {
    'bg'         : '#f5f0eb',
    'sidebar_bg' : '#ede8e3',
    'card_bg'    : '#faf7f4',
    'border'     : '#d9d0c7',
    'text'       : '#1a1a2e',
    'subtext'    : '#6b6b7b',
    'accent'     : '#1a73e8',
    'accent2'    : '#0d47a1',
    'attack'     : '#d32f2f',
    'normal'     : '#2e7d32',
    'warning'    : '#f57c00',
    'mpl_bg'     : '#faf7f4',
    'mpl_ax'     : '#f5f0eb',
    'mpl_text'   : '#1a1a2e',
    'mpl_grid'   : '#d9d0c7',
}

C = DARK if T == 'dark' else LIGHT

# ─────────────────────────────────────────
#  CSS Injection
# ─────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Base */
    [data-testid="stAppViewContainer"] {{
        background-color: {C['bg']};
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {C['sidebar_bg']};
        border-right: 1px solid {C['border']};
    }}
    [data-testid="stHeader"] {{
        background-color: {C['bg']};
    }}

    /* Text */
    html, body, [class*="css"], p, div, span, label {{
        color: {C['text']} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background: {C['card_bg']};
        border: 1px solid {C['border']};
        border-radius: 12px;
        padding: 16px 20px;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: {C['accent']} !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-size: 0.78rem !important;
        color: {C['subtext']} !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    /* Selectbox, radio */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stRadio"] > div {{
        background: {C['card_bg']};
        border: 1px solid {C['border']};
        border-radius: 8px;
    }}

    /* Buttons */
    [data-testid="stButton"] > button {{
        background: {C['accent2']};
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
        transition: all 0.2s ease;
    }}
    [data-testid="stButton"] > button:hover {{
        background: {C['accent']};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px {C['accent']}44;
    }}

    /* Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        background: {C['card_bg']};
        border-radius: 10px;
        border: 1px solid {C['border']};
        padding: 4px;
        gap: 4px;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"] {{
        border-radius: 8px;
        color: {C['subtext']} !important;
        font-weight: 500;
    }}
    [data-testid="stTabs"] [aria-selected="true"] {{
        background: {C['accent2']} !important;
        color: #ffffff !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid {C['border']};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Divider */
    hr {{
        border-color: {C['border']} !important;
        margin: 24px 0;
    }}

    /* Custom components */
    .page-title {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {C['text']};
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }}
    .page-title span {{
        color: {C['accent']} !important;
    }}
    .page-subtitle {{
        font-size: 0.88rem;
        color: {C['subtext']};
        margin-bottom: 24px;
    }}
    .card {{
        background: {C['card_bg']};
        border: 1px solid {C['border']};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }}
    .card-accent {{
        border-left: 3px solid {C['accent']};
    }}
    .result-attack {{
        background: {C['attack']}18;
        border: 1.5px solid {C['attack']};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }}
    .result-normal {{
        background: {C['normal']}18;
        border: 1.5px solid {C['normal']};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }}
    .badge-attack {{
        background: {C['attack']};
        color: #ffffff !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.05em;
    }}
    .badge-normal {{
        background: {C['normal']};
        color: #ffffff !important;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.05em;
    }}
    .mono {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem;
        color: {C['accent']} !important;
    }}
    .label-sm {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {C['subtext']};
        font-weight: 600;
    }}
    .sidebar-logo {{
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: {C['text']};
    }}
    .sidebar-logo span {{
        color: {C['accent']} !important;
    }}
    .nav-badge {{
        background: {C['accent2']};
        color: white !important;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 6px;
    }}
    .info-row {{
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid {C['border']};
        font-size: 0.88rem;
    }}
    .info-row:last-child {{ border-bottom: none; }}
    .info-key {{ color: {C['subtext']}; }}
    .info-val {{ color: {C['text']}; font-weight: 600; font-family: 'JetBrains Mono', monospace; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Matplotlib Theme Helper
# ─────────────────────────────────────────
def apply_mpl_theme(fig, ax_list=None):
    fig.patch.set_facecolor(C['mpl_bg'])
    axes = ax_list or fig.get_axes()
    if not isinstance(axes, list): axes = [axes]
    for ax in axes:
        ax.set_facecolor(C['mpl_ax'])
        ax.tick_params(colors=C['mpl_text'])
        ax.xaxis.label.set_color(C['mpl_text'])
        ax.yaxis.label.set_color(C['mpl_text'])
        ax.title.set_color(C['mpl_text'])
        for spine in ax.spines.values():
            spine.set_edgecolor(C['border'])
        ax.grid(True, color=C['mpl_grid'], linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
    return fig

# ─────────────────────────────────────────
#  Data & Model Loading
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train_cleaned.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test_cleaned.csv'))
    return train, test

@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'),        'rb') as f: scaler    = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f: le        = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_cols.pkl'),  'rb') as f: feat_cols = pickle.load(f)
    return scaler, le, feat_cols

@st.cache_resource
def load_ml_models():
    names  = ['random_forest', 'xgboost', 'decision_tree', 'logistic_regression']
    labels = ['Random Forest', 'XGBoost', 'Decision Tree', 'Logistic Regression']
    models = {}
    for n, l in zip(names, labels):
        try:
            with open(os.path.join(MODEL_DIR, f'{n}_binary.pkl'), 'rb') as f:
                models[n] = {'model': pickle.load(f), 'label': l}
        except Exception:
            pass
    return models

@st.cache_data
def compute_metrics(_models, _X_test, _y_test):
    rows = []
    for key, obj in _models.items():
        y_pred = obj['model'].predict(_X_test)
        rows.append({
            'Model'     : obj['label'],
            'Accuracy'  : round(accuracy_score(_y_test, y_pred)*100, 2),
            'Precision' : round(precision_score(_y_test, y_pred, zero_division=0)*100, 2),
            'Recall'    : round(recall_score(_y_test, y_pred, zero_division=0)*100, 2),
            'F1 Score'  : round(f1_score(_y_test, y_pred, zero_division=0)*100, 2),
        })
    return pd.DataFrame(rows).set_index('Model')

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# ─────────────────────────────────────────
#  Load
# ─────────────────────────────────────────
try:
    df_train, df_test = load_data()
    scaler, le, FEATURE_COLS = load_artifacts()
    DROP_COLS    = ['label', 'attack_category', 'binary_label']
    X_train_raw  = df_train[FEATURE_COLS].values
    X_test_raw   = df_test[FEATURE_COLS].values
    y_test_bin   = df_test['binary_label'].values
    X_train      = scaler.transform(X_train_raw)
    X_test       = scaler.transform(X_test_raw)
    ml_models    = load_ml_models()
    df_metrics   = compute_metrics(ml_models, X_test, y_test_bin)
    DATA_LOADED  = True
except Exception as e:
    DATA_LOADED  = False
    LOAD_ERROR   = str(e)

# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-logo">NIDS<span>·</span>XAI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-size:0.75rem;color:{C["subtext"]};margin-bottom:20px;">'
        f'IEEE CS Bangalore · SIMP 2026</div>',
        unsafe_allow_html=True
    )

    # Theme toggle
    theme_label = "☀️  Light mode" if T == 'dark' else "🌑  Dark mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = 'light' if T == 'dark' else 'dark'
        st.rerun()

    st.markdown("---")

    # Navigation
    st.markdown(
        f'<div class="label-sm" style="margin-bottom:10px;">Navigation</div>',
        unsafe_allow_html=True
    )
    page = st.radio("", [
        "🔍  Predict & Explain",
        "📊  Model Performance",
        "🧠  XAI Deep Dive",
        "🏠  Overview",
    ], label_visibility="collapsed")

    st.markdown("---")

    if DATA_LOADED:
        st.markdown(
            f'<div class="card" style="padding:14px 16px;">'
            f'<div class="label-sm" style="margin-bottom:10px;">Dataset</div>'
            f'<div class="info-row"><span class="info-key">Train</span><span class="info-val">{len(df_train):,}</span></div>'
            f'<div class="info-row"><span class="info-key">Test</span><span class="info-val">{len(df_test):,}</span></div>'
            f'<div class="info-row"><span class="info-key">Features</span><span class="info-val">{len(FEATURE_COLS)}</span></div>'
            f'<div class="info-row"><span class="info-key">Models</span><span class="info-val">{len(ml_models)}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-size:0.75rem;color:{C["normal"]};margin-top:8px;">● Systems operational</div>',
            unsafe_allow_html=True
        )
    else:
        st.error("Models not loaded. Run notebooks first.")

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.78rem;color:{C["subtext"]};">'
        f'<b style="color:{C["text"]};">Team</b><br>'
        f'Piyush M. Borkar<br>'
        f'<span style="font-size:0.72rem;">Team Lead</span><br><br>'
        f'Varun Gada<br>'
        f'<span style="font-size:0.72rem;">Data & Visualization</span>'
        f'</div>',
        unsafe_allow_html=True
    )

if not DATA_LOADED:
    st.error(f"Failed to load data/models.\n\n```\n{LOAD_ERROR}\n```\n\nMake sure all notebooks have been run first.")
    st.stop()


# ══════════════════════════════════════════════════════
#  PAGE 1 — PREDICT & EXPLAIN  (default)
# ══════════════════════════════════════════════════════
if page == "🔍  Predict & Explain":
    st.markdown('<div class="page-title">🔍 Predict <span>&</span> Explain</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select an instance, pick a model, and get a full AI-powered explanation.</div>', unsafe_allow_html=True)

    # ── Controls row ──
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1.6])

    with ctrl1:
        st.markdown('<div class="label-sm">Model</div>', unsafe_allow_html=True)
        model_key = st.selectbox("", list(ml_models.keys()),
                                  format_func=lambda x: ml_models[x]['label'],
                                  label_visibility="collapsed", key="pred_model")

    with ctrl2:
        st.markdown('<div class="label-sm">Instance</div>', unsafe_allow_html=True)
        instance_mode = st.selectbox("", ["Random", "Manual Index"],
                                      label_visibility="collapsed", key="inst_mode")

    with ctrl3:
        if instance_mode == "Random":
            st.markdown('<div class="label-sm">Filter</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns([1.5, 1])
            with col_a:
                attack_only = st.checkbox("Attack instances only", value=False)
            with col_b:
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                if st.button("🎲 Randomize", use_container_width=True):
                    pool = np.where(y_test_bin == 1)[0] if attack_only else np.arange(len(X_test))
                    st.session_state.instance_idx = int(np.random.choice(pool))
        else:
            st.markdown('<div class="label-sm">Index (0 – {:,})</div>'.format(len(X_test)-1), unsafe_allow_html=True)
            manual_idx = st.number_input("", min_value=0, max_value=len(X_test)-1,
                                          value=st.session_state.instance_idx,
                                          label_visibility="collapsed")
            st.session_state.instance_idx = int(manual_idx)

    st.markdown("---")

    idx        = st.session_state.instance_idx
    model_obj  = ml_models[model_key]['model']
    instance   = X_test[idx].reshape(1, -1)
    pred       = int(model_obj.predict(instance)[0])
    proba      = model_obj.predict_proba(instance)[0]
    true_label = int(y_test_bin[idx])
    conf       = round(proba[pred]*100, 1)
    correct    = pred == true_label

    # ── Result card ──
    card_class  = "result-attack" if pred == 1 else "result-normal"
    badge_class = "badge-attack"  if pred == 1 else "badge-normal"
    pred_str    = "ATTACK" if pred == 1 else "NORMAL"
    true_str    = "ATTACK" if true_label == 1 else "NORMAL"
    icon        = "🔴" if pred == 1 else "🟢"
    verdict     = f'✅ Correct' if correct else f'❌ Incorrect'

    res_col1, res_col2 = st.columns([1.6, 2.4])

    with res_col1:
        st.markdown(
            f'<div class="{card_class}">'
            f'<div class="label-sm" style="margin-bottom:10px;">Prediction Result</div>'
            f'<div style="font-size:2.2rem;margin-bottom:6px;">{icon}</div>'
            f'<span class="{badge_class}">{pred_str}</span>'
            f'<div style="margin-top:14px;font-size:0.85rem;">'
            f'<span style="color:{C["subtext"]};">Confidence: </span>'
            f'<span class="mono" style="font-size:1rem;">{conf}%</span>'
            f'</div>'
            f'<div style="margin-top:6px;font-size:0.85rem;">'
            f'<span style="color:{C["subtext"]};">True label: </span>'
            f'<b>{true_str}</b>'
            f'&nbsp;&nbsp;{verdict}'
            f'</div>'
            f'<div style="margin-top:6px;font-size:0.8rem;color:{C["subtext"]};">'
            f'Instance #{idx} &nbsp;·&nbsp; {ml_models[model_key]["label"]}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with res_col2:
        st.markdown('<div class="label-sm" style="margin-bottom:10px;">Class Probabilities</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("P(Normal)", f"{proba[0]*100:.1f}%",
                  delta=f"{'↑ ' if proba[0]>0.5 else '↓ '}{abs(proba[0]-0.5)*100:.1f}% from baseline")
        m2.metric("P(Attack)", f"{proba[1]*100:.1f}%",
                  delta=f"{'↑ ' if proba[1]>0.5 else '↓ '}{abs(proba[1]-0.5)*100:.1f}% from baseline")

        # Confidence bar
        fig, ax = plt.subplots(figsize=(6, 1.0))
        apply_mpl_theme(fig, [ax])
        ax.barh([''], [proba[0]*100], color=C['normal'], height=0.5)
        ax.barh([''], [proba[1]*100], left=[proba[0]*100], color=C['attack'], height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Confidence (%)', fontsize=8)
        ax.axvline(50, color=C['border'], linewidth=1, linestyle='--')
        ax.tick_params(axis='y', left=False)
        normal_p = mpatches.Patch(color=C['normal'], label=f'Normal {proba[0]*100:.1f}%')
        attack_p = mpatches.Patch(color=C['attack'], label=f'Attack {proba[1]*100:.1f}%')
        ax.legend(handles=[normal_p, attack_p], fontsize=7,
                  facecolor=C['card_bg'], edgecolor=C['border'],
                  labelcolor=C['mpl_text'], loc='upper right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── SHAP Explanation ──
    if model_key in ['random_forest', 'xgboost', 'decision_tree']:
        st.markdown('<div class="label-sm" style="margin-bottom:12px;">💡 SHAP Explanation — Why this prediction?</div>', unsafe_allow_html=True)

        with st.spinner("Computing SHAP values..."):
            explainer = get_explainer(model_obj)
            sv        = explainer.shap_values(instance)

            if isinstance(sv, list):
                sv_single = sv[1][0]
                base_val  = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                sv_single = sv[0, :, 1] if sv.ndim == 3 else sv[0]
                base_val  = explainer.expected_value

            # Fix: safely extract scalar from base_val
            base_val = float(np.array(base_val).flatten()[0])

        shap_exp = shap.Explanation(
            values        = sv_single,
            base_values   = float(np.array(base_val).flatten()[0]),
            data          = instance[0],
            feature_names = FEATURE_COLS
        )

        shap_col1, shap_col2 = st.columns([1.5, 1])

        with shap_col1:
            fig, ax = plt.subplots(figsize=(9, 6))
            apply_mpl_theme(fig, [ax])
            shap.plots.waterfall(shap_exp, max_display=12, show=False)
            plt.title(f'Feature Contributions → {pred_str}',
                      color=C['mpl_text'], fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with shap_col2:
            st.markdown('<div class="label-sm" style="margin-bottom:10px;">Top Contributing Features</div>', unsafe_allow_html=True)
            top_n  = 12
            order  = np.argsort(np.abs(sv_single))[::-1][:top_n]
            feat_df = pd.DataFrame({
                'Feature'     : [FEATURE_COLS[i] for i in order],
                'SHAP'        : [round(sv_single[i], 4) for i in order],
                'Direction'   : ['🔴 Attack' if sv_single[i]>0 else '🟢 Normal' for i in order],
            })
            st.dataframe(feat_df, use_container_width=True, height=380,
                         hide_index=True)
    else:
        st.info("SHAP explanations available for tree-based models (Random Forest, XGBoost, Decision Tree).")


# ══════════════════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown('<div class="page-title">📊 Model <span>Performance</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evaluation metrics across all trained models — binary classification.</div>', unsafe_allow_html=True)

    best_model = df_metrics['F1 Score'].idxmax()
    best_f1    = df_metrics['F1 Score'].max()
    best_acc   = df_metrics['Accuracy'].max()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best F1 Score",  f"{best_f1}%",  help="Weighted F1 for best model")
    k2.metric("Best Accuracy",  f"{best_acc}%", help="Best accuracy across all models")
    k3.metric("Models Trained", f"{len(df_metrics)}")
    k4.metric("Test Samples",   f"{len(X_test):,}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋  Metrics Table", "📈  Visual Comparison", "🔲  Confusion Matrix"])

    # ── Tab 1: Table ──
    with tab1:
        st.markdown(f'<div class="label-sm" style="margin-bottom:10px;">🏆 Best model: <b>{best_model}</b> — F1 {best_f1}%</div>', unsafe_allow_html=True)
        styled = df_metrics.style\
            .highlight_max(color='#2dc65330' if T=='dark' else '#a5d6a730', axis=0)\
            .format("{:.2f}")\
            .set_properties(**{
                'font-family': 'JetBrains Mono, monospace',
                'font-size'  : '0.88rem',
            })
        st.dataframe(styled, use_container_width=True)

    # ── Tab 2: Visual ──
    with tab2:
        metric_choice = st.selectbox("Metric", ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                      key='metric_compare')
        vals   = df_metrics[metric_choice]
        colors = [C['accent'] if m == best_model else C['border'] for m in df_metrics.index]

        fig, ax = plt.subplots(figsize=(10, 5))
        apply_mpl_theme(fig, [ax])
        bars = ax.bar(df_metrics.index, vals, color=colors, edgecolor=C['bg'],
                      linewidth=0.5, width=0.5)
        ax.set_ylim(max(0, vals.min()-5), 101)
        ax.set_ylabel(f'{metric_choice} (%)', fontsize=10)
        ax.set_title(f'{metric_choice} Comparison — All Models',
                     color=C['mpl_text'], fontweight='bold', pad=12)
        ax.tick_params(axis='x', rotation=10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}%',
                    ha='center', fontsize=9,
                    color=C['accent'] if val == vals.max() else C['mpl_text'],
                    fontweight='bold' if val == vals.max() else 'normal',
                    fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Radar chart
        st.markdown('<div class="label-sm" style="margin:16px 0 10px;">Radar — Overall Model Profile</div>', unsafe_allow_html=True)
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        angles = np.linspace(0, 2*np.pi, len(metrics_cols), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_facecolor(C['mpl_ax'])
        fig.patch.set_facecolor(C['mpl_bg'])
        ax.tick_params(colors=C['mpl_text'])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_cols, color=C['mpl_text'], size=9)
        ax.yaxis.set_tick_params(labelcolor=C['subtext'])
        ax.set_rlabel_position(30)
        ax.set_ylim(80, 100)
        ax.grid(color=C['mpl_grid'], linewidth=0.5)
        ax.spines['polar'].set_edgecolor(C['border'])

        palette = [C['accent'], C['normal'], C['warning'], C['attack']]
        for i, (model_name, row) in enumerate(df_metrics.iterrows()):
            vals_r = row[metrics_cols].tolist()
            vals_r += vals_r[:1]
            ax.plot(angles, vals_r, linewidth=2, color=palette[i%len(palette)], label=model_name)
            ax.fill(angles, vals_r, alpha=0.08, color=palette[i%len(palette)])

        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
                  facecolor=C['card_bg'], edgecolor=C['border'],
                  labelcolor=C['mpl_text'], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: Confusion Matrix ──
    with tab3:
        cm_model_key = st.selectbox("Model", list(ml_models.keys()),
                                     format_func=lambda x: ml_models[x]['label'],
                                     key='cm_model')
        y_pred_cm = ml_models[cm_model_key]['model'].predict(X_test)
        cm        = confusion_matrix(y_test_bin, y_pred_cm)
        tn, fp, fn, tp = cm.ravel()

        cm_col1, cm_col2 = st.columns([1.2, 1])

        with cm_col1:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            apply_mpl_theme(fig, [ax])
            cmap = 'Blues' if T == 'light' else 'YlOrRd'
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                        xticklabels=['Normal', 'Attack'],
                        yticklabels=['Normal', 'Attack'],
                        linewidths=0.5, linecolor=C['border'],
                        annot_kws={'size':14, 'weight':'bold',
                                   'color':C['mpl_text']},
                        ax=ax, cbar_kws={'shrink':0.8})
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title(f'{ml_models[cm_model_key]["label"]}',
                         fontweight='bold', pad=12)
            ax.tick_params(colors=C['mpl_text'])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with cm_col2:
            fpr = round(fp/(fp+tn)*100, 2) if (fp+tn) > 0 else 0
            fnr = round(fn/(fn+tp)*100, 2) if (fn+tp) > 0 else 0
            st.markdown(
                f'<div class="card card-accent">'
                f'<div class="label-sm" style="margin-bottom:14px;">Breakdown</div>'
                f'<div class="info-row"><span class="info-key">✅ True Negatives</span><span class="info-val">{tn:,}</span></div>'
                f'<div class="info-row"><span class="info-key">✅ True Positives</span><span class="info-val">{tp:,}</span></div>'
                f'<div class="info-row"><span class="info-key">❌ False Positives</span><span class="info-val" style="color:{C["attack"]};">{fp:,}</span></div>'
                f'<div class="info-row"><span class="info-key">❌ False Negatives</span><span class="info-val" style="color:{C["attack"]};">{fn:,}</span></div>'
                f'<div style="margin-top:16px;"></div>'
                f'<div class="info-row"><span class="info-key">False Positive Rate</span><span class="info-val">{fpr}%</span></div>'
                f'<div class="info-row"><span class="info-key">False Negative Rate</span><span class="info-val">{fnr}%</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════
#  PAGE 3 — XAI DEEP DIVE
# ══════════════════════════════════════════════════════
elif page == "🧠  XAI Deep Dive":
    st.markdown('<div class="page-title">🧠 XAI <span>Deep Dive</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Global and local explainability using SHAP and LIME.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌍  Global SHAP", "🔬  Local SHAP", "🟡  LIME"])

    # ── Tab 1: Global SHAP ──
    with tab1:
        g_model_key = st.selectbox("Model",
                                    [k for k in ml_models if k in ['random_forest','xgboost','decision_tree']],
                                    format_func=lambda x: ml_models[x]['label'],
                                    key='g_shap_model')
        sample_n = st.slider("Sample size", 100, 500, 300, 50)

        with st.spinner("Computing global SHAP values..."):
            np.random.seed(42)
            sidx   = np.random.choice(len(X_test), sample_n, replace=False)
            X_s    = X_test[sidx]
            g_exp  = get_explainer(ml_models[g_model_key]['model'])
            sv_g   = g_exp.shap_values(X_s)
            if isinstance(sv_g, list): sv_g = sv_g[1]
            if sv_g.ndim == 3:         sv_g = sv_g[:, :, 1]

        g1, g2 = st.columns(2)

        with g1:
            st.markdown('<div class="label-sm" style="margin-bottom:8px;">Beeswarm — Feature Impact</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 6))
            apply_mpl_theme(fig, [ax])
            shap.summary_plot(sv_g, X_s, feature_names=FEATURE_COLS,
                              plot_type='dot', max_display=15, show=False)
            plt.title('Global SHAP — Beeswarm', color=C['mpl_text'], fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with g2:
            st.markdown('<div class="label-sm" style="margin-bottom:8px;">Mean |SHAP| — Feature Importance</div>', unsafe_allow_html=True)
            mean_shap = pd.Series(np.abs(sv_g).mean(axis=0), index=FEATURE_COLS).nlargest(15)
            fig, ax   = plt.subplots(figsize=(7, 6))
            apply_mpl_theme(fig, [ax])
            bars = ax.barh(range(len(mean_shap)), mean_shap.values[::-1],
                           color=C['accent'], edgecolor=C['bg'], linewidth=0.5)
            ax.set_yticks(range(len(mean_shap)))
            ax.set_yticklabels(mean_shap.index[::-1], fontsize=8)
            ax.set_xlabel('Mean |SHAP value|', fontsize=9)
            ax.set_title('Feature Importance (SHAP)', color=C['mpl_text'], fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown('<div class="label-sm" style="margin-bottom:10px;">Top 20 Features — Ranked</div>', unsafe_allow_html=True)
        rank_df = pd.DataFrame({
            'Feature'      : FEATURE_COLS,
            'Mean |SHAP|'  : np.abs(sv_g).mean(axis=0),
            'Max |SHAP|'   : np.abs(sv_g).max(axis=0),
            'Std |SHAP|'   : np.abs(sv_g).std(axis=0),
        }).sort_values('Mean |SHAP|', ascending=False).head(20).reset_index(drop=True)
        rank_df.index += 1  # start rank from 1
        rank_df[['Mean |SHAP|','Max |SHAP|','Std |SHAP|']] = rank_df[['Mean |SHAP|','Max |SHAP|','Std |SHAP|']].round(4)
        st.dataframe(rank_df, use_container_width=True)

    # ── Tab 2: Local SHAP ──
    with tab2:
        l_model_key = st.selectbox("Model",
                                    [k for k in ml_models if k in ['random_forest','xgboost','decision_tree']],
                                    format_func=lambda x: ml_models[x]['label'],
                                    key='l_shap_model')
        l_idx = st.number_input("Instance Index", 0, len(X_test)-1, 0, key='l_shap_idx')

        l_model    = ml_models[l_model_key]['model']
        l_instance = X_test[l_idx].reshape(1, -1)
        l_pred     = int(l_model.predict(l_instance)[0])
        l_proba    = l_model.predict_proba(l_instance)[0]
        l_true     = int(y_test_bin[l_idx])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("True Label",  "ATTACK" if l_true==1 else "NORMAL")
        m2.metric("Prediction",  "ATTACK" if l_pred==1 else "NORMAL")
        m3.metric("Confidence",  f"{l_proba[l_pred]*100:.1f}%")
        m4.metric("Verdict",     "✅ Correct" if l_pred==l_true else "❌ Wrong")

        with st.spinner("Computing local SHAP..."):
            l_exp  = get_explainer(l_model)
            sv_l   = l_exp.shap_values(l_instance)
            if isinstance(sv_l, list):
                sv_l_s  = sv_l[1][0]
                base_l  = l_exp.expected_value[1] if isinstance(l_exp.expected_value, (list, np.ndarray)) else l_exp.expected_value
            else:
                sv_l_s  = sv_l[0, :, 1] if sv_l.ndim == 3 else sv_l[0]
                base_l  = l_exp.expected_value

        shap_exp_l = shap.Explanation(
            values        = sv_l_s,
            base_values   = float(np.array(base_l).flatten()[0]),
            data          = l_instance[0],
            feature_names = FEATURE_COLS
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        apply_mpl_theme(fig, [ax])
        shap.plots.waterfall(shap_exp_l, max_display=15, show=False)
        plt.title(f'SHAP Waterfall — Instance #{l_idx} → {"ATTACK" if l_pred==1 else "NORMAL"}',
                  color=C['mpl_text'], fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: LIME ──
    with tab3:
        lime_model_key = st.selectbox("Model", list(ml_models.keys()),
                                       format_func=lambda x: ml_models[x]['label'],
                                       key='lime_model')
        lime_idx     = st.number_input("Instance Index", 0, len(X_test)-1, 0, key='lime_idx')
        lime_nfeats  = st.slider("Features to display", 5, 20, 15, key='lime_feats')

        lime_model    = ml_models[lime_model_key]['model']
        lime_instance = X_test[lime_idx]
        lime_pred     = int(lime_model.predict(lime_instance.reshape(1,-1))[0])
        lime_true     = int(y_test_bin[lime_idx])

        l1, l2, l3 = st.columns(3)
        l1.metric("True Label",  "ATTACK" if lime_true==1 else "NORMAL")
        l2.metric("Prediction",  "ATTACK" if lime_pred==1 else "NORMAL")
        l3.metric("Verdict",     "✅ Correct" if lime_pred==lime_true else "❌ Wrong")

        with st.spinner("Running LIME..."):
            lime_exp_obj = lime.lime_tabular.LimeTabularExplainer(
                training_data        = X_train,
                feature_names        = FEATURE_COLS,
                class_names          = ['Normal', 'Attack'],
                mode                 = 'classification',
                discretize_continuous= True,
                random_state         = 42
            )
            lime_result = lime_exp_obj.explain_instance(
                lime_instance,
                lime_model.predict_proba,
                num_features = lime_nfeats,
                labels       = [0, 1]
            )

        fig = lime_result.as_pyplot_figure(label=lime_pred)
        fig.patch.set_facecolor(C['mpl_bg'])
        for ax in fig.get_axes():
            ax.set_facecolor(C['mpl_ax'])
            ax.tick_params(colors=C['mpl_text'])
            ax.xaxis.label.set_color(C['mpl_text'])
            ax.title.set_color(C['mpl_text'])
        plt.suptitle(f'LIME — Instance #{lime_idx} → {"ATTACK" if lime_pred==1 else "NORMAL"}',
                     color=C['mpl_text'], fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown('<div class="label-sm" style="margin-bottom:10px;">Feature Weights</div>', unsafe_allow_html=True)
        lime_list = lime_result.as_list(label=lime_pred)
        lime_df   = pd.DataFrame(lime_list, columns=['Condition', 'Weight'])
        lime_df['Impact']  = lime_df['Weight'].apply(lambda w: '🔴 Attack' if w>0 else '🟢 Normal')
        lime_df['Weight']  = lime_df['Weight'].round(4)
        st.dataframe(lime_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
#  PAGE 4 — OVERVIEW
# ══════════════════════════════════════════════════════
elif page == "🏠  Overview":
    st.markdown('<div class="page-title">🛡️ NIDS <span>· XAI</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Network Intrusion Detection using Machine Learning with Explainable AI</div>', unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Train Samples",  f"{len(df_train):,}")
    k2.metric("Test Samples",   f"{len(df_test):,}")
    k3.metric("Features",       f"{len(FEATURE_COLS)}")
    k4.metric("ML Models",      "4")
    k5.metric("DL Models",      "2")

    st.markdown("---")

    ov1, ov2 = st.columns(2)

    with ov1:
        st.markdown('<div class="label-sm" style="margin-bottom:10px;">Attack Category Distribution</div>', unsafe_allow_html=True)
        cat_dist = df_train['attack_category'].value_counts()
        palette  = [C['normal'], C['attack'], C['warning'], C['accent'], C['subtext']]
        fig, ax  = plt.subplots(figsize=(6, 4))
        apply_mpl_theme(fig, [ax])
        bars = ax.bar(cat_dist.index, cat_dist.values,
                      color=palette[:len(cat_dist)], edgecolor=C['bg'], linewidth=0.5)
        ax.set_ylabel('Count'); ax.tick_params(axis='x', rotation=10)
        ax.set_title('Train Set — Attack Categories', fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                    f'{bar.get_height():,}', ha='center', fontsize=8,
                    color=C['mpl_text'], fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with ov2:
        st.markdown('<div class="label-sm" style="margin-bottom:10px;">Binary Split</div>', unsafe_allow_html=True)
        bin_dist = df_train['binary_label'].value_counts()
        fig, ax  = plt.subplots(figsize=(6, 4))
        apply_mpl_theme(fig, [ax])
        ax.pie(bin_dist.values, labels=['Normal', 'Attack'],
               autopct='%1.1f%%',
               colors=[C['normal'], C['attack']],
               startangle=90, explode=(0.03, 0.03),
               textprops={'color': C['mpl_text']},
               wedgeprops={'edgecolor': C['bg'], 'linewidth': 2})
        ax.set_title('Binary Class Distribution', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # Pipeline
    st.markdown('<div class="label-sm" style="margin-bottom:14px;">Project Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("🌐", "Network Traffic", "Raw input"),
        ("🧹", "Preprocessing",   "Clean · Encode · Scale"),
        ("⚙️", "Feature Eng.",    "Select · Transform"),
        ("🤖", "ML / DL Models",  "RF · XGB · DT · LR · MLP · AE"),
        ("🎯", "Prediction",      "Normal vs Attack"),
        ("💡", "XAI Layer",       "SHAP · LIME"),
        ("📊", "Dashboard",       "This interface"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, sub) in zip(cols, steps):
        col.markdown(
            f'<div class="card" style="text-align:center;padding:14px 8px;">'
            f'<div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>'
            f'<div style="font-size:0.78rem;font-weight:700;color:{C["text"]};">{title}</div>'
            f'<div style="font-size:0.68rem;color:{C["subtext"]};margin-top:4px;">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Project info
    info1, info2 = st.columns(2)

    with info1:
        st.markdown(
            f'<div class="card card-accent">'
            f'<div class="label-sm" style="margin-bottom:12px;">Dataset — NSL-KDD</div>'
            f'<div class="info-row"><span class="info-key">Source</span><span class="info-val">Kaggle</span></div>'
            f'<div class="info-row"><span class="info-key">Features</span><span class="info-val">41 network features</span></div>'
            f'<div class="info-row"><span class="info-key">Attack types</span><span class="info-val">DoS · Probe · R2L · U2R</span></div>'
            f'<div class="info-row"><span class="info-key">Train samples</span><span class="info-val">{len(df_train):,}</span></div>'
            f'<div class="info-row"><span class="info-key">Test samples</span><span class="info-val">{len(df_test):,}</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with info2:
        st.markdown(
            f'<div class="card card-accent">'
            f'<div class="label-sm" style="margin-bottom:12px;">Project Info</div>'
            f'<div class="info-row"><span class="info-key">Program</span><span class="info-val">IEEE CS SIMP 2026</span></div>'
            f'<div class="info-row"><span class="info-key">Domain</span><span class="info-val">Cybersecurity / NIDS</span></div>'
            f'<div class="info-row"><span class="info-key">Team Lead</span><span class="info-val">Piyush M. Borkar</span></div>'
            f'<div class="info-row"><span class="info-key">Member</span><span class="info-val">Varun Gada</span></div>'
            f'<div class="info-row"><span class="info-key">Status</span><span class="info-val" style="color:{C["normal"]};">In Development</span></div>'
            f'</div>',
            unsafe_allow_html=True
        )
