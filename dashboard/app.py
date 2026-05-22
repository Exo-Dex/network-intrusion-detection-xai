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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
XAI_DIR   = os.path.join(BASE_DIR, 'results', 'xai')
GRAPH_DIR = os.path.join(BASE_DIR, 'results', 'graphs')

# ─────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "NIDS-XAI Dashboard",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #1a73e8;
        margin-bottom: 10px;
    }
    .attack-badge {
        background: #ffebee;
        color: #c62828;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .normal-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Data & Model Loading (cached)
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train_cleaned.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test_cleaned.csv'))
    return train, test

@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'),       'rb') as f: scaler   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'),'rb') as f: le       = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'feature_cols.pkl'), 'rb') as f: feat_cols= pickle.load(f)
    return scaler, le, feat_cols

@st.cache_resource
def load_ml_models():
    names = ['random_forest', 'xgboost', 'decision_tree', 'logistic_regression']
    ml = {}
    for n in names:
        try:
            with open(os.path.join(MODEL_DIR, f'{n}_binary.pkl'), 'rb') as f:
                ml[n] = pickle.load(f)
        except Exception:
            pass
    return ml

@st.cache_data
def compute_all_metrics(_models, X_test, y_test_bin, X_test_multi, y_test_multi, class_names):
    bin_rows, multi_rows = [], []
    for name, model in _models.items():
        label = name.replace('_', ' ').title()
        y_b = model.predict(X_test)
        bin_rows.append({
            'Model'    : label,
            'Accuracy' : round(accuracy_score(y_test_bin, y_b)*100, 2),
            'Precision': round(precision_score(y_test_bin, y_b, zero_division=0)*100, 2),
            'Recall'   : round(recall_score(y_test_bin, y_b, zero_division=0)*100, 2),
            'F1'       : round(f1_score(y_test_bin, y_b, zero_division=0)*100, 2),
        })
    return pd.DataFrame(bin_rows).set_index('Model')

@st.cache_resource
def get_shap_explainer(_model, X_bg):
    return shap.TreeExplainer(_model)

# ─────────────────────────────────────────
#  Load everything
# ─────────────────────────────────────────
try:
    df_train, df_test = load_data()
    scaler, le, FEATURE_COLS = load_artifacts()
    DROP_COLS = ['label', 'attack_category', 'binary_label']

    X_train_raw = df_train[FEATURE_COLS].values
    X_test_raw  = df_test[FEATURE_COLS].values
    y_test_bin  = df_test['binary_label'].values
    test_mask   = df_test['attack_category'] != 'Unknown'
    y_test_multi = le.transform(df_test['attack_category'][test_mask])
    CLASS_NAMES  = le.classes_

    X_train = scaler.transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    X_test_multi = scaler.transform(X_test_raw[test_mask])

    ml_models = load_ml_models()
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    LOAD_ERROR  = str(e)

# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ NIDS-XAI")
    st.markdown("**IEEE CS Bangalore**  \nSIMP 2026")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  Model Performance",
        "🔍  Predict & Explain",
        "🧠  XAI Deep Dive",
    ])

    st.markdown("---")
    st.markdown("**Team**")
    st.markdown("Piyush M. Borkar *(Lead)*  \nVarun Gada")
    st.markdown("---")

    if DATA_LOADED:
        st.success("✅ Models loaded")
        st.info(f"Train: {len(df_train):,} samples  \nTest: {len(df_test):,} samples")
    else:
        st.error("❌ Failed to load models")

# ─────────────────────────────────────────
#  Error guard
# ─────────────────────────────────────────
if not DATA_LOADED:
    st.error(f"Could not load data or models. Make sure you have run all notebooks first.\n\n`{LOAD_ERROR}`")
    st.stop()

# ═══════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<p class="main-header">🛡️ Network Intrusion Detection Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning + Explainable AI on NSL-KDD Dataset</p>', unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples",  f"{len(df_train):,}")
    col2.metric("Test Samples",      f"{len(df_test):,}")
    col3.metric("Features",          f"{len(FEATURE_COLS)}")
    col4.metric("ML + DL Models",    "6")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Class distribution
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Binary Class Distribution")
        bin_dist = df_train['binary_label'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(bin_dist.values, labels=['Normal', 'Attack'],
               autopct='%1.1f%%', colors=['#4caf50','#f44336'],
               startangle=90, explode=(0.03, 0.03))
        ax.set_title('Train Set — Binary Labels')
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Attack Category Distribution")
        cat_dist = df_train['attack_category'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = sns.color_palette('Set2', len(cat_dist))
        bars = ax.bar(cat_dist.index, cat_dist.values, color=colors, edgecolor='white')
        ax.set_title('Train Set — Attack Categories')
        ax.set_ylabel('Count')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{bar.get_height():,}', ha='center', fontsize=8)
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Pipeline overview
    st.subheader("Project Pipeline")
    steps = ["Network Traffic", "Preprocessing", "Feature Engineering",
             "ML/DL Models", "Prediction", "XAI Layer", "Dashboard"]
    icons = ["🌐", "🧹", "⚙️", "🤖", "🎯", "💡", "📊"]
    cols = st.columns(len(steps))
    for col, step, icon in zip(cols, steps, icons):
        col.markdown(f"<div style='text-align:center; padding:10px; background:#f0f4ff; border-radius:8px;'>"
                     f"<div style='font-size:1.5rem'>{icon}</div>"
                     f"<div style='font-size:0.75rem; font-weight:600; margin-top:4px'>{step}</div>"
                     f"</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Dataset info
    st.subheader("Dataset — NSL-KDD")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | Source | Kaggle / Canadian Institute |
        | Features | 41 network traffic features |
        | Attack types | DoS, Probe, R2L, U2R |
        | Format | Tabular (.txt / .csv) |
        | Train samples | 125,973 |
        | Test samples | 22,544 |
        """)
    with col2:
        st.markdown("""
        **Attack Categories:**
        - 🔴 **DoS** — Denial of Service (high volume)
        - 🟠 **Probe** — Surveillance / port scanning
        - 🟡 **R2L** — Remote to Local exploitation
        - 🔵 **U2R** — User to Root privilege escalation
        - 🟢 **Normal** — Legitimate network traffic
        """)


# ═══════════════════════════════════════════
#  PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluation metrics across all trained models</p>', unsafe_allow_html=True)

    df_metrics = compute_all_metrics(
        ml_models, X_test, y_test_bin, X_test_multi, y_test_multi, CLASS_NAMES
    )

    # Best model highlight
    best_model = df_metrics['F1'].idxmax()
    best_f1    = df_metrics['F1'].max()
    st.success(f"🏆 Best Binary Model: **{best_model}** — F1 Score: **{best_f1}%**")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Metrics table
    st.subheader("Binary Classification — All Models")
    st.dataframe(df_metrics.style
                 .highlight_max(color='#c8e6c9', axis=0)
                 .format("{:.2f}"),
                 use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Bar chart comparison
    st.subheader("Visual Comparison")
    metric_choice = st.selectbox("Select metric to compare:", ['Accuracy', 'Precision', 'Recall', 'F1'])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#4caf50' if m == best_model else '#90caf9' for m in df_metrics.index]
    bars = ax.bar(df_metrics.index, df_metrics[metric_choice], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(f'{metric_choice} — All Binary Models', fontweight='bold')
    ax.set_ylabel(f'{metric_choice} (%)')
    ax.set_ylim(max(0, df_metrics[metric_choice].min() - 5), 101)
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Select model:", list(ml_models.keys()),
                             format_func=lambda x: x.replace('_', ' ').title())

    y_pred_cm = ml_models[cm_model].predict(X_test)
    cm = confusion_matrix(y_test_bin, y_pred_cm)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    linewidths=0.5, ax=ax)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'{cm_model.replace("_"," ").title()} — Confusion Matrix', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        tn, fp, fn, tp = cm.ravel()
        st.markdown("**Classification Breakdown:**")
        st.markdown(f"""
        | | Count |
        |--|--|
        | ✅ True Negatives (Normal → Normal) | {tn:,} |
        | ✅ True Positives (Attack → Attack) | {tp:,} |
        | ❌ False Positives (Normal → Attack) | {fp:,} |
        | ❌ False Negatives (Attack → Normal) | {fn:,} |
        """)
        fpr = round(fp / (fp + tn) * 100, 2)
        fnr = round(fn / (fn + tp) * 100, 2)
        st.markdown(f"**False Positive Rate:** {fpr}%  \n**False Negative Rate:** {fnr}%")


# ═══════════════════════════════════════════
#  PAGE 3 — PREDICT & EXPLAIN
# ═══════════════════════════════════════════
elif page == "🔍  Predict & Explain":
    st.markdown('<p class="main-header">🔍 Predict & Explain</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Select a test instance, run prediction, and get an AI explanation</p>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("⚙️ Settings")

        # Model selector
        model_choice = st.selectbox(
            "Select Model",
            list(ml_models.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Instance selector
        instance_mode = st.radio("Instance Selection", ["Random", "Manual Index"])
        if instance_mode == "Random":
            attack_only = st.checkbox("Show attack instance only", value=False)
            if st.button("🎲 Pick Random Instance", use_container_width=True):
                if attack_only:
                    idx_pool = np.where(y_test_bin == 1)[0]
                else:
                    idx_pool = np.arange(len(X_test))
                st.session_state['instance_idx'] = int(np.random.choice(idx_pool))
        else:
            manual_idx = st.number_input("Test Index", min_value=0,
                                          max_value=len(X_test)-1, value=0)
            st.session_state['instance_idx'] = int(manual_idx)

        idx = st.session_state.get('instance_idx', 0)
        st.info(f"Selected instance: **#{idx}**  \nTrue label: **{'Attack' if y_test_bin[idx] == 1 else 'Normal'}**")

    with col_right:
        st.subheader("🎯 Prediction Result")
        idx = st.session_state.get('instance_idx', 0)
        model = ml_models[model_choice]
        instance = X_test[idx].reshape(1, -1)

        pred      = model.predict(instance)[0]
        proba     = model.predict_proba(instance)[0]
        conf      = round(proba[pred] * 100, 2)
        true_label= y_test_bin[idx]

        pred_label = "Attack" if pred == 1 else "Normal"
        true_lbl   = "Attack" if true_label == 1 else "Normal"
        correct    = pred == true_label

        # Result display
        badge_class = "attack-badge" if pred == 1 else "normal-badge"
        badge_icon  = "🔴" if pred == 1 else "🟢"
        st.markdown(
            f"<div style='padding:16px; background:#f8f9fa; border-radius:10px; margin-bottom:16px;'>"
            f"<span style='font-size:0.9rem; color:#666;'>Prediction:</span><br>"
            f"<span class='{badge_class}'>{badge_icon} {pred_label}</span>"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span style='font-size:0.85rem; color:#888;'>Confidence: <b>{conf}%</b></span>"
            f"<br><br>"
            f"<span style='font-size:0.85rem;'>True Label: <b>{true_lbl}</b> &nbsp; "
            f"{'✅ Correct' if correct else '❌ Incorrect'}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Confidence bar
        c1, c2 = st.columns(2)
        c1.metric("P(Normal)", f"{proba[0]*100:.1f}%")
        c2.metric("P(Attack)", f"{proba[1]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([''], [proba[0]*100], color='#4caf50', label='Normal')
        ax.barh([''], [proba[1]*100], left=[proba[0]*100], color='#f44336', label='Attack')
        ax.set_xlim(0, 100); ax.set_xlabel('Confidence (%)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Prediction Confidence', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # SHAP Explanation
    if model_choice in ['random_forest', 'xgboost', 'decision_tree']:
        st.subheader("💡 SHAP Explanation")
        st.caption("Why did the model make this prediction?")

        with st.spinner("Computing SHAP values..."):
            explainer = get_shap_explainer(model, X_train[:200])
            sv = explainer.shap_values(instance)

            if isinstance(sv, list):
                sv_single = sv[1][0]
                base_val  = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                sv_single = sv[0] if sv.ndim == 2 else sv[0, :, 1]
                base_val  = explainer.expected_value

        shap_exp = shap.Explanation(
            values        = sv_single,
            base_values   = float(base_val),
            data          = instance[0],
            feature_names = FEATURE_COLS
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_exp, max_display=15, show=False)
        plt.title(f'SHAP Waterfall — {pred_label} Prediction', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top contributing features table
        st.subheader("📋 Top Contributing Features")
        feat_df = pd.DataFrame({
            'Feature'   : FEATURE_COLS,
            'SHAP Value': sv_single,
            'Feature Value': instance[0]
        }).reindex(pd.Series(np.abs(sv_single), index=FEATURE_COLS).nlargest(15).index)

        feat_df['Direction'] = feat_df['SHAP Value'].apply(
            lambda x: '🔴 Pushes toward Attack' if x > 0 else '🟢 Pushes toward Normal'
        )
        feat_df['SHAP Value'] = feat_df['SHAP Value'].round(4)
        feat_df['Feature Value'] = feat_df['Feature Value'].round(4)
        st.dataframe(feat_df.reset_index(drop=True), use_container_width=True)
    else:
        st.info("SHAP explanation available for tree-based models (RF, XGBoost, DT).")


# ═══════════════════════════════════════════
#  PAGE 4 — XAI DEEP DIVE
# ═══════════════════════════════════════════
elif page == "🧠  XAI Deep Dive":
    st.markdown('<p class="main-header">🧠 XAI Deep Dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Global & local explainability using SHAP and LIME</p>', unsafe_allow_html=True)

    xai_tab1, xai_tab2, xai_tab3 = st.tabs(["🌍 Global SHAP", "🔬 Local SHAP", "🟡 LIME"])

    # ── Tab 1: Global SHAP ──
    with xai_tab1:
        st.subheader("Global Feature Importance — SHAP")
        st.caption("Which features matter most across all predictions?")

        xai_model_choice = st.selectbox(
            "Select model for global SHAP",
            ['random_forest', 'xgboost', 'decision_tree'],
            format_func=lambda x: x.replace('_', ' ').title(),
            key='global_shap_model'
        )

        with st.spinner("Computing global SHAP values (sample of 300)..."):
            np.random.seed(42)
            sample_idx = np.random.choice(len(X_test), 300, replace=False)
            X_shap     = X_test[sample_idx]
            y_shap     = y_test_bin[sample_idx]

            explainer_g = get_shap_explainer(ml_models[xai_model_choice], X_train[:200])
            sv_g = explainer_g.shap_values(X_shap)

            if isinstance(sv_g, list):
                sv_g_2d = sv_g[1]
            else:
                sv_g_2d = sv_g[:, :, 1] if sv_g.ndim == 3 else sv_g

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Beeswarm Plot** — feature impact distribution")
            fig, ax = plt.subplots(figsize=(7, 6))
            shap.summary_plot(sv_g_2d, X_shap, feature_names=FEATURE_COLS,
                              plot_type='dot', max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Mean |SHAP| — Feature Importance**")
            mean_shap = pd.Series(
                np.abs(sv_g_2d).mean(axis=0),
                index=FEATURE_COLS
            ).nlargest(15)

            fig, ax = plt.subplots(figsize=(7, 6))
            mean_shap[::-1].plot(kind='barh', ax=ax,
                                  color=sns.color_palette('Set2')[0], edgecolor='white')
            ax.set_title('Top 15 Features by Mean |SHAP|', fontweight='bold')
            ax.set_xlabel('Mean |SHAP value|')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Feature table
        st.subheader("Top 20 Features — Ranked by Global Importance")
        shap_rank_df = pd.DataFrame({
            'Feature'         : FEATURE_COLS,
            'Mean |SHAP|'     : np.abs(sv_g_2d).mean(axis=0),
            'Max |SHAP|'      : np.abs(sv_g_2d).max(axis=0),
        }).sort_values('Mean |SHAP|', ascending=False).head(20).reset_index(drop=True)
        shap_rank_df.index += 1
        shap_rank_df['Mean |SHAP|'] = shap_rank_df['Mean |SHAP|'].round(4)
        shap_rank_df['Max |SHAP|']  = shap_rank_df['Max |SHAP|'].round(4)
        st.dataframe(shap_rank_df, use_container_width=True)

    # ── Tab 2: Local SHAP ──
    with xai_tab2:
        st.subheader("Local SHAP — Single Instance Explanation")

        local_idx = st.number_input("Test Instance Index", min_value=0,
                                     max_value=len(X_test)-1, value=10, key='local_shap_idx')
        local_model_choice = st.selectbox(
            "Model",
            ['random_forest', 'xgboost', 'decision_tree'],
            format_func=lambda x: x.replace('_', ' ').title(),
            key='local_shap_model'
        )

        local_model    = ml_models[local_model_choice]
        local_instance = X_test[local_idx].reshape(1, -1)
        local_pred     = local_model.predict(local_instance)[0]
        local_proba    = local_model.predict_proba(local_instance)[0]
        local_true     = y_test_bin[local_idx]

        col1, col2, col3 = st.columns(3)
        col1.metric("True Label",  "Attack" if local_true == 1 else "Normal")
        col2.metric("Prediction",  "Attack" if local_pred == 1 else "Normal")
        col3.metric("Confidence",  f"{local_proba[local_pred]*100:.1f}%")

        with st.spinner("Computing local SHAP..."):
            explainer_l = get_shap_explainer(local_model, X_train[:200])
            sv_l = explainer_l.shap_values(local_instance)

            if isinstance(sv_l, list):
                sv_l_single  = sv_l[1][0]
                base_val_l   = explainer_l.expected_value[1] if isinstance(explainer_l.expected_value, (list, np.ndarray)) else explainer_l.expected_value
            else:
                sv_l_single  = sv_l[0] if sv_l.ndim == 2 else sv_l[0, :, 1]
                base_val_l   = explainer_l.expected_value

        shap_exp_l = shap.Explanation(
            values        = sv_l_single,
            base_values   = float(base_val_l),
            data          = local_instance[0],
            feature_names = FEATURE_COLS
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_exp_l, max_display=15, show=False)
        plt.title(f'SHAP Waterfall — Instance #{local_idx}', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: LIME ──
    with xai_tab3:
        st.subheader("LIME — Local Interpretable Model-Agnostic Explanations")
        st.caption("LIME approximates the model locally with an interpretable linear model.")

        lime_idx   = st.number_input("Test Instance Index", min_value=0,
                                      max_value=len(X_test)-1, value=10, key='lime_idx')
        lime_model_key = st.selectbox(
            "Model",
            list(ml_models.keys()),
            format_func=lambda x: x.replace('_', ' ').title(),
            key='lime_model'
        )
        num_features = st.slider("Number of features to show", 5, 20, 15)

        lime_model    = ml_models[lime_model_key]
        lime_instance = X_test[lime_idx]
        lime_pred     = lime_model.predict(lime_instance.reshape(1,-1))[0]
        lime_true     = y_test_bin[lime_idx]

        col1, col2 = st.columns(2)
        col1.metric("True Label", "Attack" if lime_true == 1 else "Normal")
        col2.metric("Prediction", "Attack" if lime_pred == 1 else "Normal")

        with st.spinner("Running LIME explanation..."):
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data        = X_train,
                feature_names        = FEATURE_COLS,
                class_names          = ['Normal', 'Attack'],
                mode                 = 'classification',
                discretize_continuous= True,
                random_state         = 42
            )
            lime_exp = lime_explainer.explain_instance(
                lime_instance,
                lime_model.predict_proba,
                num_features = num_features,
                labels       = [0, 1]
            )

        fig = lime_exp.as_pyplot_figure(label=int(lime_pred))
        plt.title(f'LIME — Instance #{lime_idx} | Pred: {"Attack" if lime_pred==1 else "Normal"}',
                  fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature weights table
        st.subheader("Feature Weights")
        lime_list = lime_exp.as_list(label=int(lime_pred))
        lime_df   = pd.DataFrame(lime_list, columns=['Feature Condition', 'Weight'])
        lime_df['Impact'] = lime_df['Weight'].apply(
            lambda w: '🔴 Attack' if w > 0 else '🟢 Normal'
        )
        lime_df['Weight'] = lime_df['Weight'].round(4)
        st.dataframe(lime_df, use_container_width=True)
