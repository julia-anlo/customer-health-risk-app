# ── PAGE 3: MODEL PERFORMANCE ─────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import joblib

# ── LOAD ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf       = joblib.load("model/rf_model.joblib")
    metadata = joblib.load("model/model_metadata.joblib")
    return rf, metadata

rf, metadata = load_model()

OBESITY_ORDER = metadata['obesity_order']
FEATURES      = metadata['features']
y_test        = metadata['y_test']
y_pred        = metadata['y_pred']
acc           = metadata['accuracy']

FEATURE_LABELS = {
    'Age':    'Age',
    'Height': 'Height',
    'Weight': 'Weight',
    'FCVC':   'Vegetable intake',
    'NCP':    'Meals/day',
    'CH2O':   'Water intake',
    'FAF':    'Physical activity',
    'TUE':    'Screen time',
}

# ── PAGE HEADER ───────────────────────────────────────────────
st.markdown("# 📈 Model Performance")
st.markdown("### How the Random Forest makes its predictions")
st.markdown("---")

# ── METRICS ROW ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='section-title'>Test Accuracy</div>
        <div style='font-size:28px; font-weight:700;
                    color:#52B788;'>{acc*100:.2f}%</div>
        <div style='font-size:12px; color:#666;'>on 634 held-out individuals</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='section-title'>Trees</div>
        <div style='font-size:28px; font-weight:700; color:#52B788;'>100</div>
        <div style='font-size:12px; color:#666;'>error stabilizes at ~30</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='section-title'>Features</div>
        <div style='font-size:28px; font-weight:700; color:#52B788;'>8</div>
        <div style='font-size:12px; color:#666;'>lifestyle variables</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    naive = max(pd.Series(y_test).value_counts()) / len(y_test)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='section-title'>vs Baseline</div>
        <div style='font-size:28px; font-weight:700;
                    color:#52B788;'>+{(acc - naive)*100:.1f}pp</div>
        <div style='font-size:12px; color:#666;'>above naive classifier</div>
    </div>
    """, unsafe_allow_html=True)

# ── CONFUSION MATRIX ──────────────────────────────────────────
st.markdown("#### Confusion Matrix")

short_labels = [o.replace('_', ' ') for o in OBESITY_ORDER]
cm = confusion_matrix(y_test, y_pred, labels=OBESITY_ORDER)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig_cm = go.Figure(go.Heatmap(
    z            = cm_pct,
    x            = short_labels,
    y            = short_labels,
    colorscale   = [[0, '#1A1D23'], [1, '#52B788']],
    text         = cm,
    texttemplate = "%{text}",
    textfont     = dict(size=12),
    showscale    = True,
    colorbar     = dict(title='% of actual')
))
fig_cm.update_layout(
    height        = 420,
    margin        = dict(l=0, r=0, t=20, b=60),
    paper_bgcolor = '#1A1D23',
    plot_bgcolor  = '#1A1D23',
    xaxis = dict(title='Predicted', tickangle=-20),
    yaxis = dict(title='Actual'),
    template      = 'plotly_dark'
)
st.plotly_chart(fig_cm, use_container_width=True)

# ── VARIABLE IMPORTANCE ───────────────────────────────────────
st.markdown("#### Variable Importance")

importances = rf.feature_importances_
feat_df = pd.DataFrame({
    'Feature':    [FEATURE_LABELS[f] for f in FEATURES],
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig_imp = px.bar(
    feat_df,
    x           = 'Importance',
    y           = 'Feature',
    orientation = 'h',
    template    = 'plotly_dark',
    color       = 'Importance',
    color_continuous_scale = [[0, '#2D6A4F'], [1, '#52B788']],
    text        = feat_df['Importance'].apply(lambda x: f"{x:.3f}")
)
fig_imp.update_layout(
    height        = 360,
    margin        = dict(l=0, r=60, t=20, b=0),
    paper_bgcolor = '#1A1D23',
    plot_bgcolor  = '#1A1D23',
    showlegend    = False,
    coloraxis_showscale = False
)
fig_imp.update_traces(textposition='outside')
st.plotly_chart(fig_imp, use_container_width=True)

# ── MODEL COMPARISON ──────────────────────────────────────────
st.markdown("#### Model Comparison — Full Pipeline")

models_data = {
    'Model': ['Naive Bayes', 'Decision Tree', 'KNN (k=1)',
              'RF w/o Weight', 'Logistic Regression',
              'Random Forest', 'SVM Linear'],
    'Accuracy': [66.25, 81.86, 87.07, 81.55, 85.80, 96.37, 97.48],
    'Type': ['Baseline', 'Interpretable', 'Distance-based',
             'Ablation', 'Interpretable',
             'Ensemble', 'Kernel-based']
}
comp_df = pd.DataFrame(models_data).sort_values('Accuracy')

fig_comp = px.bar(
    comp_df,
    x           = 'Accuracy',
    y           = 'Model',
    orientation = 'h',
    color       = 'Accuracy',
    color_continuous_scale = [[0, '#2D3748'], [0.5, '#2D6A4F'], [1, '#52B788']],
    text        = comp_df['Accuracy'].apply(lambda x: f"{x:.1f}%"),
    template    = 'plotly_dark',
    hover_data  = ['Type']
)
fig_comp.add_vline(x=90, line_dash='dot',
                   line_color='#52B78866',
                   annotation_text="90% threshold",
                   annotation_font_color='#52B788')
fig_comp.update_layout(
    height        = 380,
    margin        = dict(l=0, r=80, t=20, b=0),
    paper_bgcolor = '#1A1D23',
    plot_bgcolor  = '#1A1D23',
    coloraxis_showscale = False,
    xaxis = dict(range=[50, 105])
)
fig_comp.update_traces(textposition='outside')
st.plotly_chart(fig_comp, use_container_width=True)
```

---

## `requirements.txt`
```
streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.18.0
joblib>=1.3.0