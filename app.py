# ── APP.PY — MAIN ENTRY POINT ─────────────────────────────────
import streamlit as st

st.set_page_config(
    page_title = "Obesity Risk Analytics",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── GLOBAL STYLES ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0E1117; }

    /* Cards */
    .metric-card {
        background-color: #1A1D23;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2D6A4F33;
        margin-bottom: 12px;
    }

    /* Section headers */
    .section-title {
        color: #52B788;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1D23;
        border-right: 1px solid #2D6A4F44;
    }

    /* Plotly chart background */
    .js-plotly-plot { border-radius: 12px; }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.05em;
    }
    .risk-safe    { background: #2D6A4F33; color: #52B788; border: 1px solid #52B788; }
    .risk-warning { background: #F4A26133; color: #F4A261; border: 1px solid #F4A261; }
    .risk-danger  { background: #E6394633; color: #E63946; border: 1px solid #E63946; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Obesity Risk")
    st.markdown("**Analytics & Prediction App**")
    st.markdown("---")
    st.markdown("""
    Navigate using the pages above:

    📊 **Data Explorer**  
    Explore lifestyle patterns across obesity categories

    🔮 **Risk Predictor**  
    Get your personal obesity risk prediction

    📈 **Model Performance**  
    Understand how the model works
    """)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#666;'>
    Dataset: 2,111 individuals<br>
    Mexico · Peru · Colombia<br>
    Source: UCI ML Repository<br>
    Model: Random Forest · 96.37%
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE ─────────────────────────────────────────────────
st.markdown("# 🌿 Obesity Risk Analytics")
st.markdown("### From lifestyle data to actionable health insights")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <div class='section-title'>Dataset</div>
        <div style='font-size:28px; font-weight:700; color:#52B788;'>2,111</div>
        <div style='font-size:13px; color:#888;'>individuals profiled</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <div class='section-title'>Variables</div>
        <div style='font-size:28px; font-weight:700; color:#52B788;'>17</div>
        <div style='font-size:13px; color:#888;'>lifestyle features</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <div class='section-title'>Model Accuracy</div>
        <div style='font-size:28px; font-weight:700; color:#52B788;'>96.4%</div>
        <div style='font-size:13px; color:#888;'>Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='metric-card'>
        <div class='section-title'>Risk Classes</div>
        <div style='font-size:28px; font-weight:700; color:#52B788;'>7</div>
        <div style='font-size:13px; color:#888;'>from insufficient to obese III</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<br>
<div style='background:#1A1D23; border-radius:12px; padding:24px;
            border-left: 3px solid #52B788;'>
    <b style='color:#52B788;'>Business context</b><br><br>
    Health insurers, wellness apps, and preventive care platforms need to
    assess risk from lifestyle data alone — no blood tests, no clinical visits.
    This app demonstrates how 8 survey questions can predict obesity risk
    category with 96% accuracy, enabling real-time personalization at scale.
</div>
""", unsafe_allow_html=True)