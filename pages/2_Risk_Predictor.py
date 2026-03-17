# ── PAGE 2: PERSONAL RISK PREDICTOR ──────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf       = joblib.load("model/rf_model.joblib")
    metadata = joblib.load("model/model_metadata.joblib")
    return rf, metadata

rf, metadata = load_model()

OBESITY_ORDER = metadata['obesity_order']
FEATURES      = metadata['features']

OBESITY_COLORS = {
    'Insufficient_Weight': '#95D5B2',
    'Normal_Weight':       '#52B788',
    'Overweight_Level_I':  '#F4A261',
    'Overweight_Level_II': '#E07B39',
    'Obesity_Type_I':      '#E63946',
    'Obesity_Type_II':     '#C1121F',
    'Obesity_Type_III':    '#7D0000',
}

RISK_LEVEL = {
    'Insufficient_Weight': ('⚠️ Underweight',    'warning'),
    'Normal_Weight':       ('✅ Normal Weight',   'safe'),
    'Overweight_Level_I':  ('🟡 Overweight I',   'warning'),
    'Overweight_Level_II': ('🟠 Overweight II',  'warning'),
    'Obesity_Type_I':      ('🔴 Obesity Type I', 'danger'),
    'Obesity_Type_II':     ('🔴 Obesity Type II','danger'),
    'Obesity_Type_III':    ('🔴 Obesity Type III','danger'),
}

RECOMMENDATIONS = {
    'Insufficient_Weight': [
        "Increase caloric intake with nutrient-dense foods",
        "Add strength training 3x per week to build muscle mass",
        "Consult a nutritionist for a personalized weight-gain plan",
    ],
    'Normal_Weight': [
        "Maintain your current lifestyle — you're in a healthy range",
        "Keep physical activity at 3–4 days per week",
        "Regular check-ups to monitor long-term trends",
    ],
    'Overweight_Level_I': [
        "Aim for 150 min of moderate exercise per week",
        "Reduce processed food and increase vegetable intake",
        "Small changes compound — even 5% weight reduction improves markers",
    ],
    'Overweight_Level_II': [
        "Consult a healthcare provider for a structured weight management plan",
        "Daily physical activity of at least 30 minutes",
        "Track meals — awareness is the first step to change",
    ],
    'Obesity_Type_I': [
        "Medical supervision recommended for safe weight loss",
        "Combine dietary changes with progressive physical activity",
        "Behavioral therapy has strong evidence for sustained results",
    ],
    'Obesity_Type_II': [
        "Consult a specialist — this level carries significant health risk",
        "Consider structured clinical weight management programs",
        "Focus on sustainable habit change, not rapid weight loss",
    ],
    'Obesity_Type_III': [
        "Immediate medical consultation strongly recommended",
        "Explore all options with your doctor — lifestyle, pharmacological, surgical",
        "Support networks and professional guidance are essential",
    ],
}

# ── PAGE HEADER ───────────────────────────────────────────────
st.markdown("# 🔮 Personal Risk Predictor")
st.markdown("### Answer 8 lifestyle questions — get your risk profile")
st.markdown("---")

st.markdown("""
<div style='background:#1A1D23; border-radius:8px; padding:12px 20px;
            border-left:3px solid #52B788; margin-bottom:24px;'>
    <b style='color:#52B788;'>How this works:</b> 
    Adjust the sliders to match your lifestyle. 
    The Random Forest model (96.4% accuracy) will predict your obesity 
    risk category based on patterns learned from 2,111 individuals.
</div>
""", unsafe_allow_html=True)

# ── INPUT SLIDERS ─────────────────────────────────────────────
st.markdown("### Your Lifestyle Profile")

col1, col2 = st.columns(2)

with col1:
    age    = st.slider("🎂 Age",           14,  61, 25)
    height = st.slider("📏 Height (m)",    1.45, 1.98, 1.70, step=0.01)
    weight = st.slider("⚖️ Weight (kg)",   39,  173, 70)
    fcvc   = st.slider("🥦 Vegetable consumption (1=low · 3=high)",
                       1.0, 3.0, 2.0, step=0.1)

with col2:
    ncp    = st.slider("🍽️ Meals per day",           1.0, 4.0, 3.0, step=0.5)
    ch2o   = st.slider("💧 Water intake (1=low · 3=high)",
                       1.0, 3.0, 2.0, step=0.1)
    faf    = st.slider("🏃 Physical activity (days/week)", 0.0, 3.0, 1.0, step=0.5)
    tue    = st.slider("📱 Screen time (0=low · 2=high)",  0.0, 2.0, 1.0, step=0.1)

# ── PREDICT ───────────────────────────────────────────────────
input_data = np.array([[age, height, weight, fcvc, ncp, ch2o, faf, tue]])
prediction = rf.predict(input_data)[0]
proba      = rf.predict_proba(input_data)[0]
proba_dict = dict(zip(rf.classes_, proba))

risk_label, risk_class = RISK_LEVEL[prediction]

st.markdown("---")
st.markdown("### Your Result")

# Risk badge + main metric
col_res1, col_res2 = st.columns([1, 2])

with col_res1:
    color = OBESITY_COLORS[prediction]
    st.markdown(f"""
    <div style='background:#1A1D23; border-radius:16px; padding:32px 24px;
                border: 2px solid {color}; text-align:center;'>
        <div style='font-size:13px; color:#888; margin-bottom:8px;
                    letter-spacing:0.1em; text-transform:uppercase;'>
            Predicted Category
        </div>
        <div style='font-size:22px; font-weight:700; color:{color};
                    margin-bottom:4px;'>
            {prediction.replace("_", " ")}
        </div>
        <div style='font-size:14px; color:#aaa;'>{risk_label}</div>
        <div style='margin-top:16px; font-size:28px; font-weight:800;
                    color:{color};'>
            {proba_dict[prediction]*100:.1f}%
        </div>
        <div style='font-size:12px; color:#666;'>model confidence</div>
    </div>
    """, unsafe_allow_html=True)

with col_res2:
    # Probability bar chart for all classes
    proba_df = pd.DataFrame({
        'Category':    OBESITY_ORDER,
        'Probability': [proba_dict.get(c, 0) * 100 for c in OBESITY_ORDER]
    })

    fig_proba = go.Figure(go.Bar(
        x     = proba_df['Probability'],
        y     = proba_df['Category'],
        orientation = 'h',
        marker_color = [
            OBESITY_COLORS[c] if c == prediction
            else '#2D3748'
            for c in OBESITY_ORDER
        ],
        text  = proba_df['Probability'].apply(lambda x: f"{x:.1f}%"),
        textposition = 'outside'
    ))
    fig_proba.update_layout(
        height        = 300,
        margin        = dict(l=0, r=60, t=10, b=0),
        paper_bgcolor = '#1A1D23',
        plot_bgcolor  = '#1A1D23',
        xaxis = dict(range=[0, 105], showgrid=False),
        yaxis = dict(categoryorder='array',
                     categoryarray=OBESITY_ORDER[::-1]),
        template = 'plotly_dark'
    )
    st.plotly_chart(fig_proba, use_container_width=True)

# ── GAUGE CHART ───────────────────────────────────────────────
risk_score = OBESITY_ORDER.index(prediction) / (len(OBESITY_ORDER) - 1)

fig_gauge = go.Figure(go.Indicator(
    mode  = "gauge+number",
    value = round(risk_score * 100, 1),
    title = dict(text="Risk Score", font=dict(color='#aaa', size=14)),
    gauge = dict(
        axis       = dict(range=[0, 100],
                          tickcolor='#444',
                          tickfont=dict(color='#666')),
        bar        = dict(color=OBESITY_COLORS[prediction]),
        bgcolor    = '#2D3748',
        bordercolor = '#444',
        steps = [
            dict(range=[0,  20],  color='rgba(45,106,79,0.2)'),
            dict(range=[20, 40],  color='rgba(82,183,136,0.2)'),
            dict(range=[40, 60],  color='rgba(244,162,97,0.2)'),
            dict(range=[60, 80],  color='rgba(230,57,70,0.2)'),
            dict(range=[80, 100], color='rgba(125,0,0,0.2)'),
        ],
        threshold = dict(
            line  = dict(color='white', width=2),
            value = round(risk_score * 100, 1)
        )
    ),
    number = dict(suffix="%", font=dict(color='white', size=32))
))
fig_gauge.update_layout(
    height        = 250,
    margin        = dict(l=20, r=20, t=40, b=20),
    paper_bgcolor = '#1A1D23',
    font          = dict(color='white')
)
st.plotly_chart(fig_gauge, use_container_width=True)

# ── RECOMMENDATIONS ───────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Personalised Recommendations")

recs = RECOMMENDATIONS[prediction]
for i, rec in enumerate(recs, 1):
    st.markdown(f"""
    <div style='background:#1A1D23; border-radius:8px; padding:14px 20px;
                border-left:3px solid {OBESITY_COLORS[prediction]};
                margin-bottom:10px;'>
        <b style='color:{OBESITY_COLORS[prediction]};'>{i}.</b> {rec}
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:20px; padding:12px 20px; background:#111;
            border-radius:8px; font-size:12px; color:#555;'>
    ⚠️ This tool is for educational and portfolio purposes only.
    It is not a medical device and does not provide clinical advice.
    Always consult a healthcare professional for health decisions.
</div>
""", unsafe_allow_html=True)