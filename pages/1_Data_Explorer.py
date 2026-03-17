# ── PAGE 1: DATA EXPLORER ─────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ────────────────────────────────────────────────────
TEAL  = ['#2D6A4F', '#40916C', '#52B788', '#74C69D', '#95D5B2',
         '#B7E4C7', '#D8F3DC']
PLOTLY_TEMPLATE = "plotly_dark"

OBESITY_ORDER = [
    'Insufficient_Weight', 'Normal_Weight',
    'Overweight_Level_I',  'Overweight_Level_II',
    'Obesity_Type_I',      'Obesity_Type_II', 'Obesity_Type_III'
]

OBESITY_COLORS = {
    'Insufficient_Weight': '#95D5B2',
    'Normal_Weight':       '#52B788',
    'Overweight_Level_I':  '#F4A261',
    'Overweight_Level_II': '#E07B39',
    'Obesity_Type_I':      '#E63946',
    'Obesity_Type_II':     '#C1121F',
    'Obesity_Type_III':    '#7D0000',
}

FEATURE_LABELS = {
    'Age':    'Age (years)',
    'Height': 'Height (m)',
    'Weight': 'Weight (kg)',
    'FCVC':   'Vegetable consumption (1–3)',
    'NCP':    'Meals per day',
    'CH2O':   'Water intake (1–3)',
    'FAF':    'Physical activity (days/week)',
    'TUE':    'Screen time (0–2)',
}

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/obesity.csv")
    df['NObeyesdad'] = pd.Categorical(
        df['NObeyesdad'], categories=OBESITY_ORDER, ordered=True
    )
    return df

df = load_data()

# ── PAGE HEADER ───────────────────────────────────────────────
st.markdown("# 📊 Data Explorer")
st.markdown("### Discover lifestyle patterns across obesity categories")
st.markdown("---")

# ── SIDEBAR FILTERS ───────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filters")

    selected_classes = st.multiselect(
        "Obesity category",
        options  = OBESITY_ORDER,
        default  = OBESITY_ORDER,
        help     = "Filter by one or more obesity categories"
    )

    age_range = st.slider(
        "Age range",
        min_value = int(df['Age'].min()),
        max_value = int(df['Age'].max()),
        value     = (int(df['Age'].min()), int(df['Age'].max()))
    )

    selected_feature = st.selectbox(
        "Feature to explore",
        options = list(FEATURE_LABELS.keys()),
        index   = 2,   # Weight by default
        format_func = lambda x: FEATURE_LABELS[x]
    )

    st.markdown("---")
    st.markdown(f"**{len(df)} total individuals**")

# ── APPLY FILTERS ─────────────────────────────────────────────
mask = (
    df['NObeyesdad'].isin(selected_classes) &
    df['Age'].between(age_range[0], age_range[1])
)
filtered = df[mask]

st.markdown(f"""
<div style='background:#1A1D23; border-radius:8px; padding:12px 20px;
            border-left:3px solid #52B788; margin-bottom:20px;'>
    <b style='color:#52B788;'>Active filter:</b>
    {len(filtered):,} individuals · {len(selected_classes)} categories · 
    Age {age_range[0]}–{age_range[1]}
</div>
""", unsafe_allow_html=True)

# ── ROW 1: CLASS DISTRIBUTION + FEATURE DISTRIBUTION ─────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Class Distribution")
    class_counts = (
        filtered['NObeyesdad']
        .value_counts()
        .reindex(OBESITY_ORDER)
        .dropna()
        .reset_index()
    )
    class_counts.columns = ['Category', 'Count']
    class_counts['Pct'] = (class_counts['Count'] /
                            class_counts['Count'].sum() * 100).round(1)

    fig_bar = px.bar(
        class_counts,
        x            = 'Count',
        y            = 'Category',
        orientation  = 'h',
        color        = 'Category',
        color_discrete_map = OBESITY_COLORS,
        text         = class_counts['Pct'].apply(lambda x: f"{x}%"),
        template     = PLOTLY_TEMPLATE,
    )
    fig_bar.update_layout(
        showlegend   = False,
        height       = 320,
        margin       = dict(l=0, r=20, t=20, b=0),
        paper_bgcolor = '#1A1D23',
        plot_bgcolor  = '#1A1D23',
        yaxis = dict(categoryorder='array',
                     categoryarray=OBESITY_ORDER[::-1])
    )
    fig_bar.update_traces(textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown(f"#### {FEATURE_LABELS[selected_feature]} by Category")
    fig_box = px.box(
        filtered,
        x        = 'NObeyesdad',
        y        = selected_feature,
        color    = 'NObeyesdad',
        color_discrete_map = OBESITY_COLORS,
        template = PLOTLY_TEMPLATE,
        category_orders = {'NObeyesdad': OBESITY_ORDER}
    )
    fig_box.update_layout(
        showlegend    = False,
        height        = 320,
        margin        = dict(l=0, r=0, t=20, b=60),
        paper_bgcolor = '#1A1D23',
        plot_bgcolor  = '#1A1D23',
        xaxis_tickangle = -30
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ── ROW 2: CORRELATION HEATMAP ────────────────────────────────
st.markdown("#### Correlation Heatmap — Numerical Features")

num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
corr     = filtered[num_cols].corr().round(2)

fig_heat = go.Figure(go.Heatmap(
    z            = corr.values,
    x            = corr.columns,
    y            = corr.index,
    colorscale   = [[0, '#E63946'], [0.5, '#1A1D23'], [1, '#52B788']],
    zmid         = 0,
    text         = corr.values,
    texttemplate = "%{text}",
    textfont     = dict(size=11),
    showscale    = True
))
fig_heat.update_layout(
    height        = 380,
    margin        = dict(l=0, r=0, t=20, b=0),
    paper_bgcolor = '#1A1D23',
    plot_bgcolor  = '#1A1D23',
    template      = PLOTLY_TEMPLATE,
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── ROW 3: SCATTER + TRANSPORT ────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Weight vs Height — Colored by Risk")
    fig_sc = px.scatter(
        filtered,
        x        = 'Height',
        y        = 'Weight',
        color    = 'NObeyesdad',
        color_discrete_map = OBESITY_COLORS,
        opacity  = 0.6,
        template = PLOTLY_TEMPLATE,
        category_orders = {'NObeyesdad': OBESITY_ORDER},
        hover_data = ['Age', 'FAF', 'TUE']
    )
    fig_sc.update_layout(
        height        = 360,
        margin        = dict(l=0, r=0, t=20, b=0),
        paper_bgcolor = '#1A1D23',
        plot_bgcolor  = '#1A1D23',
        legend        = dict(font=dict(size=10))
    )
    st.plotly_chart(fig_sc, use_container_width=True)

with col4:
    st.markdown("#### Transport Mode vs Obesity Category")
    if 'MTRANS' in filtered.columns:
        transport_counts = (
            filtered.groupby(['MTRANS', 'NObeyesdad'])
            .size()
            .reset_index(name='Count')
        )
        fig_tr = px.bar(
            transport_counts,
            x        = 'MTRANS',
            y        = 'Count',
            color    = 'NObeyesdad',
            color_discrete_map = OBESITY_COLORS,
            template = PLOTLY_TEMPLATE,
            barmode  = 'stack',
            category_orders = {'NObeyesdad': OBESITY_ORDER}
        )
        fig_tr.update_layout(
            height        = 360,
            margin        = dict(l=0, r=0, t=20, b=60),
            paper_bgcolor = '#1A1D23',
            plot_bgcolor  = '#1A1D23',
            xaxis_tickangle = -20,
            legend = dict(font=dict(size=10))
        )
        st.plotly_chart(fig_tr, use_container_width=True)

# ── AUTO INSIGHT ──────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💡 Auto Insight")

if len(filtered) > 0:
    obese_pct = (
        filtered['NObeyesdad']
        .isin(['Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'])
        .mean() * 100
    )
    avg_activity = filtered['FAF'].mean()
    avg_screen   = filtered['TUE'].mean()
    top_transport = (filtered['MTRANS'].value_counts().index[0]
                     if 'MTRANS' in filtered.columns else 'N/A')

    st.markdown(f"""
    <div style='background:#1A1D23; border-radius:12px; padding:20px 24px;
                border-left:3px solid #52B788;'>
        In the current selection of <b style='color:#52B788;'>{len(filtered):,} individuals</b>:
        <ul style='margin-top:10px; color:#ccc;'>
            <li><b style='color:#52B788;'>{obese_pct:.1f}%</b> fall into an obesity category 
                (Type I, II, or III)</li>
            <li>Average physical activity: 
                <b style='color:#52B788;'>{avg_activity:.2f} days/week</b></li>
            <li>Average screen time score: 
                <b style='color:#52B788;'>{avg_screen:.2f}</b></li>
            <li>Most common transport: 
                <b style='color:#52B788;'>{top_transport}</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)