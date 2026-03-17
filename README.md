# 🌿 Obesity Risk Analytics App
### An interactive ML-powered health risk assessment tool

**Author:** Julia Anglada Lomaeva  
**Stack:** Python · Streamlit · scikit-learn · Plotly  
**Live app:** [obesity-risk-app.streamlit.app](LINK)  
**Part of:** [Obesity & Lifestyle Analytics](https://github.com/julia-anlo/obesity-lifestyle-analytics)

---

## What This Is

A deployed web application that turns a 9-repo statistical analysis
into an interactive product usable by anyone — no code required.

Built on top of the Random Forest model trained in
[Repo 7](https://github.com/julia-anlo/random-forest-obesity)
(96.37% accuracy · 2,111 individuals · 8 lifestyle variables).

---

## Three Pages

### 📊 Data Explorer
Filter the dataset by obesity category, age range, and lifestyle variable.
Every chart updates in real time. Includes an auto-insight that summarises
the active selection in plain English.

### 🔮 Personal Risk Predictor
Adjust 8 lifestyle sliders → get your predicted obesity risk category,
model confidence per class, a risk gauge, and personalised recommendations.

### 📈 Model Performance
Confusion matrix, variable importance, and a full model comparison
across all 9 techniques in the pipeline.

---

## Business Context

Health insurers, wellness apps, and preventive care platforms need
to assess risk from lifestyle data alone — no blood tests, no clinical visits.

This app demonstrates how 8 survey questions can predict obesity risk
with 96% accuracy, enabling real-time personalization at scale.

**Interview question this answers:**
> "Can you take a model from a notebook to something a non-technical
> stakeholder can actually use?"

Yes.

---

## Run Locally
```bash
git clone https://github.com/julia-anlo/customer-health-risk-app
cd customer-health-risk-app

pip install -r requirements.txt

# Train and serialize the model first
python model/train_model.py

# Launch the app
streamlit run app.py
```

---

## Project Structure
```
customer-health-risk-app/
├── app.py                        ← home page + global styles
├── pages/
│   ├── 1_Data_Explorer.py        ← interactive EDA
│   ├── 2_Risk_Predictor.py       ← personal prediction
│   └── 3_Model_Performance.py    ← model evaluation
├── model/
│   ├── train_model.py            ← trains + serializes RF
│   ├── rf_model.joblib           ← serialized model
│   └── model_metadata.joblib     ← test results + feature names
├── data/
│   └── obesity.csv
├── requirements.txt
└── README.md
```

---

## Key Technical Decisions

| Decision | Why |
|---|---|
| `@st.cache_resource` for model | Loads once — not on every user interaction |
| `joblib` over `pickle` | Faster for numpy arrays, standard in sklearn ecosystem |
| RF on raw data (not scaled) | Matches Repo 7 — trees don't need scaling |
| `predict_proba()` not just `predict()` | Shows confidence, not just class — more useful for users |
| Plotly over matplotlib | Interactive charts render natively in Streamlit |

---

## Deploy on Streamlit Cloud

1. Push repo to GitHub (including `.joblib` files)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → `main` branch → `app.py`
4. Deploy — live URL in ~2 minutes