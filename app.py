# app.py
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(
    layout="wide",
    page_title="BankScore — Prédiction bancaire",
    page_icon="🏦"
)


class PositiveClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.where(np.isfinite(X), X, 0)
        return np.clip(X, a_min=0, a_max=None)


def set_bg_from_local(image_path: str):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(
            135deg,
            rgba(10, 25, 60, 0.82) 0%,
            rgba(15, 52, 110, 0.75) 50%,
            rgba(10, 25, 60, 0.88) 100%
        );
        z-index: 0;
        pointer-events: none;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_bg_from_local("assets/business_fond_bank.png")

# ── Global CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 100% !important;
    padding: 2rem 2.5rem !important;
}

/* ── Page header ── */
.page-header {
    text-align: center;
    padding: 1.5rem 0 2rem;
}
.page-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    margin: 0 0 .5rem;
}
.page-header h1 span { color: #60a5fa; }
.page-header p {
    color: rgba(255,255,255,0.55);
    font-size: .95rem;
    font-weight: 400;
    margin: 0;
}
.header-badge {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    background: rgba(96,165,250,0.12);
    border: 1px solid rgba(96,165,250,0.25);
    color: #93c5fd;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: .3rem .9rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.header-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #60a5fa;
    box-shadow: 0 0 8px #60a5fa;
    animation: pulse 2s ease infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; }
    50%      { opacity:.3; }
}

/* ── Column cards ── */
div[data-testid="column"] > div {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 20px;
    padding: 2rem;
    min-height: 82vh;
    box-shadow:
        0 4px 6px rgba(0,0,0,0.07),
        0 20px 50px rgba(0,0,0,0.25),
        inset 0 1px 0 rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-testid="column"] > div:hover {
    border-color: rgba(96,165,250,0.2);
    box-shadow:
        0 4px 6px rgba(0,0,0,0.07),
        0 24px 60px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.10);
}

/* ── Section headers ── */
h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 1.25rem !important;
    padding-bottom: .75rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── Labels & text ── */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: rgba(255,255,255,0.75) !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    letter-spacing: .02em !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: .875rem !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: rgba(96,165,250,0.5) !important;
    box-shadow: 0 0 0 3px rgba(96,165,250,0.12) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: #3b82f6 !important;
}

/* ── Submit button ── */
div[data-testid="stFormSubmitButton"] button {
    width: 100% !important;
    border-radius: 12px !important;
    padding: .9rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: .95rem !important;
    letter-spacing: .02em !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    transition: all .2s ease !important;
    margin-top: .5rem;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    box-shadow: 0 8px 25px rgba(37,99,235,0.50) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0) !important;
}

/* ── Info / success / warning boxes ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    font-weight: 500 !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 14px;
    padding: 1.25rem 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.60) !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1.25rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(96,165,250,0.3);
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Page Header ──
st.markdown("""
<div class="page-header">
    <div class="header-badge">Modèle ML · Scoring Bancaire</div>
    <h1>Bank<span>Score</span> — Prédiction de souscription</h1>
    <p>Pipeline : log1p · binarisation · OneHot · standardisation · Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ──
model = joblib.load("models/model_pipeline.pkl")

left, right = st.columns(2, gap="large")

# ── Formulaire ──
with left:
    st.header("🧾 Paramètres client")

    with st.form("form_client"):
        col1, col2, col3 = st.columns(3)
        with col1:
            duration = st.slider("Durée appel (sec)", 0, 5000, 180)
        with col2:
            balance = st.number_input("Solde moyen (€)", -2000, 100000, 1000)
        with col3:
            campaign = st.slider("Contacts campagne", 1, 50, 1)

        col4, col5 = st.columns(2)
        with col4:
            pdays = st.slider("Jours depuis dernier contact", -1, 999, -1)
        with col5:
            previous = st.slider("Nb contacts précédents", 0, 100, 0)

        col6, col7 = st.columns(2)
        with col6:
            job = st.selectbox("Profession", [
                'admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                'management', 'retired', 'self-employed', 'services',
                'student', 'technician', 'unemployed'
            ])
            contact  = st.selectbox("Type de contact", ['cellular', 'telephone', 'unknown'])
            poutcome = st.selectbox("Résultat précédente", ['success', 'failure', 'other', 'non_contacté'])
        with col7:
            education = st.selectbox("Éducation", ['primary', 'secondary', 'tertiary'])
            month     = st.selectbox("Mois du contact", [
                'jan','feb','mar','apr','may','jun',
                'jul','aug','sep','oct','nov','dec'
            ])
            cluster = st.selectbox("Cluster client", [0, 1, 2])

        submitted = st.form_submit_button("🎯  Lancer la prédiction")

# ── Session state ──
if "has_pred" not in st.session_state:
    st.session_state.has_pred = False
if submitted:
    st.session_state.has_pred = True

# ── Résultat ──
with right:
    st.header("📈 Résultat de l'analyse")

    if st.session_state.has_pred:
        client_input = pd.DataFrame([{
            'duration': duration, 'balance': balance, 'campaign': campaign,
            'pdays': pdays, 'previous': previous, 'job': job,
            'education': education, 'contact': contact, 'month': month,
            'poutcome': poutcome, 'cluster': cluster
        }])

        proba = model.predict_proba(client_input)[0, 1]
        pred  = model.predict(client_input)[0]

        st.metric("Probabilité de souscription à un dépôt à terme", f"{proba*100:.2f} %")
        st.markdown("<hr>", unsafe_allow_html=True)

        if pred == 1:
            st.success("✅ **Le client est susceptible de souscrire.** Profil favorable détecté.")
        else:
            st.warning("❌ **Le client ne semble pas intéressé.** Probabilité insuffisante.")

        import plotly.graph_objects as go

        p_pct       = round(float(proba) * 100, 1)
        main_color  = "#3b82f6" if pred == 1 else "#ef4444"
        accent_color = "#60a5fa" if pred == 1 else "#f87171"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_pct,
            number={
                "suffix": "%",
                "font": {"size": 52, "color": accent_color, "family": "Syne"}
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "rgba(255,255,255,0.3)",
                    "tickfont": {"color": "rgba(255,255,255,0.5)", "size": 11}
                },
                "bar": {"color": main_color, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "rgba(239,68,68,0.12)"},
                    {"range": [40, 70], "color": "rgba(234,179,8,0.12)"},
                    {"range": [70,100], "color": "rgba(59,130,246,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "rgba(255,255,255,0.6)", "width": 2},
                    "thickness": 0.75,
                    "value": 50
                }
            },
            title={
                "text": "Niveau de probabilité",
                "font": {"size": 14, "color": "rgba(255,255,255,0.55)", "family": "Inter"}
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
            height=300,
            margin=dict(l=30, r=30, t=60, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Score bar ──
        st.markdown(f"""
        <div style="margin-top:.5rem;">
            <div style="display:flex;justify-content:space-between;
                        font-size:.78rem;color:rgba(255,255,255,0.5);margin-bottom:.4rem;">
                <span>0%</span><span>50%</span><span>100%</span>
            </div>
            <div style="background:rgba(255,255,255,0.08);border-radius:100px;height:8px;overflow:hidden;">
                <div style="
                    width:{p_pct}%;
                    height:100%;
                    background:linear-gradient(90deg, {main_color}, {accent_color});
                    border-radius:100px;
                    transition:width .6s ease;
                "></div>
            </div>
            <div style="text-align:right;font-size:.78rem;
                        color:{accent_color};font-weight:600;margin-top:.4rem;">
                {p_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; min-height:55vh; text-align:center; gap:1rem;
        ">
            <div style="font-size:3.5rem;">🏦</div>
            <div style="
                font-family:'Syne',sans-serif; font-size:1.3rem;
                font-weight:700; color:#ffffff;
            ">Prêt à analyser</div>
            <div style="color:rgba(255,255,255,0.45); font-size:.9rem; max-width:280px; line-height:1.6;">
                Remplis les paramètres client puis clique sur
                <strong style="color:#60a5fa;">Lancer la prédiction</strong>
                pour afficher le résultat.
            </div>
        </div>
        """, unsafe_allow_html=True)
