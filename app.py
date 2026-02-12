# app.py
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# Pleine largeur
st.set_page_config(layout="wide")


class PositiveClipper(BaseEstimator, TransformerMixin):
    """
    Transformateur perso utilisé dans le pipeline pour :
    - Remplacer inf / NaN par 0
    - Clipper les valeurs négatives à 0 (avant log)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.where(np.isfinite(X), X, 0)
        return np.clip(X, a_min=0, a_max=None)


def set_bg_from_local(image_path: str):
    """Fond d'écran + voile de lisibilité."""
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        position: relative;
        background-color: rgba(255, 255, 255, 0.80);
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        opacity: .30;
        background-color: rgba(255, 255, 255, 0.80);
        z-index: 0;
        pointer-events: none;
    }}

    .stApp > * {{
        position: relative;
        z-index: 1;
        color: #000;
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Fond
set_bg_from_local("business_fond_bank.png")

# -----------------------------
# CSS Split-screen PRO (2 blocs réels)
# - style appliqué aux colonnes Streamlit (PAS de <div> wrapper)
# - glassmorphism
# - hover
# - séparation verticale
# - bouton stylisé
# -----------------------------
BLOCK_HEIGHT = "85vh"

st.markdown(f"""
<style>
/* Pleine largeur */
.block-container {{
    max-width: 100% !important;
    padding-left: 2rem;
    padding-right: 2rem;
}}

/* Wrapper intérieur de chaque colonne (c'est LUI qu'on transforme en "carte") */
div[data-testid="column"] > div {{
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(255,255,255,0.30);
    border-radius: 22px;
    padding: 24px;
    min-height: {BLOCK_HEIGHT};
    box-shadow: 0 12px 34px rgba(0,0,0,0.18);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, background .18s ease;
}}

/* Hover sur la "carte" colonne */
div[data-testid="column"] > div:hover {{
    transform: translateY(-3px);
    box-shadow: 0 16px 44px rgba(0,0,0,0.22);
    border-color: rgba(255,255,255,0.40);
    background: rgba(255,255,255,0.32);
}}

/* Séparation verticale subtile */
div[data-testid="column"]:nth-of-type(1) > div {{
    border-right: 1px solid rgba(255,255,255,0.20);
}}

/* Titres (Streamlit header/subheader) un peu plus visibles */
h2, h3 {{
    margin-top: 0.2rem;
}}

/* Bouton "Prédire" stylisé */
div[data-testid="stFormSubmitButton"] button {{
    width: 100%;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    font-weight: 800 !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    background: linear-gradient(90deg, rgba(34,197,94,0.95), rgba(16,185,129,0.95)) !important;
    color: white !important;
    box-shadow: 0 10px 22px rgba(16,185,129,0.25) !important;
    transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
}}
div[data-testid="stFormSubmitButton"] button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(16,185,129,0.35) !important;
    filter: brightness(1.02);
}}
div[data-testid="stFormSubmitButton"] button:active {{
    transform: translateY(0px);
    box-shadow: 0 8px 16px rgba(16,185,129,0.25) !important;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI
# -----------------------------
st.title("Prédiction de souscription bancaire")
st.markdown("Modèle basé sur pipeline : log1p + binarisation + OneHot + standardisation")

model = joblib.load("model_pipeline.pkl")

left, right = st.columns(2, gap="large")

# Formulaire à gauche
with left:
    st.header("🧾 Paramètres client")

    with st.form("form_client"):
        col1, col2, col3 = st.columns(3)
        with col1:
            duration = st.slider("Durée de l'appel (sec)", 0, 5000, 180)
        with col2:
            balance = st.number_input("Solde moyen (€)", -2000, 100000, 1000)
        with col3:
            campaign = st.slider("Contacts campagne", 1, 50, 1)

        col4, col5 = st.columns(2)
        with col4:
            pdays = st.slider("Jours depuis dernier contact", -1, 999, -1)
        with col5:
            previous = st.slider("Nb de contacts précédents", 0, 100, 0)

        col6, col7 = st.columns(2)
        with col6:
            job = st.selectbox("Profession", [
                'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'
            ])
            contact = st.selectbox("Type de contact", ['cellular', 'telephone', 'unknown'])
            poutcome = st.selectbox("Résultat précédente", ['success', 'failure', 'other', 'non_contacté'])

        with col7:
            education = st.selectbox("Éducation", ['primary', 'secondary', 'tertiary'])
            month = st.selectbox("Mois du contact", [
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                'aug', 'sep', 'oct', 'nov', 'dec'
            ])
            cluster = st.selectbox("Cluster client", [0, 1, 2])

        submitted = st.form_submit_button("🎯 Prédire")

# Etat
if "has_pred" not in st.session_state:
    st.session_state.has_pred = False
if submitted:
    st.session_state.has_pred = True

# Résultat à droite
with right:
    st.header("📈 Résultat")

    if st.session_state.has_pred:
        client_input = pd.DataFrame([{
            'duration': duration,
            'balance': balance,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'job': job,
            'education': education,
            'contact': contact,
            'month': month,
            'poutcome': poutcome,
            'cluster': cluster
        }])

        proba = model.predict_proba(client_input)[0, 1]
        pred = model.predict(client_input)[0]

        st.metric("Probabilité de souscription à des dépôts à terme", f"{proba*100:.2f} %")

        if pred == 1:
            st.success("✅ Le client est susceptible de souscrire.")
        else:
            st.warning("❌ Le client ne semble pas intéressé.")

        import plotly.graph_objects as go

        p_pct = round(float(proba) * 100, 1)
        theme_base = st.get_option("theme.base")
        plotly_template = "plotly_dark" if theme_base == "dark" else "plotly_white"
        main_color = "green" if pred == 1 else "red"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p_pct,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": main_color},
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.25)"},
                    {"range": [40, 70], "color": "rgba(234,179,8,0.25)"},
                    {"range": [70, 100], "color": "rgba(34,197,94,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "white" if theme_base == "dark" else "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50
                }
            },
            title={"text": "Niveau de probabilité de souscription"}
        ))

        fig.update_layout(
            template=plotly_template,
            height=320,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Remplis les paramètres puis clique sur **🎯 Prédire** pour afficher la prédiction.")
