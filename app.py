# app.py
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import base64

from sklearn.base import BaseEstimator, TransformerMixin

class PositiveClipper(BaseEstimator, TransformerMixin):
    """
    Ce transformateur personnalisé est utilisé dans le pipeline pour :
    - Nettoyer les données avant une transformation logarithmique
    - Remplacer les valeurs infinies ou manquantes par 0
    - Éliminer les valeurs négatives en les 'clippant' à 0
    """    
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.where(np.isfinite(X), X, 0)
        return np.clip(X, a_min=0, a_max=None)


def set_bg_from_local(image_path):
    """Affiche une image de fond éclaircie pour améliorer la lisibilité"""
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
    }}

    /* Overlay éclaircissant */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.6);  /* Blanc semi-transparent */
        z-index: 0;
    }}

    /* S'assurer que tout le contenu passe au-dessus */
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Titre
st.title(" Prédiction de souscription bancaire")
st.markdown("Modèle basé sur pipeline : log1p + binarisation + OneHot + standardisation")

# Chargement du pipeline entraîné
model = joblib.load("model_pipeline.pkl")  

# Entrée utilisateur
st.header("🧾 Paramètres client")

# Groupe 1 : variables numériques
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

# Groupe 2 : variables catégorielles
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

# Données réunies dans un DataFrame
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

# Prédiction
proba = model.predict_proba(client_input)[0, 1]
pred = model.predict(client_input)[0]

# Affichage
st.subheader("📈 Résultat")
st.metric("Probabilité de souscription à des dépôts à terme", f"{proba*100:.2f} %")
if pred == 1:
    st.success("✅ Le client est susceptible de souscrire.")
else:
    st.warning("❌ Le client ne semble pas intéressé.")

# Visualisation simple (jauge)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.barh(["Client"], [proba], color="green" if pred == 1 else "red")
ax.set_xlim(0, 1)
ax.set_title("Niveau de probabilité de souscription")
st.pyplot(fig)
